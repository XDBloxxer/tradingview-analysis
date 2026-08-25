#!/usr/bin/env python3
"""
diagnose_symbol_fingerprint_leak.py
====================================
Follow-up to diagnose_feature_leakage.py, specifically for the HV_10/20/30
family (and any other --features you pass in).

diagnose_feature_leakage.py computes univariate AUC in-sample, with no
symbol purge -- so it cannot tell the difference between:

  (a) real signal: historical volatility genuinely predicts a >=20%
      same-day move, or
  (b) symbol-fingerprint leak: hv_30 is highly autocorrelated per-symbol
      (see synthetic_leak_test.py), so if the winners screener has any
      per-symbol selection bias, HV looks predictive purely by
      re-identifying which stock a row belongs to, not by forecasting
      that day's move.

This script decomposes each feature into a between-symbol component and a
within-symbol component and scores each against the label separately:

  - between_symbol_auc: AUC of (per-symbol MEAN of the feature, broadcast
    to every row of that symbol) vs label. This is exactly the
    "does re-identifying the symbol explain the label" signal -- a row's
    score here only depends on which symbol it belongs to, not on that
    day's value.
  - within_symbol_auc: AUC of (feature value minus its own symbol's mean,
    i.e. day-to-day deviation) vs label. This is the signal left over
    once symbol identity is held constant -- it can only reflect real
    day-to-day variation, not fingerprinting.

Interpretation (per the synthetic_leak_test.py logic, applied to real
data instead of synthetic):
  - between_symbol_auc high, within_symbol_auc ~0.5   -> fingerprint leak.
    The label rate tracks *which symbol it is*, not that day's HV. Block
    or symbol-demean the feature.
  - within_symbol_auc clearly above 0.5 (regardless of between_symbol_auc)
    -> real day-to-day signal exists. Likely legitimate, safe to keep
    (though the between-symbol part may still want demeaning).
  - both ~0.5 -> feature isn't doing much either way.

Also reports each symbol's row count (n_appearances), since a handful of
heavily-repeated symbols dominating the between-symbol estimate is itself
a red flag worth knowing about, independent of the AUC numbers.

Usage (from repo root, same env as diagnose_feature_leakage.py):

    python diagnose_symbol_fingerprint_leak.py
    python diagnose_symbol_fingerprint_leak.py --features t1_close_HV_10 t1_close_HV_20 t1_close_HV_30
    python diagnose_symbol_fingerprint_leak.py --min-symbol-rows 3
    python diagnose_symbol_fingerprint_leak.py --model-features
    python diagnose_symbol_fingerprint_leak.py --all-features

By default this scans every t1_*/t3_/t5_/t10_ HV_10/HV_20/HV_30 variant
(open+close, every lag) plus BBB_20_2.0_2.0 (bandwidth), matching the
family flagged in the latest diagnose_feature_leakage.py run. Pass
--features to scan a different set instead, --model-features to test
exactly the live model's current selected-feature set (read from
model_metadata.json's "features" key -- see --model-metadata-file to
point at a different metadata file), or --all-features to exhaustively
scan every column in the prepared feature matrix (the full pre-selection
pool, typically ~300-400 columns -- slow, but useful for a full audit).
Precedence when more than one is given: --features > --model-features >
--all-features > the HV/BBB-only default.

Applies the same manual feature blocklist as diagnose_feature_leakage.py
by default, for consistency -- but since the whole point here is to
inspect specific (possibly still-unblocked) features, --features bypasses
the blocklist check for names you pass explicitly.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import ml_retrain_model as rt  # noqa: E402
from src.ml_predictor.feature_selection import (  # noqa: E402
    DEFAULT_EXCLUDED_FEATURES_PATH,
    apply_feature_exclusions,
    load_excluded_features,
)

DEFAULT_TARGET_BASES = ("HV_10", "HV_20", "HV_30", "BBB_20_2.0_2.0")
LAG_PREFIXES = ("t1_close_", "t1_open_", "t3_", "t5_", "t10_")
DEFAULT_MODEL_METADATA_PATH = "ml_models/model_metadata.json"


def _load_model_features(metadata_path: str) -> list[str]:
    """Read the live model's selected feature list straight out of
    model_metadata.json ("features" key), so this script can be pointed at
    exactly what production is using instead of only the HV/BBB family."""
    import json
    path = Path(metadata_path)
    if not path.exists():
        print(f"ERROR: {metadata_path} not found -- cannot load model feature "
              "list. Pass --features explicitly instead.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        meta = json.load(f)
    feats = meta.get("features")
    if not feats:
        print(f"ERROR: no 'features' key found in {metadata_path}.", file=sys.stderr)
        sys.exit(1)
    return feats


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--features", nargs="+", default=None,
        help="Exact column name(s) to test. Default: every lag/side variant "
             f"of {DEFAULT_TARGET_BASES}.",
    )
    p.add_argument(
        "--model-features", action="store_true",
        help="Test exactly the live model's current feature set (read from "
             "--model-metadata-file's 'features' key) instead of the "
             "HV/BBB-only default. Overridden by --features if both are given.",
    )
    p.add_argument(
        "--model-metadata-file", default=DEFAULT_MODEL_METADATA_PATH,
        help=f"Path to model_metadata.json, used with --model-features. "
             f"Default: {DEFAULT_MODEL_METADATA_PATH}",
    )
    p.add_argument(
        "--all-features", action="store_true",
        help="Test every column in the prepared feature matrix X (the full "
             "pre-selection pool, typically ~300-400 columns), not just the "
             "HV/BBB family or the live model's subset. Slow but exhaustive. "
             "Overridden by --features/--model-features if given.",
    )
    p.add_argument(
        "--exclude-features-file",
        default=DEFAULT_EXCLUDED_FEATURES_PATH,
        help="JSON blocklist file, applied before auto-discovering columns "
             "(no effect on names passed via --features). "
             f"Default: {DEFAULT_EXCLUDED_FEATURES_PATH}",
    )
    p.add_argument(
        "--no-exclude-features", action="store_true",
        help="Don't apply the blocklist when auto-discovering columns.",
    )
    p.add_argument(
        "--min-symbol-rows", type=int, default=2,
        help="Drop symbols with fewer than this many train-only rows before "
             "computing within-symbol deviation (need >=2 rows per symbol "
             "for a within-symbol comparison to mean anything). Default: 2.",
    )
    return p.parse_args()


def decompose(feature: pd.Series, symbol: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split `feature` into (per-symbol mean broadcast to rows, within-symbol
    deviation from that mean), both aligned to feature's index."""
    symbol_mean = feature.groupby(symbol).transform("mean")
    within = feature - symbol_mean
    return symbol_mean, within


def safe_auc(y: pd.Series, scores: pd.Series) -> float | None:
    scores = scores.fillna(scores.median())
    if scores.nunique() < 2:
        return None
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        return None
    return max(auc, 1 - auc)  # direction-agnostic, matches diagnose_feature_leakage.py


def main():
    args = _parse_args()
    lookback_days = int(os.environ.get("LOOKBACK_DAYS", "365")) or None
    client = rt.get_supabase_client()
    base_df = rt.load_base_training_data(client, lookback_days=lookback_days)
    t1_df = rt.load_t1_data(client, lookback_days=lookback_days)
    combined_df = rt.combine_datasets(base_df, t1_df)
    X, y, _w = rt.prepare_features(combined_df)

    if "symbol" not in combined_df.columns:
        print("ERROR: 'symbol' column not present on combined_df -- cannot "
              "run symbol-grouped diagnostics.", file=sys.stderr)
        sys.exit(1)
    symbol_all = combined_df["symbol"]

    date_col = "detection_date" if "detection_date" in combined_df.columns else "event_date"
    dates = combined_df[date_col] if date_col in combined_df.columns else pd.Series(pd.NaT, index=combined_df.index)
    cutoff = rt._compute_val_cutoff(combined_df)
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    train_mask = parsed_dates.isna() | (parsed_dates < cutoff)

    X = X.loc[train_mask].reset_index(drop=True)
    y = y.loc[train_mask].reset_index(drop=True)
    symbol = symbol_all.loc[train_mask].reset_index(drop=True)
    print(f"Train-only rows: {len(X)}  (cutoff={cutoff.date()})  "
          f"pos={int(y.sum())} neg={int((y == 0).sum())}  "
          f"unique symbols={symbol.nunique()}\n")

    # Optional: drop rows for symbols that barely repeat -- within-symbol
    # deviation is meaningless with only 1 row.
    counts = symbol.value_counts()
    keep_symbols = counts[counts >= args.min_symbol_rows].index
    row_mask = symbol.isin(keep_symbols)
    n_dropped = (~row_mask).sum()
    if n_dropped:
        print(f"[filter] dropping {n_dropped} row(s) whose symbol has < "
              f"{args.min_symbol_rows} train-only appearances "
              f"({symbol.nunique() - len(keep_symbols)} symbol(s) affected)\n")
    X, y, symbol = X.loc[row_mask].reset_index(drop=True), y.loc[row_mask].reset_index(drop=True), symbol.loc[row_mask].reset_index(drop=True)

    # Resolve which columns to test. Priority: --features > --model-features
    # > --all-features > the original HV/BBB-only default.
    if args.features:
        target_cols = [c for c in args.features if c in X.columns]
        missing = [c for c in args.features if c not in X.columns]
        if missing:
            print(f"[warn] not present in feature matrix, skipping: {missing}\n")
    elif args.model_features:
        model_feats = _load_model_features(args.model_metadata_file)
        target_cols = [c for c in model_feats if c in X.columns]
        missing = [c for c in model_feats if c not in X.columns]
        print(f"[model-features] loaded {len(model_feats)} feature(s) from "
              f"{args.model_metadata_file}")
        if missing:
            print(f"[warn] not present in feature matrix, skipping: {missing}\n")
    elif args.all_features:
        if args.no_exclude_features:
            scan_X = X
        else:
            exclude_features, exclude_base_features = load_excluded_features(args.exclude_features_file)
            scan_X = apply_feature_exclusions(X, exclude_features, exclude_base_features)
        target_cols = list(scan_X.columns)
    else:
        if args.no_exclude_features:
            scan_X = X
        else:
            exclude_features, exclude_base_features = load_excluded_features(args.exclude_features_file)
            scan_X = apply_feature_exclusions(X, exclude_features, exclude_base_features)
        target_cols = [
            c for c in scan_X.columns
            if any(
                c[len(pfx):].upper() == base.upper()
                for pfx in LAG_PREFIXES if c.startswith(pfx)
                for base in DEFAULT_TARGET_BASES
            )
        ]

    if not target_cols:
        print("No target columns found to test. Pass --features explicitly, "
              "or --no-exclude-features if they're being blocked.")
        return

    print(f"Testing {len(target_cols)} column(s): {target_cols}\n")

    rows = []
    for col in target_cols:
        feat = pd.to_numeric(X[col], errors="coerce")
        if feat.notna().sum() < 20:
            print(f"[skip] {col}: fewer than 20 non-NaN values in train-only slice")
            continue

        symbol_mean, within = decompose(feat, symbol)

        raw_auc = safe_auc(y, feat)
        between_auc = safe_auc(y, symbol_mean)
        within_auc = safe_auc(y, within)

        rows.append({
            "feature": col,
            "raw_auc": raw_auc,
            "between_symbol_auc": between_auc,
            "within_symbol_auc": within_auc,
            "n_nonnull": int(feat.notna().sum()),
        })

    report = pd.DataFrame(rows).set_index("feature")
    report = report.sort_values("between_symbol_auc", ascending=False)
    print("=== Symbol-fingerprint decomposition (0.5 = uninformative) ===")
    print(report.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "NaN"))
    print()

    FINGERPRINT_THRESHOLD = 0.65   # between-symbol AUC clearly above chance
    REAL_SIGNAL_THRESHOLD = 0.55   # within-symbol AUC clearly above chance

    print("=== Verdicts ===")
    for feat, r in report.iterrows():
        bsa, wsa = r["between_symbol_auc"], r["within_symbol_auc"]
        if bsa is None or wsa is None or pd.isna(bsa) or pd.isna(wsa):
            print(f"  {feat}: INCONCLUSIVE (insufficient variation to score)")
            continue
        if bsa >= FINGERPRINT_THRESHOLD and wsa < REAL_SIGNAL_THRESHOLD:
            print(f"  {feat}: LIKELY FINGERPRINT LEAK "
                  f"(between={bsa:.3f} >= {FINGERPRINT_THRESHOLD}, "
                  f"within={wsa:.3f} < {REAL_SIGNAL_THRESHOLD}) -- "
                  f"label tracks symbol identity, not that day's value. "
                  f"Recommend blocking or symbol-demeaning.")
        elif wsa >= REAL_SIGNAL_THRESHOLD:
            print(f"  {feat}: LIKELY REAL DAY-TO-DAY SIGNAL "
                  f"(within={wsa:.3f} >= {REAL_SIGNAL_THRESHOLD}) -- "
                  f"even holding symbol identity constant, day-to-day "
                  f"variation predicts the label. Probably legitimate.")
        else:
            print(f"  {feat}: WEAK/NO SIGNAL EITHER WAY "
                  f"(between={bsa:.3f}, within={wsa:.3f})")

    print()
    print("Reminder: this is still an in-sample univariate check (same caveat")
    print("as diagnose_feature_leakage.py) -- it isolates WHICH component of the")
    print("feature correlates with the label, not whether that correlation")
    print("survives a proper symbol-purged, embargoed out-of-sample split.")
    print("A 'LIKELY REAL' verdict here still needs to hold up under")
    print("time_aware_splits(..., symbols=..., embargo_days=...) before you")
    print("fully trust it.")


if __name__ == "__main__":
    main()
