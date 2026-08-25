#!/usr/bin/env python3
"""
diagnose_cold_start_generalization.py
======================================
Follow-up to diagnose_symbol_fingerprint_leak.py, for features that test as
"LIKELY FINGERPRINT LEAK" there (high between_symbol_auc, low within_symbol_auc)
-- specifically the upper-band/channel family (DCU_20_20, BBU_20_2.0_2.0,
KCUe_20_2 t3/t5/t10) that diagnose_symbol_fingerprint_leak.py flagged.

WHY THIS SCRIPT EXISTS
-----------------------
The between/within decomposition in diagnose_symbol_fingerprint_leak.py can't
distinguish two very different explanations for a high between_symbol_auc:

  (a) SYMBOL-IDENTITY MEMORIZATION: the model is effectively looking up
      "have I seen this exact ticker win before" -- a real overfitting
      problem that will NOT generalize to a stock the model has never
      scored before.

  (b) A REAL, SLOW-MOVING TRAIT computed fresh from that ticker's own price
      data, which happens to correlate with which stocks get selected into
      the winners screener in the first place (see daily_winners_detector.py:
      winners require change_pct >= 20%, non_winners are hard-capped at
      change_pct < 20% -- so "prone to extending above its channel" is
      structurally more common in the winners population, independent of
      any specific day's catalyst). This is available for ANY ticker,
      including one the model has never scored before, because it's
      computed live from that ticker's own OHLCV data -- there's nothing to
      "memorize."

time_aware_splits()'s symbol_purge_window_days already guards against (a)
for the *same* ticker reappearing near a fold boundary. It does nothing for
(b), because (b) isn't about the same ticker appearing twice -- it's about
the label-vs-feature relationship holding at the population level.

This script tells (a) from (b) directly: split rows into COLD-START (a
symbol's first-ever appearance in the training window -- by definition,
nothing has been "learned" about this specific ticker yet) vs REPEAT
(every appearance after the first). If a feature's predictive power is
memorization, cold-start AUC should collapse toward 0.5 while repeat AUC
stays high. If it's a real, freshly-computed trait, cold-start AUC should
be roughly as strong as repeat AUC.

Usage:
    python diagnose_cold_start_generalization.py
    python diagnose_cold_start_generalization.py --features t1_close_DCU_20_20 t3_bbu_20_2_0_2_0
    python diagnose_cold_start_generalization.py --fingerprint-report symbol_fingerprint_report.txt
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import ml_retrain_model as rt  # noqa: E402
from src.ml_predictor.symbol_demeaning import compute_cold_start_mask  # noqa: E402

# Default target set: the upper-band/channel family flagged as fingerprint
# leak in the 2026-08-25 --all-features run (see symbol_fingerprint_report.txt).
DEFAULT_TARGET_BASES = ("DCU_20_20", "BBU_20_2.0_2.0", "KCUe_20_2")
LAG_PREFIXES = ("t1_close_", "t1_open_", "t3_", "t5_", "t10_")

FINGERPRINT_VERDICT_RE = re.compile(
    r"^\s*(\S+): LIKELY FINGERPRINT LEAK", re.MULTILINE
)


def _load_flagged_features(report_path: str) -> list[str]:
    """Pull every feature name marked LIKELY FINGERPRINT LEAK out of a
    saved diagnose_symbol_fingerprint_leak.py report (its --verdicts
    section), so this script can be pointed at exactly what that run
    flagged instead of a hardcoded default."""
    text = Path(report_path).read_text()
    feats = FINGERPRINT_VERDICT_RE.findall(text)
    if not feats:
        print(f"[warn] no 'LIKELY FINGERPRINT LEAK' lines found in {report_path}")
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
        "--fingerprint-report", default=None,
        help="Path to a saved diagnose_symbol_fingerprint_leak.py report; "
             "tests every feature it marked LIKELY FINGERPRINT LEAK instead "
             "of the hardcoded default. Overridden by --features if given.",
    )
    p.add_argument(
        "--min-group-rows", type=int, default=30,
        help="Skip a feature if either the cold-start or repeat group has "
             "fewer than this many non-NaN rows in the train-only slice "
             "(AUC on a tiny group is too noisy to trust). Default: 30.",
    )
    return p.parse_args()


def safe_auc(y: pd.Series, scores: pd.Series) -> float | None:
    scores = scores.fillna(scores.median())
    if scores.nunique() < 2 or y.nunique() < 2:
        return None
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        return None
    return max(auc, 1 - auc)  # direction-agnostic, matches the sibling scripts


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
              "run cold-start diagnostics.", file=sys.stderr)
        sys.exit(1)
    date_col = "detection_date" if "detection_date" in combined_df.columns else "event_date"
    if date_col not in combined_df.columns:
        print(f"ERROR: no '{date_col}' column on combined_df -- cannot "
              "order rows for cold-start detection.", file=sys.stderr)
        sys.exit(1)

    symbol_all = combined_df["symbol"]
    dates_all = combined_df[date_col]

    # Same train-only cutoff as diagnose_symbol_fingerprint_leak.py, so
    # results are directly comparable to that report.
    cutoff = rt._compute_val_cutoff(combined_df)
    parsed_dates = pd.to_datetime(dates_all, errors="coerce")
    train_mask = parsed_dates.isna() | (parsed_dates < cutoff)

    X = X.loc[train_mask].reset_index(drop=True)
    y = y.loc[train_mask].reset_index(drop=True)
    symbol = symbol_all.loc[train_mask].reset_index(drop=True)
    dates = dates_all.loc[train_mask].reset_index(drop=True)

    # Cold-start mask: True = this symbol's first appearance in the
    # train-only window, by (symbol, date) order -- exactly the rows
    # demean_training_features() leaves raw. Everything else is "repeat":
    # this symbol has at least one strictly-earlier row the model could,
    # in principle, have partially memorized against.
    cold_start = compute_cold_start_mask(symbol, dates)

    n_cold, n_repeat = int(cold_start.sum()), int((~cold_start).sum())
    print(f"Train-only rows: {len(X)}  (cutoff={cutoff.date()})  "
          f"unique symbols={symbol.nunique()}")
    print(f"  cold-start (first appearance): {n_cold} rows "
          f"({n_cold / len(X):.1%})")
    print(f"  repeat (2nd+ appearance):       {n_repeat} rows "
          f"({n_repeat / len(X):.1%})\n")

    # Resolve which columns to test.
    if args.features:
        target_cols = [c for c in args.features if c in X.columns]
        missing = [c for c in args.features if c not in X.columns]
        if missing:
            print(f"[warn] not present in feature matrix, skipping: {missing}\n")
    elif args.fingerprint_report:
        flagged = _load_flagged_features(args.fingerprint_report)
        target_cols = [c for c in flagged if c in X.columns]
        missing = [c for c in flagged if c not in X.columns]
        print(f"[fingerprint-report] loaded {len(flagged)} flagged feature(s) "
              f"from {args.fingerprint_report}")
        if missing:
            print(f"[warn] not present in feature matrix, skipping: {missing}\n")
    else:
        target_cols = [
            c for c in X.columns
            if any(
                c[len(pfx):].upper() == base.upper()
                for pfx in LAG_PREFIXES if c.startswith(pfx)
                for base in DEFAULT_TARGET_BASES
            )
        ]

    if not target_cols:
        print("No target columns found to test. Pass --features explicitly, "
              "or --fingerprint-report pointing at a saved report.")
        return

    print(f"Testing {len(target_cols)} column(s): {target_cols}\n")

    rows = []
    for col in target_cols:
        feat = pd.to_numeric(X[col], errors="coerce")

        cold_feat, cold_y = feat[cold_start], y[cold_start]
        repeat_feat, repeat_y = feat[~cold_start], y[~cold_start]

        if cold_feat.notna().sum() < args.min_group_rows or repeat_feat.notna().sum() < args.min_group_rows:
            print(f"[skip] {col}: fewer than {args.min_group_rows} non-NaN "
                  f"rows in cold-start ({cold_feat.notna().sum()}) or repeat "
                  f"({repeat_feat.notna().sum()}) group")
            continue

        cold_auc = safe_auc(cold_y, cold_feat)
        repeat_auc = safe_auc(repeat_y, repeat_feat)

        rows.append({
            "feature": col,
            "cold_start_auc": cold_auc,
            "repeat_auc": repeat_auc,
            "gap": (repeat_auc - cold_auc) if (cold_auc is not None and repeat_auc is not None) else None,
            "n_cold": int(cold_feat.notna().sum()),
            "n_repeat": int(repeat_feat.notna().sum()),
        })

    report = pd.DataFrame(rows).set_index("feature")
    report = report.sort_values("gap", ascending=False)
    print("=== Cold-start vs. repeat AUC (0.5 = uninformative) ===")
    print(report.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "NaN"))
    print()

    REAL_SIGNAL_THRESHOLD = 0.55     # cold-start AUC clearly above chance
    MEMORIZATION_GAP_THRESHOLD = 0.10  # repeat meaningfully stronger than cold-start

    print("=== Verdicts ===")
    for feat, r in report.iterrows():
        ca, ra, gap = r["cold_start_auc"], r["repeat_auc"], r["gap"]
        if ca is None or ra is None or pd.isna(ca) or pd.isna(ra):
            print(f"  {feat}: INCONCLUSIVE (insufficient variation to score)")
            continue
        if ca >= REAL_SIGNAL_THRESHOLD and (gap is None or gap < MEMORIZATION_GAP_THRESHOLD):
            print(f"  {feat}: GENERALIZES TO NEW TICKERS "
                  f"(cold-start={ca:.3f} >= {REAL_SIGNAL_THRESHOLD}, "
                  f"gap vs. repeat={gap:.3f}) -- signal holds up on tickers "
                  f"the model has never scored before. Likely real momentum "
                  f"signal, not memorization. Keep as-is.")
        elif ca < REAL_SIGNAL_THRESHOLD and gap is not None and gap >= MEMORIZATION_GAP_THRESHOLD:
            print(f"  {feat}: LIKELY REQUIRES PRIOR EXPOSURE "
                  f"(cold-start={ca:.3f} < {REAL_SIGNAL_THRESHOLD}, "
                  f"repeat={ra:.3f}, gap={gap:.3f}) -- signal is much "
                  f"weaker on tickers the model hasn't seen before than on "
                  f"repeats. Consistent with archetype/identity memorization "
                  f"rather than fresh-computed signal. Candidate for "
                  f"symbol-demeaning or exclusion.")
        else:
            print(f"  {feat}: AMBIGUOUS (cold-start={ca:.3f}, repeat={ra:.3f}, "
                  f"gap={gap:.3f}) -- doesn't cleanly separate; treat with "
                  f"caution either way.")

    print(
        "\nReminder: like diagnose_symbol_fingerprint_leak.py, this is still "
        "an in-sample univariate check -- it isolates whether a feature's "
        "signal requires prior exposure to a ticker, not whether it "
        "survives a proper embargoed walk-forward split. A "
        "'GENERALIZES TO NEW TICKERS' verdict here still needs to hold up "
        "under time_aware_splits(..., symbols=..., embargo_days=...) "
        "before fully trusting it."
    )


if __name__ == "__main__":
    main()
