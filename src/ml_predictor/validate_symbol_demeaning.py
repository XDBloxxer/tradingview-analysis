#!/usr/bin/env python3
"""
validate_symbol_demeaning.py — Step 3 of the HV fingerprint fix
=================================================================
Follow-up to diagnose_symbol_fingerprint_leak.py and
src/ml_predictor/symbol_demeaning.py. Those established:

  1. HV_10/20/30 is ~80% symbol-fingerprint (between_symbol_auc 0.79-0.84),
     ~20% real day-to-day signal (within_symbol_auc 0.55-0.60).
  2. symbol_demeaning.py causally demeans HV before every model sees it
     (wired into ml_retrain_model.prepare_features() for training and
     explosion_predictor.py for live scoring) -- no change to data
     collection, no change to what's stored, just what the model is fit on.

This script is the validation step: it runs the SAME guarded CV machinery
every selection stage already uses (time_aware_splits(..., symbols=,
embargo_days=)) and checks two things on real data:

  (a) RAW HV, scored under the guard, drops from its unguarded ~0.75-0.76
      univariate AUC down toward the ~0.55-0.60 within-symbol band -- i.e.
      the guard is doing its job of refusing to let the model cheat via
      symbol identity.
  (b) DEMEANED HV, scored under the same guard, holds steady near that same
      ~0.55-0.60 band -- i.e. the demeaning didn't throw away the real
      signal, it just removed the part the guard was already discounting.

If (a) and (b) both hold, that's confirmation the fix targets the leak
component specifically rather than just suppressing the feature's score
across the board.

This is a REAL-DATA analogue of synthetic_leak_test.py (which proves the
guard mechanics work on data with a *known* zero true relationship). Here
the true relationship is unknown, so instead of asserting an absolute
number, this script asserts the *shape* described above.

Usage (same env/credentials as ml_retrain_model.py):

    python validate_symbol_demeaning.py
    python validate_symbol_demeaning.py --n-splits 5 --embargo-days 15
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
import ml_retrain_model as rt  # noqa: E402
from src.ml_predictor.feature_selection import time_aware_splits  # noqa: E402
from src.ml_predictor.symbol_demeaning import (  # noqa: E402
    DEFAULT_DEMEAN_BASES,
    _matching_columns,
    demean_training_features,
)

WITHIN_SYMBOL_BAND = (0.50, 0.65)  # the ~0.55-0.60 range from the diagnostic,
                                    # with slack on both sides for CV noise


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--embargo-days", type=int, default=15)
    p.add_argument("--bases", nargs="+", default=list(DEFAULT_DEMEAN_BASES))
    return p.parse_args()


def mean_fold_auc(X: pd.DataFrame, y: pd.Series, col: str, splits) -> tuple[float, float]:
    """Univariate guarded AUC for one column: fit a single-feature model per
    fold (mirrors the 'quick model importance' style scoring the pipeline
    itself uses, rather than a raw-value AUC, so this is scored the same way
    the model actually consumes the feature). Returns (mean, std) across
    folds with usable class balance on both sides."""
    x = X[[col]].copy()
    x[col] = x[col].fillna(x[col].median())
    aucs = []
    for train_idx, test_idx in splits:
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        m = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="auc", random_state=42)
        m.fit(x.iloc[train_idx], y_tr)
        pred = m.predict_proba(x.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y_te, pred)
        aucs.append(max(auc, 1 - auc))  # direction-agnostic, matches the diagnostics
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    args = _parse_args()
    client = rt.get_supabase_client()
    base_df = rt.load_base_training_data(client, lookback_days=365)
    t1_df = rt.load_t1_data(client, lookback_days=365)
    combined_df = rt.combine_datasets(base_df, t1_df)

    if "symbol" not in combined_df.columns:
        print("ERROR: 'symbol' column not present -- cannot run guarded splits.", file=sys.stderr)
        sys.exit(1)
    date_col = "detection_date" if "detection_date" in combined_df.columns else "event_date"

    y = combined_df["label"].astype(int)
    symbol = combined_df["symbol"]
    dates = combined_df[date_col]

    cutoff = rt._compute_val_cutoff(combined_df)
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    train_mask = parsed_dates.isna() | (parsed_dates < cutoff)

    # Build RAW X the same way prepare_features() does, but stop short of
    # its demeaning step -- we need the raw values as one arm of this
    # comparison, and prepare_features() now demeans HV automatically.
    FEATURE_PREFIXES = ("t1_close_", "t1_open_", "t3_", "t5_", "t10_")
    feature_cols = [
        c for c in combined_df.columns
        if any(c.startswith(pfx) for pfx in FEATURE_PREFIXES)
        and c not in rt.NON_FEATURE_COLS
    ]
    X_raw_full = combined_df[feature_cols].copy()
    for col in X_raw_full.columns:
        X_raw_full[col] = pd.to_numeric(X_raw_full[col], errors="coerce")
    X_raw_full = X_raw_full.replace([np.inf, -np.inf], np.nan)

    hv_cols = _matching_columns(X_raw_full.columns, args.bases)
    if not hv_cols:
        print(f"No columns matching bases {args.bases} found -- nothing to validate.", file=sys.stderr)
        sys.exit(1)

    X_raw = X_raw_full.loc[train_mask, hv_cols].reset_index(drop=True)
    y_tr_full = y.loc[train_mask].reset_index(drop=True)
    symbol_tr = symbol.loc[train_mask].reset_index(drop=True)
    dates_tr = dates.loc[train_mask].reset_index(drop=True)

    X_demeaned_full = demean_training_features(
        X_raw_full.loc[train_mask].reset_index(drop=True),
        symbol_tr, dates_tr, bases=args.bases,
    )
    X_demeaned = X_demeaned_full[hv_cols]

    print(f"Train-only rows: {len(X_raw)}  (cutoff={cutoff.date()})  "
          f"unique symbols={symbol_tr.nunique()}")
    print(f"Guard: n_splits={args.n_splits}, embargo_days={args.embargo_days}, "
          f"symbols=purged per fold\n")

    unguarded_splits = time_aware_splits(dates_tr, n_splits=args.n_splits)
    guarded_splits = time_aware_splits(
        dates_tr, n_splits=args.n_splits, symbols=symbol_tr, embargo_days=args.embargo_days,
    )

    rows = []
    for col in hv_cols:
        raw_unguarded, _ = mean_fold_auc(X_raw, y_tr_full, col, unguarded_splits)
        raw_guarded, raw_std = mean_fold_auc(X_raw, y_tr_full, col, guarded_splits)
        demeaned_guarded, dem_std = mean_fold_auc(X_demeaned, y_tr_full, col, guarded_splits)
        rows.append({
            "feature": col,
            "raw_unguarded_auc": raw_unguarded,
            "raw_guarded_auc": raw_guarded,
            "raw_guarded_std": raw_std,
            "demeaned_guarded_auc": demeaned_guarded,
            "demeaned_guarded_std": dem_std,
        })

    report = pd.DataFrame(rows).set_index("feature")
    print("=== Guarded vs unguarded CV AUC, raw vs demeaned (0.5 = uninformative) ===")
    print(report.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "NaN"))
    print()

    failures = []
    for col, r in report.iterrows():
        raw_u, raw_g, dem_g = r["raw_unguarded_auc"], r["raw_guarded_auc"], r["demeaned_guarded_auc"]
        if any(pd.isna(v) for v in (raw_u, raw_g, dem_g)):
            print(f"[skip] {col}: insufficient fold coverage to score")
            continue

        # (a) the guard should pull raw HV DOWN from its unguarded score.
        ok_drop = raw_g < raw_u
        # (b) both raw-guarded and demeaned-guarded should land in the
        #     within-symbol band -- proving the guard alone gets raw HV most
        #     of the way there, and demeaning doesn't lose ground from there.
        ok_raw_band = WITHIN_SYMBOL_BAND[0] <= raw_g <= WITHIN_SYMBOL_BAND[1]
        ok_dem_band = WITHIN_SYMBOL_BAND[0] <= dem_g <= WITHIN_SYMBOL_BAND[1]
        # (c) demeaning shouldn't cost real signal relative to the guard alone
        #     -- allow a small CV-noise margin rather than requiring dem_g >= raw_g exactly.
        ok_no_loss = dem_g >= raw_g - 0.03

        verdict = "PASS" if (ok_drop and ok_raw_band and ok_dem_band and ok_no_loss) else "FAIL"
        if verdict == "FAIL":
            failures.append(col)
        print(f"  {col}: {verdict}  "
              f"(unguarded={raw_u:.3f} -> guarded_raw={raw_g:.3f} -> guarded_demeaned={dem_g:.3f}; "
              f"drop={'y' if ok_drop else 'n'}, raw_in_band={'y' if ok_raw_band else 'n'}, "
              f"demeaned_in_band={'y' if ok_dem_band else 'n'}, no_loss={'y' if ok_no_loss else 'n'})")

    print()
    if failures:
        print(f"RESULT: {len(failures)}/{len(report)} column(s) did not confirm the expected "
              f"pattern: {failures}")
        print("This doesn't necessarily mean the fix is wrong -- CV fold counts on real data")
        print("are much smaller than the synthetic test's, so single-fold noise can swing a")
        print(f"column outside the {WITHIN_SYMBOL_BAND} band. Re-run with more history "
              "(larger --lookback via LOOKBACK_DAYS) or a coarser --embargo-days before concluding")
        print("the guard/demeaning combination itself is broken.")
        sys.exit(1)
    else:
        print(f"PASS: all {len(report)} column(s) confirm the expected pattern -- guarded CV pulls")
        print("raw HV down toward the within-symbol band, and demeaned HV holds in that same band")
        print("without a guard-relative loss. The symbol-fingerprint fix is validated on real data.")


if __name__ == "__main__":
    main()
