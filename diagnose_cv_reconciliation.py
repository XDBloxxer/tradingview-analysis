#!/usr/bin/env python3
"""
diagnose_cv_reconciliation.py
==============================
Follow-up to the 2026-08-26 retrain: the final model's own walk-forward CV
(cv_walk_forward_evaluate() in ml_retrain_model.py, as stored in
ml_models/model_metadata.json's "cv_walk_forward") reported mean_auc=0.373
with EVERY fold below 0.5, while feature selection's RFECV stage
(rfecv_time_aware() in src/ml_predictor/feature_selection.py, as stored in
ml_models/feature_selection/stage3_rfecv_curve.csv) reported mean_auc=0.751
on the exact same 13-feature final set, under guarded time-aware CV.

Two guarded CV runs on the same features giving 0.75 vs 0.37 means
something differs in SETUP between the two evaluations, not that the
features are secretly worthless. This script isolates that by doing two
things:

  1. RECONCILE: run cv_walk_forward_evaluate() itself, sweeping
     --lookback-days over the values each pipeline actually used by
     default (feature selection: 365, weekly retrain: 200) -- same
     function, same code path, only the data window changes. If the gap
     mostly closes at one of these values, the "discrepancy" is just two
     pipelines pulling different amounts of history, not a bug.

  2. CALENDAR CHECK: for whichever lookback_days you settle on, print each
     fold's test-window date range next to its AUC. A U-shape or a clear
     breakpoint aligned with a specific calendar stretch is evidence of
     regime dependence (add a regime/volatility-context feature) rather
     than a broken pipeline (look for a bug instead).

Usage:
    # Reconcile: try both defaults and print a side-by-side table
    python diagnose_cv_reconciliation.py --reconcile

    # Reconcile across custom values
    python diagnose_cv_reconciliation.py --reconcile --lookback-days-list 200 365 0

    # Calendar check at one specific lookback window
    python diagnose_cv_reconciliation.py --lookback-days 365

    # Use the exact hyperparams the search rejected, or override n_splits/embargo
    python diagnose_cv_reconciliation.py --lookback-days 365 --n-splits 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ml_retrain_model as rt  # noqa: E402

MODEL_METADATA_PATH = Path(__file__).parent / "ml_models" / "model_metadata.json"


def _load_production_features() -> list[str]:
    """Pull the exact feature list the current production model was
    trained on, out of ml_models/model_metadata.json, so this script
    tests the SAME features that produced the 0.373 vs 0.751 numbers
    instead of a re-derived or hand-picked set."""
    if not MODEL_METADATA_PATH.exists():
        print(f"[warn] {MODEL_METADATA_PATH} not found; --features required.")
        return []
    meta = json.loads(MODEL_METADATA_PATH.read_text())
    return meta.get("features", [])


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--features", nargs="+", default=None,
        help="Exact column name(s) to restrict X to. Default: the current "
             "production model's feature list from ml_models/model_metadata.json.",
    )
    p.add_argument(
        "--lookback-days", type=int, default=None,
        help="Single lookback window (days of T-1/base history, 0=unbounded) "
             "for the calendar-check report. Ignored if --reconcile is set.",
    )
    p.add_argument(
        "--reconcile", action="store_true",
        help="Sweep --lookback-days-list instead of a single window, running "
             "cv_walk_forward_evaluate() at each and printing a side-by-side "
             "mean_auc/std_auc comparison table.",
    )
    p.add_argument(
        "--lookback-days-list", type=int, nargs="+", default=[200, 365],
        help="Lookback windows to sweep under --reconcile. Default: 200 "
             "(ml_weekly_retrain.yml's default) and 365 "
             "(ml_feature_selection.yml's default) -- the two values that "
             "actually produced the two conflicting numbers being reconciled.",
    )
    p.add_argument(
        "--n-splits", type=int, default=rt.CV_N_SPLITS,
        help=f"Number of walk-forward folds (default matches ml_retrain_model.py: {rt.CV_N_SPLITS}).",
    )
    p.add_argument(
        "--min-train-frac", type=float, default=rt.CV_MIN_TRAIN_FRAC,
        help=f"Minimum fraction of dated rows reserved for the first fold's "
             f"train split (default matches ml_retrain_model.py: {rt.CV_MIN_TRAIN_FRAC}).",
    )
    return p.parse_args()


def _prepare(lookback_days: int | None, features: list[str] | None):
    """Load + combine data and restrict X to `features`, exactly like
    ml_retrain_model.py's own pipeline (same loaders, same
    combine_datasets/prepare_features), so any AUC difference is
    attributable only to lookback_days / n_splits / embargo, not to a
    different data-prep path."""
    client = rt.get_supabase_client()
    base_df = rt.load_base_training_data(client, lookback_days=lookback_days)
    t1_df = rt.load_t1_data(client, lookback_days=lookback_days)
    combined_df = rt.combine_datasets(base_df, t1_df)
    X, y, w = rt.prepare_features(combined_df)

    if features:
        missing = [c for c in features if c not in X.columns]
        if missing:
            print(f"[warn] not present in feature matrix, skipping: {missing}")
        keep = [c for c in features if c in X.columns]
        X = X[keep]

    return X, y, w, combined_df


def _fold_date_table(X, combined_df, n_splits, min_train_frac) -> pd.DataFrame:
    """Rebuild the same fold boundaries cv_walk_forward_evaluate() uses
    internally (via _build_cv_splits), and attach each fold's test-window
    calendar date range -- info cv_walk_forward_evaluate()'s own return
    value doesn't carry, since model_metadata.json only needs the AUC for
    production monitoring, not the date range."""
    splits, embargo_or_reason = rt._build_cv_splits(
        X, combined_df, n_splits=n_splits, min_train_frac=min_train_frac
    )
    if splits is None:
        print(f"[error] could not build CV splits: {embargo_or_reason}")
        return pd.DataFrame()

    sort_date, _symbols = rt._cv_sort_date_and_symbols(combined_df)
    sort_date = pd.to_datetime(sort_date, errors="coerce")

    rows = []
    for i, (train_pos, test_pos) in enumerate(splits):
        test_dates = sort_date.iloc[test_pos].dropna()
        rows.append({
            "fold": i,
            "test_start": test_dates.min().date() if len(test_dates) else None,
            "test_end": test_dates.max().date() if len(test_dates) else None,
            "n_test": len(test_pos),
        })
    return pd.DataFrame(rows).set_index("fold")


def run_calendar_check(args, features):
    print(f"=== Calendar check: lookback_days={args.lookback_days}, "
          f"n_splits={args.n_splits}, min_train_frac={args.min_train_frac} ===\n")
    X, y, w, combined_df = _prepare(args.lookback_days, features)
    print(f"Rows: {len(X)}  Features: {len(X.columns)}  "
          f"Positive rate: {y.mean():.1%}\n")

    date_table = _fold_date_table(X, combined_df, args.n_splits, args.min_train_frac)
    cv_results = rt.cv_walk_forward_evaluate(
        X, y, w, combined_df, n_splits=args.n_splits, min_train_frac=args.min_train_frac,
    )
    fold_df = pd.DataFrame(cv_results["fold_results"]).set_index("fold")
    merged = date_table.join(fold_df[["auc", "best_iteration", "skipped"]], how="left")

    print(merged.to_string())
    print(f"\nmean_auc={cv_results['mean_auc']:.4f}  std_auc={cv_results['std_auc']:.4f}  "
          f"n_folds_used={cv_results['n_folds_used']}/{cv_results['n_splits_requested']}\n")

    below_half = merged["auc"].dropna().lt(0.5).sum()
    total = merged["auc"].dropna().shape[0]
    if below_half == total and total > 1:
        print(f"[flag] ALL {total} folds scored below 0.5 -- this is not just "
              "'no signal', it's consistent anti-correlation across every "
              "held-out time window. Check for a sign/label bug before "
              "assuming this is a regime effect.")
    elif below_half >= total / 2:
        print(f"[flag] {below_half}/{total} folds below 0.5 -- look at whether "
              "the worst fold(s) cluster in a specific calendar stretch above "
              "(regime dependence) versus scattered evenly (more likely noise "
              "from small per-fold positive counts).")
    else:
        print(f"[ok] {total - below_half}/{total} folds at or above 0.5.")


def run_reconcile(args, features):
    print(f"=== Reconciling cv_walk_forward_evaluate() across lookback_days="
          f"{args.lookback_days_list} (n_splits={args.n_splits}, "
          f"min_train_frac={args.min_train_frac}) ===\n")
    print("Same function (cv_walk_forward_evaluate), same feature set, same "
          "n_splits/embargo -- only the data window changes below. If mean_auc "
          "converges toward stage3_rfecv_curve.csv's 0.751 at one of these "
          "windows, the discrepancy was a lookback-window mismatch between "
          "ml_feature_selection.yml (365) and ml_weekly_retrain.yml (200), not "
          "a real problem with the features.\n")

    rows = []
    for lb in args.lookback_days_list:
        lb_arg = None if lb == 0 else lb
        print(f"--- lookback_days={lb if lb else 'unbounded'} ---")
        X, y, w, combined_df = _prepare(lb_arg, features)
        cv_results = rt.cv_walk_forward_evaluate(
            X, y, w, combined_df, n_splits=args.n_splits, min_train_frac=args.min_train_frac,
        )
        fold_aucs = [f["auc"] for f in cv_results["fold_results"] if f["auc"] is not None]
        n_below_half = sum(1 for a in fold_aucs if a < 0.5)
        rows.append({
            "lookback_days": lb if lb else "unbounded",
            "n_rows": len(X),
            "mean_auc": cv_results["mean_auc"],
            "std_auc": cv_results["std_auc"],
            "n_folds_used": cv_results["n_folds_used"],
            "folds_below_0.5": f"{n_below_half}/{len(fold_aucs)}",
            "fold_aucs": [round(a, 3) for a in fold_aucs],
        })
        print(f"  rows={len(X)}  mean_auc={cv_results['mean_auc']:.4f}  "
              f"std_auc={cv_results['std_auc']:.4f}  "
              f"fold_aucs={[round(a, 3) for a in fold_aucs]}\n")

    summary = pd.DataFrame(rows).set_index("lookback_days")
    print("=== Summary ===")
    print(summary.to_string())

    print(f"\nFor reference: stage3_rfecv_curve.csv (feature-selection-time CV "
          f"on this same 13-feature set) reported mean_auc=0.7513. "
          f"model_metadata.json's cv_walk_forward (production retrain) "
          f"reported mean_auc=0.3730.")
    best = summary["mean_auc"].idxmax()
    print(f"\nBest mean_auc among tested windows: lookback_days={best} "
          f"({summary.loc[best, 'mean_auc']:.4f})")
    if summary.loc[best, "mean_auc"] < 0.6:
        print("[flag] Even the best window tested doesn't get close to "
              "0.7513 -- the discrepancy is likely NOT just a lookback_days "
              "mismatch. Check n_splits/embargo_days/scale_pos_weight "
              "differences between rfecv_time_aware() and "
              "cv_walk_forward_evaluate() next, or suspect a genuine bug in "
              "one of the two scoring paths.")


def main():
    args = _parse_args()
    features = args.features if args.features else _load_production_features()
    if not features:
        print("[error] no features resolved (model_metadata.json missing/empty "
              "and --features not given). Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"Testing {len(features)} production feature(s): {features}\n")

    if args.reconcile:
        run_reconcile(args, features)
    else:
        if args.lookback_days is None:
            args.lookback_days = 365
            print("[info] --lookback-days not given, defaulting to 365 "
                  "(ml_feature_selection.yml's window) for the calendar check.\n")
        run_calendar_check(args, features)


if __name__ == "__main__":
    main()
