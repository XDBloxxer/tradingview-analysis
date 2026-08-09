#!/usr/bin/env python3
"""
diagnose_feature_leakage.py
============================
Run this BEFORE the next feature_selection.py pipeline run, on the same
train-only slice it would use. It checks for leakage that survives symbol
purge + embargo — i.e. leakage baked into the features/missingness
themselves rather than time/symbol adjacency.

Usage (from repo root, same env as feature_selection.py / ml_retrain_model.py):

    python diagnose_feature_leakage.py

What it does:
  1. Loads data via the exact same path feature_selection.py's _cli() uses
     (rt.load_base_training_data / load_t1_data / combine_datasets /
     prepare_features), then applies the same train-only cutoff.
  2. Computes univariate ROC-AUC of every single feature against the label,
     on the train-only rows, with NaN filled to the column median (matches
     what the RFECV loop actually does internally, so this reproduces what
     the model can see, not more).
  3. Compares per-feature NaN rate between class 0 and class 1 rows, since
     missingness that correlates with the label can leak just as hard as
     a "real" feature value, especially after fillna(median).
  4. Prints the top offenders on both checks.

Anything with univariate AUC > ~0.75 on its own, or a NaN-rate gap of more
than ~5-10 percentage points between classes, deserves a hard look before
you trust any downstream feature-selection or model AUC number.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import ml_retrain_model as rt  # noqa: E402


def main():
    lookback_days = int(os.environ.get("LOOKBACK_DAYS", "365")) or None  # 0 -> unbounded
    client = rt.get_supabase_client()
    base_df = rt.load_base_training_data(client, lookback_days=lookback_days)
    t1_df = rt.load_t1_data(client, lookback_days=lookback_days)
    combined_df = rt.combine_datasets(base_df, t1_df)
    X, y, w = rt.prepare_features(combined_df)

    date_col = "detection_date" if "detection_date" in combined_df.columns else "event_date"
    dates = combined_df[date_col] if date_col in combined_df.columns else pd.Series(pd.NaT, index=combined_df.index)

    cutoff = rt._compute_val_cutoff(combined_df)
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    train_mask = parsed_dates.isna() | (parsed_dates < cutoff)
    X, y = X.loc[train_mask].reset_index(drop=True), y.loc[train_mask].reset_index(drop=True)
    print(f"Train-only rows: {len(X)}  (cutoff={cutoff.date()})  pos={int(y.sum())} neg={int((y==0).sum())}\n")

    # --- 1. Univariate AUC per feature ---
    aucs = {}
    for col in X.columns:
        vals = X[col]
        if vals.notna().sum() < 20:
            continue
        filled = vals.fillna(vals.median())
        if filled.nunique() < 2:
            continue
        try:
            score = roc_auc_score(y, filled)
        except ValueError:
            continue
        aucs[col] = max(score, 1 - score)  # direction-agnostic

    auc_series = pd.Series(aucs).sort_values(ascending=False)
    print("=== TOP 20 features by univariate AUC vs label (higher = more suspicious) ===")
    print(auc_series.head(20).to_string())
    print()

    # --- 2. NaN-rate gap between classes ---
    nan_pos = X.loc[y == 1].isna().mean()
    nan_neg = X.loc[y == 0].isna().mean()
    nan_gap = (nan_pos - nan_neg).abs().sort_values(ascending=False)
    print("=== TOP 20 features by |NaN-rate gap| between class 1 and class 0 ===")
    gap_df = pd.DataFrame({
        "nan_rate_pos": nan_pos,
        "nan_rate_neg": nan_neg,
        "abs_gap": nan_gap,
    }).sort_values("abs_gap", ascending=False)
    print(gap_df.head(20).to_string())
    print()

    flagged_auc = auc_series[auc_series > 0.75]
    flagged_gap = gap_df[gap_df["abs_gap"] > 0.05]
    print(f"Flagged (univariate AUC > 0.75): {len(flagged_auc)} feature(s)")
    print(f"Flagged (NaN-rate gap > 5pp):    {len(flagged_gap)} feature(s)")
    if len(flagged_auc) == 0 and len(flagged_gap) == 0:
        print("\nNo single-feature or missingness smoking gun found — leakage is likely")
        print("distributed across many correlated features rather than one column.")
        print("Consider checking row provenance (query/join differences between the")
        print("winners_* and non_winners_* source tables) instead.")


if __name__ == "__main__":
    main()
