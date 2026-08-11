"""
One-off check: does the 5min/daily_fallback ratio differ by label, and (if
it does) is the _t1_source_propensity_weights() fix in
ml_retrain_model.load_t1_data() actually neutralising it?
Run with the same env/credentials as validate_symbol_demeaning.py.

This script does three things in sequence:

  (a) RAW mix -- the original check. Counts t1_data_source vs label,
      unweighted. This is unaffected by the propensity-reweighting fix (the
      fix only touches sample_weight, never the raw data), so a large gap
      here is expected to persist even after the fix is deployed -- it's
      not a regression, it's just measuring something the fix doesn't
      change.

  (b) WEIGHTED mix -- same crosstab, but rows counted by sample_weight
      instead of by 1. This is what the classifier, gain regressor, and
      feature-selection pipeline actually train on. If the propensity fix
      is working, this should converge to (approximately) the same
      5min/daily_fallback split for both labels, even though (a) doesn't.

  (c) GUARDED AUC -- fits a tiny univariate model (is_5min -> label) under
      the same guarded time-aware CV (symbol purge + embargo) the rest of
      the pipeline uses, once unweighted and once with sample_weight passed
      through. Unweighted should stay clearly above 0.5 (same leak as (a));
      weighted should drop toward ~0.5 if the fix is actually neutralising
      t1_data_source as a label proxy in training, not just on paper.

(b) and (c) only run if a sample_weight column is present -- older runs of
this script (before the fix existed) will just get (a), same as before.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ml_retrain_model as rt  # noqa: E402

NEAR_CHANCE_BAND = (0.45, 0.65)  # weighted AUC should land near 0.5; slack for CV noise
WEIGHTED_GAP_PASS = 0.03         # weighted mix gap should be under this to call it fixed


def _mean_fold_auc(x_col, y, splits, sample_weight=None):
    """Univariate guarded AUC for a single 0/1 column. Same style as
    validate_symbol_demeaning.py's mean_fold_auc -- kept as a local copy
    rather than importing, since it's a five-line helper and this script
    should stay runnable even if that file changes shape later."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    x = x_col.to_frame()
    aucs = []
    for train_idx, test_idx in splits:
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        m = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="auc", random_state=42)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.iloc[train_idx].values
        m.fit(x.iloc[train_idx], y_tr, **fit_kwargs)
        pred = m.predict_proba(x.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y_te, pred)
        aucs.append(max(auc, 1 - auc))  # direction-agnostic
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs))


client = rt.get_supabase_client()
base_df = rt.load_base_training_data(client, lookback_days=365)
t1_df = rt.load_t1_data(client, lookback_days=365)

# ── (a) RAW mix -- unchanged from the original version of this script ──────
props = t1_df.groupby("label")["t1_data_source"].value_counts(normalize=True).unstack()
counts = t1_df.groupby("label")["t1_data_source"].value_counts().unstack()

print("=== (a) RAW proportions (5min vs daily_fallback) by label -- unweighted ===")
print(props)
print()
print("=== (a) RAW counts by label ===")
print(counts)
print()

raw_gap = None
if "5min" in props.columns and 0 in props.index and 1 in props.index:
    p0 = props.loc[0, "5min"]
    p1 = props.loc[1, "5min"]
    raw_gap = abs(p0 - p1)
    print(f"5min share: non-winners={p0:.1%}, winners={p1:.1%}, gap={raw_gap:.1%}")
    print()
    if raw_gap < 0.03:
        print("VERDICT (a): Gap is small (<3pp). t1_data_source is NOT meaningfully")
        print("skewed by label in the raw data.")
    elif raw_gap < 0.08:
        print("VERDICT (a): Moderate gap (3-8pp). Worth a closer look.")
    else:
        print("VERDICT (a): Large gap (>8pp). t1_data_source IS meaningfully skewed by")
        print("label in the raw data -- see (b)/(c) below for whether the propensity-")
        print("reweighting fix is actually neutralising this in training.")
else:
    print("VERDICT (a): Could not compute a clean comparison (missing label or source "
          "category) -- inspect the raw tables above manually.")
print()

# ── (b) + (c) only make sense once combine_datasets() has run, since that's
# where sample_weight (and the propensity fix) gets attached -- load_t1_data
# alone doesn't carry it through combine_datasets' downstream columns like
# 'symbol' that (c) needs for the guard.
combined_df = rt.combine_datasets(base_df, t1_df)

if "sample_weight" not in combined_df.columns:
    print("(b)/(c) SKIPPED: no sample_weight column present -- either the "
          "propensity-reweighting fix hasn't been deployed yet, or "
          "t1_data_source wasn't found this run (see load_t1_data warnings above).")
    sys.exit(0)

df = combined_df.dropna(subset=["t1_data_source", "label"]).copy()
df["label"] = df["label"].astype(int)

# ── (b) WEIGHTED mix ─────────────────────────────────────────────────────
weighted = df.groupby(["label", "t1_data_source"])["sample_weight"].sum().unstack()
weighted_props = weighted.div(weighted.sum(axis=1), axis=0)
print("=== (b) WEIGHTED proportions by label -- what the model actually trains on ===")
print(weighted_props)
print()

mix_ok = False
if "5min" in weighted_props.columns and 0 in weighted_props.index and 1 in weighted_props.index:
    wp0, wp1 = weighted_props.loc[0, "5min"], weighted_props.loc[1, "5min"]
    weighted_gap = abs(wp0 - wp1)
    print(f"weighted 5min share: label=0 {wp0:.1%}, label=1 {wp1:.1%}, gap={weighted_gap:.1%}")
    mix_ok = weighted_gap < WEIGHTED_GAP_PASS
    print(f"VERDICT (b): {'PASS' if mix_ok else 'FAIL'} "
          f"(gap {'<' if mix_ok else '>='} {WEIGHTED_GAP_PASS:.0%})")
else:
    print("VERDICT (b): SKIPPED (missing a label or source category)")
print()

# ── (c) GUARDED AUC, unweighted vs. weighted ────────────────────────────
auc_ok = None
if "symbol" not in df.columns:
    print("(c) SKIPPED: 'symbol' column not present -- cannot run guarded splits.")
else:
    from src.ml_predictor.feature_selection import time_aware_splits

    date_col = "detection_date" if "detection_date" in df.columns else "event_date"
    df = df.reset_index(drop=True)

    y = df["label"]
    symbol = df["symbol"]
    dates = df[date_col]
    is_5min = (df["t1_data_source"] == "5min").astype(int)
    weight = df["sample_weight"].astype(float)

    guarded_splits = time_aware_splits(dates, n_splits=5, symbols=symbol, embargo_days=15)

    unweighted_auc, unweighted_std = _mean_fold_auc(is_5min, y, guarded_splits, sample_weight=None)
    weighted_auc, weighted_std = _mean_fold_auc(is_5min, y, guarded_splits, sample_weight=weight)

    print("=== (c) Guarded CV AUC of is_5min predicting label (0.5 = uninformative) ===")
    print(f"  unweighted: {unweighted_auc:.4f} (+/- {unweighted_std:.4f})  "
          f"-- expected to stay clearly above 0.5, same leak as (a)")
    print(f"  weighted:   {weighted_auc:.4f} (+/- {weighted_std:.4f})  "
          f"-- expected to drop toward ~0.5 if the fix is working")
    print()

    if any(pd.isna(v) for v in (unweighted_auc, weighted_auc)):
        print("VERDICT (c): insufficient fold coverage to score -- re-run with more "
              "history or fewer/coarser splits.")
    else:
        auc_ok = NEAR_CHANCE_BAND[0] <= weighted_auc <= NEAR_CHANCE_BAND[1]
        print(f"VERDICT (c): {'PASS' if auc_ok else 'FAIL'} (weighted AUC in {NEAR_CHANCE_BAND} band)")
print()

# ── Overall ──────────────────────────────────────────────────────────────
if auc_ok is None:
    print("OVERALL: could not fully evaluate (c) -- see above for why. Check (b) on its own.")
elif mix_ok and auc_ok:
    print("OVERALL PASS: raw skew is still present in the data (as expected), but the "
          "propensity-reweighting fix neutralises it in training -- weighted mix is matched "
          "across labels and a model can no longer use t1_data_source as a label proxy.")
else:
    print("OVERALL FAIL: at least one of (b)/(c) did not confirm the fix. This doesn't "
          "necessarily mean the code is wrong -- CV fold counts on real data are small, so "
          "single-fold noise can swing the weighted AUC outside the band. Re-run with more "
          "history (bump the lookback_days above 365) before concluding the reweighting "
          "itself is broken.")
    sys.exit(1)
