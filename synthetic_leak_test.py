# scripts/synthetic_leak_test.py — no Supabase, no secrets, no network calls
#
# Demonstrates (and now regression-tests) symbol-level leakage in
# time_aware_splits(). Originally this script only showed the leak existed;
# it did not check whether the fix (symbols= / embargo_days= in
# time_aware_splits, threaded through rfecv_time_aware / genetic_search /
# stability_select / run_pipeline in feature_selection.py) actually closes
# it. It now runs the SAME synthetic data through both the old call
# (no guards — reproduces the leak) and the new call (guards on — should
# fall back toward ~0.5 AUC), repeated across several random seeds and
# averaged, so a future regression shows up as a failing assertion here
# instead of a suspiciously-flat RFECV curve discovered much later in
# production.
#
# Scenario, per synthetic "symbol":
#   - baseline_hv:       a fixed per-symbol volatility level (mimics hv_30:
#                         highly autocorrelated for the same stock across
#                         nearby dates).
#   - symbol_win_prob:   a fixed per-symbol label tendency, drawn completely
#                         INDEPENDENTLY of baseline_hv.
# There is therefore NO real population-level relationship between hv_30 and
# the label. Any AUC lift above ~0.5 can only come from the model using
# hv_30 as a symbol-identity fingerprint (it's ~constant per symbol, +/-
# small noise) to recall that *specific* symbol's label tendency from
# training rows of the same symbol — i.e. exactly the failure mode FIX 5 in
# ml_retrain_model.train_val_split guards against, and which only shows up
# when the same symbol appears on both sides of a split.
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.ml_predictor.feature_selection import time_aware_splits

N_SEEDS = 5
N_SYMBOLS = 500


def make_synthetic_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym_id in range(N_SYMBOLS):
        baseline_hv = rng.normal(50, 20)
        symbol_win_prob = rng.uniform(0.05, 0.95)
        n_appearances = rng.integers(10, 25)  # symbol repeats heavily across dates
        dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
            np.sort(rng.choice(180, size=n_appearances, replace=False)), unit="D"
        )
        for d in dates:
            label = int(rng.random() < symbol_win_prob)
            hv = baseline_hv + rng.normal(0, 0.5)  # tight noise -> clean fingerprint
            rows.append({"symbol": sym_id, "date": d, "hv_30": hv, "label": label})
    return pd.DataFrame(rows)


def mean_fold_auc(df: pd.DataFrame, splits) -> float:
    X, y = df[["hv_30"]], df["label"]
    aucs = []
    for train_idx, test_idx in splits:
        if len(np.unique(y.iloc[test_idx])) < 2 or len(np.unique(y.iloc[train_idx])) < 2:
            continue
        m = XGBClassifier(n_estimators=300, max_depth=8, eval_metric="auc")
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        aucs.append(roc_auc_score(y.iloc[test_idx], m.predict_proba(X.iloc[test_idx])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


leaky_means, guarded_means = [], []
for seed in range(N_SEEDS):
    df = make_synthetic_df(seed)
    dates, symbols = df["date"], df["symbol"]

    leaky_splits = time_aware_splits(dates, n_splits=5)
    guarded_splits = time_aware_splits(dates, n_splits=5, symbols=symbols, embargo_days=10)

    leaky_auc = mean_fold_auc(df, leaky_splits)
    guarded_auc = mean_fold_auc(df, guarded_splits)
    leaky_means.append(leaky_auc)
    guarded_means.append(guarded_auc)

    print(f"seed {seed}: BEFORE (no guards) mean AUC={leaky_auc:.3f}   "
          f"AFTER (symbols=+embargo_days=10) mean AUC={guarded_auc:.3f}")

leaky_overall = float(np.mean(leaky_means))
guarded_overall = float(np.mean(guarded_means))

print()
print(f"Overall across {N_SEEDS} seeds — BEFORE: {leaky_overall:.3f}   AFTER: {guarded_overall:.3f}")
print("(labels are pure per-symbol noise with zero true relationship to hv_30, so a")
print(" clean split should average ~0.50; consistent lift above that in BEFORE and")
print(" its collapse back down in AFTER is the leak, and the fix, respectively.)")

assert guarded_overall < leaky_overall, (
    f"FAIL: guarded split ({guarded_overall:.3f}) is not lower than the unguarded "
    f"split ({leaky_overall:.3f}) — the symbol purge / embargo guards don't appear "
    "to be reducing the leak. Check that symbols=/embargo_days= are being threaded "
    "through correctly."
)
assert guarded_overall < 0.55, (
    f"FAIL: guarded split mean AUC ({guarded_overall:.3f}) is still well above the "
    "~0.50 expected for a leak-free split on pure-noise labels."
)
print("\nPASS: symbol purge + embargo gap measurably close the leak.")
