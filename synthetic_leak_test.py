# scripts/synthetic_leak_test.py — no Supabase, no secrets, no network calls
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.ml_predictor.feature_selection import time_aware_splits

rng = np.random.default_rng(0)
n_symbols = 300
rows = []

for sym_id in range(n_symbols):
    # each "symbol" has a fixed, random baseline volatility signature
    baseline_hv = rng.normal(50, 20)
    n_appearances = rng.integers(1, 6)  # some symbols repeat across dates
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 180, n_appearances)), unit="D"
    )
    for d in dates:
        # label is PURE NOISE — no real relationship to any feature
        label = rng.integers(0, 2)
        # feature is symbol's baseline + small noise (mimics real hv_30 behavior:
        # highly autocorrelated for the same stock across nearby dates)
        hv = baseline_hv + rng.normal(0, 3)
        rows.append({"symbol": sym_id, "date": d, "hv_30": hv, "label": label})

df = pd.DataFrame(rows)
X = df[["hv_30"]]
y = df["label"]
dates = df["date"]

splits = time_aware_splits(dates, n_splits=5)
aucs = []
for train_idx, test_idx in splits:
    m = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="auc")
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    aucs.append(roc_auc_score(y.iloc[test_idx], m.predict_proba(X.iloc[test_idx])[:, 1]))

print("AUC with pure noise labels + symbol-autocorrelated feature:", np.round(aucs, 3))
print("(should be ~0.5 if the split is clean; anywhere above ~0.55 means")
print(" symbol-level autocorrelation alone is enough to fake a lift)")
