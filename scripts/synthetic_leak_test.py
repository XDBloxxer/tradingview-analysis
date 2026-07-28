# scripts/synthetic_leak_test.py — no Supabase, no secrets, tests the CLASSIFIER's actual split
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from ml_retrain_model import train_val_split  # the function that actually feeds train_model()

rng = np.random.default_rng(0)
n_symbols = 300
rows = []

for sym_id in range(n_symbols):
    baseline_hv = rng.normal(50, 20)          # symbol's fixed volatility "fingerprint"
    n_appearances = rng.integers(1, 6)        # some symbols recur over the 6-month window
    dates = pd.to_datetime("2026-01-30") + pd.to_timedelta(
        np.sort(rng.integers(0, 178, n_appearances)), unit="D"
    )
    for d in dates:
        label = rng.integers(0, 2)            # PURE NOISE — no real signal, by construction
        hv = baseline_hv + rng.normal(0, 3)   # autocorrelated per-symbol, like real hv_30
        rows.append({
            "symbol": sym_id,
            "detection_date": d,
            "hv_30": hv,
            "label": label,
        })

df = pd.DataFrame(rows)
X = df[["hv_30"]]
y = df["label"]
w = pd.Series(1.0, index=df.index)

X_train, X_val, y_train, y_val, w_train, *_ = train_val_split(X, y, w, df)

m = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="auc", early_stopping_rounds=30)
m.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

print(f"AUC with pure-noise labels through the REAL train_val_split(): {auc:.4f}")
print("(should be ~0.5 if the split is clean; anywhere meaningfully above")
print(" 0.5 means symbol repetition across the cutoff — not real signal —")
print(" is what's producing your classifier's high AUC)")
