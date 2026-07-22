# Feature selection pipeline

`feature_selection.py` reduces the ~395-column feature matrix to a small,
defensible subset in four stages. It's a standalone module — it doesn't
change any default behaviour of `ml_retrain_model.py` unless you opt in.

## Run it

```bash
python -m src.ml_predictor.feature_selection --verbose
```

This pulls training data through the exact same `load_base_training_data` /
`load_t1_data` / `combine_datasets` / `prepare_features` path that
`ml_retrain_model.py` uses, then runs:

1. **Correlation clustering** — hierarchical-clusters on `1 - |corr|`, cuts
   at `--corr-threshold` (default 0.90), keeps the best representative per
   cluster (highest |correlation| with the label, tie-broken by NaN rate).
2. **Boruta** — self-contained shadow-feature permutation test. No external
   `boruta` package (it's unmaintained against current sklearn/numpy); this
   reimplements the same shadow-feature + binomial-test logic directly
   against XGBoost.
3. **RFECV** — recursive elimination scored with the *same walk-forward
   time-aware split philosophy* as `train_val_split()` in
   `ml_retrain_model.py` (never random/K-fold — that would leak future
   regime information into feature selection).
4. **Genetic algorithm** (optional, `--skip-genetic` to disable) — subset
   search over the RFECV survivors, each candidate scored with fresh nested
   walk-forward CV rather than a single static split.

Artifacts land in `ml_models/feature_selection/`:

- `selected_features.json` — final feature list + stage counts
- `stage1_correlation_clusters.csv` — every original feature, its cluster,
  and whether it was kept
- `stage2_boruta_history.csv` — hit rate / p-value / status per feature
- `stage3_rfecv_curve.csv` — score vs. feature-count curve (find the elbow)
- `stage4_ga_log.csv` — best/mean fitness per GA generation

## Use the result in training

The selected subset is opt-in and does not change default behaviour:

```bash
USE_SELECTED_FEATURES=1 python ml_retrain_model.py
```

`prepare_features()` checks for `ml_models/feature_selection/selected_features.json`
only when that env var is set; otherwise every retrain uses the full feature
set exactly as before.

## Using the pieces individually

Each stage is also a plain function you can call on any `(X, y, dates)` you
already have in memory — useful for notebook exploration:

```python
from src.ml_predictor.feature_selection import (
    correlation_cluster_selection, boruta_select, rfecv_time_aware,
    genetic_search, time_aware_splits, GAConfig, run_pipeline,
)

features, report = correlation_cluster_selection(X, y, corr_threshold=0.85)
```

## Tuning notes

- `--corr-threshold`: lower (e.g. 0.80) drops more features but risks
  merging genuinely distinct signals; 0.85–0.90 is a reasonable range.
- `--boruta-iterations`: 100 is a solid default; bump to 200+ if the
  confirmed/tentative split near the boundary looks noisy.
- `--rfecv-min-features` / `--rfecv-step`: smaller step = finer-grained
  elbow curve but more XGBoost fits (slower).
- The genetic step is the most expensive stage per feature — only run it
  once the pool is already down to Boruta/RFECV survivors (roughly 60-150).
