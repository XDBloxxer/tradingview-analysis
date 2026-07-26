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

## Blocking specific features

By default the pipeline loads `ml_models/feature_selection/excluded_features.json`
and drops any column named there **before Stage 0** — so a blocklisted
feature is invisible to the stability check, correlation clustering,
Boruta, RFECV, and the GA alike, and can never come back through any stage.

The file is either a plain JSON list of column names:

```json
["some_leaky_feature", "another_feature_to_never_use"]
```

or a dict with an `excluded_features` key (so you can leave yourself a note):

```json
{"excluded_features": ["some_leaky_feature"], "note": "why"}
```

A missing file is treated as an empty blocklist (no error). Flags:

- `--exclude-features-file PATH` — point at a different blocklist (default
  `ml_models/feature_selection/excluded_features.json`)
- `--no-exclude-features` — disable the blocklist entirely for this run,
  even if the file exists

In the `ML Feature Selection` GitHub Actions workflow this is the
`use_excluded_features` input, on by default.

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
- `--rfecv-min-features` (default 8): keep this well below your expected
  Boruta output count. If it's set close to (or above) how many features
  Boruta actually confirms, RFECV only gets to take one elimination step
  and you get a 2-point "curve" instead of a real elbow — check
  `stage3_rfecv_curve.csv` and confirm it has more than 2-3 rows.
- `GAConfig.min_features` (default 5) is intentionally independent of
  `--rfecv-min-features`. Earlier versions derived the GA's floor from the
  RFECV parameter, so the GA would mechanically converge to that derived
  wall and it looked like a discovered optimum. Don't reintroduce that
  coupling — if you want the GA to explore a narrower range, set
  `GAConfig(min_features=..., max_features=...)` explicitly and treat it
  as a constraint you chose, not a result.
- `GAConfig.n_splits` (default 6): the GA's fitness is walk-forward CV AUC
  on a small candidate pool — too few folds makes `mean_fitness` swing
  wildly between generations (a real run showed it bouncing ~0.53-0.97).
  If `stage4_ga_log.csv`'s `mean_fitness` column is still noisy at 6 splits,
  go higher before trusting the winning subset.
- Stability check worth doing occasionally: run the whole pipeline 2-3
  times with different `--boruta-iterations`/seeds and diff
  `final_features` across runs. Large churn between runs means the GA step
  is picking up noise rather than signal — lean on the Boruta/RFECV output
  instead, which is more stable by construction.
- The genetic step is the most expensive stage per feature — only run it
  once the pool is already down to Boruta/RFECV survivors (roughly 60-150).
