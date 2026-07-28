#!/usr/bin/env python3
"""
feature_selection.py — 4-stage feature reduction pipeline
===========================================================

Reduces the ~395-feature matrix used by ml_retrain_model.py down to a small,
statistically-defensible subset, using the same time-aware (walk-forward)
splitting scheme already used elsewhere in this repo for the train/val split.

Stages (each is independently callable; run_pipeline() chains all four):

  1. correlation_cluster_selection()
       Hierarchical-clusters features on `1 - |correlation|`, cuts the tree at
       `corr_threshold`, and keeps one representative per cluster (highest
       |correlation| with the label, tie-broken by lowest NaN rate). Free,
       model-agnostic, and typically the single biggest reduction.

  2. boruta_select()
       Self-contained shadow-feature permutation test (the "real" version of
       "shuffle a column and see if it still matters"). Each real feature is
       duplicated as a shuffled shadow copy; an XGBoost model is trained on
       real+shadow columns; any real feature that beats the best shadow
       feature's importance is scored a "hit". Repeated for n_iterations, and
       a two-sided binomial test (same test the reference Boruta R/py package
       uses) decides Confirmed / Rejected / Tentative at significance `alpha`.

  3. rfecv_time_aware()
       Recursive Feature Elimination with walk-forward (not random/k-fold) CV,
       reusing the same "most recent slice held out" philosophy as
       train_val_split() in ml_retrain_model.py. Produces a score-vs-feature-
       count curve so you can eyeball or auto-pick the elbow.

  4. genetic_search() [optional polish step]
       Genetic-algorithm search over subsets of the ~60-150 RFECV survivors.
       Each candidate subset is scored with *nested* walk-forward CV (mean
       across folds), not a single static split, to discourage p-hacking a
       lucky subset. Only worth running once the candidate pool is small
       (RFECV output), since the search space explodes otherwise.

Nothing in this file talks to Supabase or touches the production model files.
It operates purely on an (X, y, dates[, w]) triple and writes its artifacts
(selected_features.json + a few CSV/JSON diagnostics) to an output directory.
See the `if __name__ == "__main__":` block for a ready-to-run CLI that pulls
the same training data ml_retrain_model.py uses.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ===========================================================================
# Shared: excluded-features blocklist
# ===========================================================================

DEFAULT_EXCLUDED_FEATURES_PATH = "ml_models/feature_selection/excluded_features.json"


def load_excluded_features(path: str | Path) -> list[str]:
    """
    Load a manual feature blocklist for the selection pipeline.

    Expected JSON shape is a simple list of column names:

        ["some_leaky_feature", "another_feature_to_never_use"]

    A dict with an "excluded_features" key (list[str]) is also accepted, so
    the file can carry comments/metadata alongside the list if you want:

        {"excluded_features": ["some_leaky_feature"], "note": "why"}

    Missing file -> returns [] (nothing excluded) rather than raising, so
    this is safe to point at a path that doesn't exist yet.
    """
    p = Path(path)
    if not p.exists():
        logger.info(f"[exclude] no blocklist file at {p} — nothing excluded")
        return []

    with open(p) as f:
        data = json.load(f)

    if isinstance(data, list):
        excluded = data
    elif isinstance(data, dict):
        excluded = data.get("excluded_features", [])
    else:
        raise ValueError(
            f"{p}: expected a JSON list or a dict with an 'excluded_features' "
            f"key, got {type(data).__name__}"
        )

    excluded = [str(c) for c in excluded]
    logger.info(f"[exclude] loaded {len(excluded)} feature(s) to exclude from {p}")
    return excluded


def apply_feature_exclusions(X: pd.DataFrame, excluded_features: list[str]) -> pd.DataFrame:
    """
    Drop `excluded_features` from X before any selection stage sees them.

    Names in the blocklist that aren't actually present in X are logged and
    ignored (not an error) — the pipeline's feature set changes over time,
    so a stale entry shouldn't break the run.
    """
    if not excluded_features:
        return X
    present = [c for c in excluded_features if c in X.columns]
    missing = [c for c in excluded_features if c not in X.columns]
    if missing:
        logger.info(f"[exclude] {len(missing)} blocklisted name(s) not present in X (ignored): {missing}")
    if present:
        logger.info(f"[exclude] dropping {len(present)} blocklisted feature(s): {present}")
    return X.drop(columns=present)


# ===========================================================================
# Shared: time-aware (walk-forward) CV splitter
# ===========================================================================

def time_aware_splits(
    dates: pd.Series,
    n_splits: int = 5,
    min_train_frac: float = 0.4,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward CV splits ordered by `dates` (ties broken by index order).

    Unlike sklearn.model_selection.TimeSeriesSplit (which needs the frame
    pre-sorted and gives you positional folds), this works directly off a
    date column so it matches the "sort by detection/event date, hold out
    the most recent slice" logic already used in train_val_split().

    Rows with NaT dates are always placed in the train portion of every fold
    (mirrors FIX 2 in ml_retrain_model.train_val_split — mistake/undated rows
    must never leak into a validation slice).

    Returns a list of (train_idx, test_idx) pairs of *positional* indices
    into `dates.reset_index(drop=True)`. Each successive fold's test window
    is a later, non-overlapping slice of the timeline — this is the walk-
    forward scheme the caller should reuse instead of random/K-fold CV.
    """
    d = pd.to_datetime(dates, errors="coerce").reset_index(drop=True)
    n = len(d)
    nat_mask = d.isna()

    dated_pos = np.where(~nat_mask)[0]
    order = dated_pos[np.argsort(d.iloc[dated_pos].values)]  # positions sorted by date

    n_dated = len(order)
    start = int(n_dated * min_train_frac)
    if n_dated - start < n_splits:
        raise ValueError(
            f"Not enough dated rows ({n_dated}) for {n_splits} walk-forward "
            f"folds with min_train_frac={min_train_frac}. Reduce n_splits."
        )

    fold_edges = np.linspace(start, n_dated, n_splits + 1, dtype=int)
    nat_positions = np.where(nat_mask)[0]

    splits = []
    for i in range(n_splits):
        train_end = fold_edges[i]
        test_end = fold_edges[i + 1]
        train_pos = order[: max(train_end - gap, 0)]
        test_pos = order[train_end:test_end]
        train_pos = np.concatenate([train_pos, nat_positions])
        splits.append((train_pos, test_pos))
    return splits


def _prep_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    Xc = X.copy()
    for c in Xc.columns:
        Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    return Xc, y.astype(int).values


def _quick_model_importance(
    Xc: pd.DataFrame,
    yv: np.ndarray,
    random_state: int = 42,
) -> pd.Series:
    """
    Single-fit XGBoost gain importance over the *full* feature matrix.

    Used by correlation_cluster_selection() to pick cluster representatives
    on the same kind of signal (nonlinear, interaction-aware model
    importance) that every downstream stage (Boruta/RFECV/GA) actually
    optimizes for — instead of raw Pearson correlation with y, which can
    disagree with model importance and silently drop the real signal
    carrier of a cluster before it ever reaches Boruta/RFECV/GA.

    This is one extra cheap fit (not walk-forward, not CV) — good enough
    for a representative-selection tiebreak, not meant to replace the
    proper time-aware evaluation done in later stages.
    """
    import xgboost as xgb

    Xf = Xc.fillna(Xc.median(numeric_only=True)).fillna(0.0)
    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(Xf, yv)
    return pd.Series(model.feature_importances_, index=Xc.columns)


# ===========================================================================
# Stage 1 — Correlation clustering
# ===========================================================================

def correlation_cluster_selection(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.90,
    method: str = "average",
    min_periods: int = 30,
) -> tuple[list[str], pd.DataFrame]:
    """
    Hierarchical-cluster features on `1 - |pairwise correlation|`, cut the
    dendrogram at `corr_threshold`, keep one representative per cluster.

    Representative choice, in priority order:
      1. Highest XGBoost gain importance from a single full-matrix fit
         (see _quick_model_importance) — chosen instead of raw Pearson
         label-correlation because downstream stages (Boruta/RFECV/GA) all
         score features with model importance, not linear correlation.
         Ranking cluster reps on a different signal than everything
         downstream uses is exactly the kind of mismatch that can bury the
         real signal carrier of a cluster (e.g. a nonlinear volatility
         feature) behind a weaker but more linearly-correlated cluster-mate
         before it ever reaches Boruta/RFECV/GA — producing run-to-run
         "churn" that looks like noise but is actually a selection-metric
         bug.
      2. Lowest NaN rate (best coverage), as a tiebreak.

    Returns (selected_feature_names, cluster_report_df). The report has one
    row per *original* feature with its cluster id and whether it was kept,
    so you can audit what got dropped and why.
    """
    Xc, yv = _prep_xy(X, y)
    cols = list(Xc.columns)

    corr = Xc.corr(min_periods=min_periods).fillna(0.0)
    corr = corr.reindex(index=cols, columns=cols).fillna(0.0)
    dist = 1.0 - corr.abs().values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry against fp noise
    dist[dist < 0] = 0.0

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    cluster_ids = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")

    nan_rate = Xc.isna().mean()
    model_imp = _quick_model_importance(Xc, yv)
    # Kept only for diagnostics in the report (no longer used to rank reps).
    label_corr = Xc.apply(lambda s: s.corr(pd.Series(yv, index=Xc.index)))
    label_corr = label_corr.abs().fillna(0.0)

    report_rows = []
    selected = []
    for cid in sorted(set(cluster_ids)):
        members = [c for c, k in zip(cols, cluster_ids) if k == cid]
        ranked = sorted(
            members,
            key=lambda c: (-model_imp.get(c, 0.0), nan_rate.get(c, 1.0)),
        )
        rep = ranked[0]
        selected.append(rep)
        for c in members:
            report_rows.append({
                "feature": c,
                "cluster_id": int(cid),
                "cluster_size": len(members),
                "model_importance": round(float(model_imp.get(c, 0.0)), 5),
                "label_corr_abs": round(float(label_corr.get(c, 0.0)), 4),
                "nan_rate": round(float(nan_rate.get(c, 1.0)), 4),
                "kept_as_representative": c == rep,
            })

    report = pd.DataFrame(report_rows).sort_values(
        ["cluster_size", "cluster_id"], ascending=[False, True]
    ).reset_index(drop=True)

    logger.info(
        f"[corr-cluster] {len(cols)} -> {len(selected)} features "
        f"({len(set(cluster_ids))} clusters at r={corr_threshold})"
    )
    return selected, report


# ===========================================================================
# Stage 2 — Boruta (shadow-feature permutation test)
# ===========================================================================

@dataclass
class BorutaResult:
    confirmed: list[str]
    tentative: list[str]
    rejected: list[str]
    history: pd.DataFrame  # per-feature hit counts / p-values


def boruta_select(
    X: pd.DataFrame,
    y: pd.Series,
    w: Optional[pd.Series] = None,
    n_iterations: int = 100,
    alpha: float = 0.05,
    max_depth: int = 6,
    n_estimators: int = 100,
    random_state: int = 42,
    keep_tentative: bool = False,
) -> BorutaResult:
    """
    Self-contained Boruta implementation (no external `boruta` package, so
    there's no dependency on its now-unmaintained sklearn/numpy API surface).

    Algorithm, repeated `n_iterations` times:
      1. Build a shadow copy of every real feature by independently
         shuffling its values (destroys any real relationship with y while
         preserving the marginal distribution).
      2. Fit an XGBoost classifier on [real | shadow] columns.
      3. `shadow_max` = highest shadow-feature importance this round.
      4. Any real feature whose importance beats shadow_max scores a "hit".

    After all iterations, each feature's hit count is compared against a
    Binomial(n_iterations, 0.5) null (the same null the reference Boruta
    package uses — "a useless feature beats the max shadow ~50% of the time
    by chance") via a two-sided binomial test. Confirmed = p < alpha and
    hit-rate > 0.5; Rejected = p < alpha and hit-rate < 0.5; everything else
    is Tentative.
    """
    import xgboost as xgb

    Xc, yv = _prep_xy(X, y)
    Xc = Xc.fillna(Xc.median(numeric_only=True)).fillna(0.0)
    cols = list(Xc.columns)
    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    spw = n_neg / max(n_pos, 1)
    sample_weight = w.values if w is not None else None

    rng = np.random.RandomState(random_state)
    hits = pd.Series(0, index=cols)

    for it in range(n_iterations):
        shadow = Xc.apply(lambda s: rng.permutation(s.values))
        shadow.columns = [f"shadow__{c}" for c in cols]
        Xit = pd.concat([Xc, shadow], axis=1)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=random_state + it,
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(Xit, yv, sample_weight=sample_weight)
        importances = pd.Series(model.feature_importances_, index=Xit.columns)

        shadow_max = importances.filter(like="shadow__").max()
        real_imp = importances.loc[cols]
        hits[real_imp > shadow_max] += 1

        if (it + 1) % max(1, n_iterations // 5) == 0:
            logger.info(f"[boruta] iteration {it + 1}/{n_iterations}")

    rows = []
    confirmed, tentative, rejected = [], [], []
    for c in cols:
        k = int(hits[c])
        p = binomtest(k, n_iterations, 0.5, alternative="two-sided").pvalue
        hit_rate = k / n_iterations
        if p < alpha and hit_rate > 0.5:
            status = "confirmed"
            confirmed.append(c)
        elif p < alpha and hit_rate < 0.5:
            status = "rejected"
            rejected.append(c)
        else:
            status = "tentative"
            tentative.append(c)
        rows.append({
            "feature": c, "hits": k, "hit_rate": round(hit_rate, 3),
            "p_value": round(float(p), 5), "status": status,
        })

    history = pd.DataFrame(rows).sort_values("hit_rate", ascending=False).reset_index(drop=True)
    logger.info(
        f"[boruta] confirmed={len(confirmed)} tentative={len(tentative)} "
        f"rejected={len(rejected)} (of {len(cols)}, {n_iterations} iterations)"
    )
    result = BorutaResult(confirmed=confirmed, tentative=tentative, rejected=rejected, history=history)
    if keep_tentative:
        result.confirmed = confirmed + tentative
    return result


# ===========================================================================
# Stage 3 — RFECV with time-aware CV
# ===========================================================================

def rfecv_time_aware(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    w: Optional[pd.Series] = None,
    min_features: int = 20,
    step: int = 5,
    n_splits: int = 5,
    max_depth: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """
    Manual RFECV loop (not sklearn.feature_selection.RFECV, which only
    accepts a single scalar `cv` and would force either k-fold or a plain
    generator without our walk-forward semantics baked in). At each step:

      1. Score every remaining feature by mean CV-fold gain importance.
      2. Drop the `step` weakest features.
      3. Re-fit and re-evaluate with time_aware_splits().

    Returns (features_at_best_score, curve_df) where curve_df has one row
    per elimination round (n_features, mean_auc, std_auc) — plot this to
    find the elbow ("AUC barely moves after 60 features, then degrades").
    """
    import xgboost as xgb

    Xc, yv = _prep_xy(X, y)
    Xc = Xc.fillna(Xc.median(numeric_only=True)).fillna(0.0)
    remaining = list(Xc.columns)
    splits = time_aware_splits(dates, n_splits=n_splits)
    sample_weight = w.values if w is not None else np.ones(len(yv))

    curve_rows = []
    round_feature_sets = []

    while True:
        fold_aucs = []
        importances_accum = pd.Series(0.0, index=remaining)

        for train_pos, test_pos in splits:
            if len(np.unique(yv[test_pos])) < 2 or len(np.unique(yv[train_pos])) < 2:
                continue
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(yv[train_pos] == 0).sum() / max((yv[train_pos] == 1).sum(), 1),
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            )
            model.fit(
                Xc.iloc[train_pos][remaining], yv[train_pos],
                sample_weight=sample_weight[train_pos],
            )
            proba = model.predict_proba(Xc.iloc[test_pos][remaining])[:, 1]
            fold_aucs.append(roc_auc_score(yv[test_pos], proba))
            importances_accum += pd.Series(model.feature_importances_, index=remaining)

        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        std_auc = float(np.std(fold_aucs)) if fold_aucs else float("nan")
        curve_rows.append({"n_features": len(remaining), "mean_auc": mean_auc, "std_auc": std_auc})
        round_feature_sets.append(list(remaining))
        logger.info(f"[rfecv] n_features={len(remaining):4d}  mean_auc={mean_auc:.4f} +/- {std_auc:.4f}")

        if len(remaining) <= min_features:
            break

        importances_accum /= max(len(fold_aucs), 1)
        ranked = importances_accum.sort_values(ascending=True)
        n_drop = min(step, len(remaining) - min_features)
        to_drop = set(ranked.index[:n_drop])
        remaining = [c for c in remaining if c not in to_drop]

    curve = pd.DataFrame(curve_rows)
    best_round = curve["mean_auc"].idxmax()
    best_features = round_feature_sets[best_round]
    logger.info(
        f"[rfecv] best round: {curve.loc[best_round, 'n_features']} features, "
        f"mean_auc={curve.loc[best_round, 'mean_auc']:.4f}"
    )
    return best_features, curve


# ===========================================================================
# Stage 4 — Genetic-algorithm subset search (optional polish)
# ===========================================================================

@dataclass
class GAConfig:
    population_size: int = 40
    n_generations: int = 25
    crossover_rate: float = 0.7
    mutation_rate: float = 0.05
    tournament_size: int = 3
    # FIX: 4 splits on an already-small candidate pool gave a very noisy
    # fitness signal (mean_fitness swinging ~0.53-0.97 between generations
    # in practice) — each fold's test AUC was based on too few positives to
    # be stable. More folds -> more, smaller, but averaged evaluations.
    n_splits: int = 6
    # FIX: previously this was implicitly forced to equal rfecv_min_features//2
    # by run_pipeline() below, so the GA would converge exactly to its own
    # configured wall and it looked like "10 features is optimal" when it was
    # really just "10 is the floor I was given." Lowering the default here
    # (and decoupling it from rfecv_min_features in run_pipeline) lets the GA
    # actually explore smaller subsets instead of hitting a hidden ceiling.
    min_features: int = 5
    max_features: Optional[int] = None
    random_state: int = 42


def genetic_search(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    candidate_features: list[str],
    w: Optional[pd.Series] = None,
    config: Optional[GAConfig] = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Genetic-algorithm search over subsets of `candidate_features`
    (intended to be the ~60-150 survivors of rfecv_time_aware — the search
    space is only tractable once it's been cut down that far).

    Each individual is a bitmask over candidate_features. Fitness = mean
    walk-forward CV AUC (via time_aware_splits, re-evaluated fresh for every
    candidate — never reused from a cached static split) minus a small
    penalty per feature, so the search doesn't just converge to "keep
    everything".

    Returns (best_feature_subset, generation_log_df).
    """
    import xgboost as xgb

    config = config or GAConfig()
    rng = random.Random(config.random_state)

    Xc, yv = _prep_xy(X, y)
    Xc = Xc.fillna(Xc.median(numeric_only=True)).fillna(0.0)
    splits = time_aware_splits(dates, n_splits=config.n_splits)
    sample_weight = w.values if w is not None else np.ones(len(yv))
    n_feat = len(candidate_features)
    max_features = config.max_features or n_feat
    min_features = min(config.min_features, n_feat)

    def random_individual() -> list[bool]:
        k = rng.randint(min_features, max_features)
        idx = set(rng.sample(range(n_feat), k))
        return [i in idx for i in range(n_feat)]

    def fitness(mask: list[bool]) -> float:
        feats = [f for f, keep in zip(candidate_features, mask) if keep]
        if len(feats) < min_features:
            return -1.0
        aucs = []
        for train_pos, test_pos in splits:
            if len(np.unique(yv[test_pos])) < 2 or len(np.unique(yv[train_pos])) < 2:
                continue
            model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(yv[train_pos] == 0).sum() / max((yv[train_pos] == 1).sum(), 1),
                random_state=config.random_state, n_jobs=-1, eval_metric="logloss",
            )
            model.fit(Xc.iloc[train_pos][feats], yv[train_pos], sample_weight=sample_weight[train_pos])
            proba = model.predict_proba(Xc.iloc[test_pos][feats])[:, 1]
            aucs.append(roc_auc_score(yv[test_pos], proba))
        if not aucs:
            return -1.0
        return float(np.mean(aucs)) - 0.0005 * len(feats)  # tiny parsimony penalty

    def tournament(pop: list, fits: list) -> list[bool]:
        idxs = rng.sample(range(len(pop)), config.tournament_size)
        best = max(idxs, key=lambda i: fits[i])
        return pop[best]

    def crossover(a: list[bool], b: list[bool]) -> tuple[list[bool], list[bool]]:
        point = rng.randint(1, n_feat - 1)
        return a[:point] + b[point:], b[:point] + a[point:]

    def mutate(mask: list[bool]) -> list[bool]:
        return [not bit if rng.random() < config.mutation_rate else bit for bit in mask]

    population = [random_individual() for _ in range(config.population_size)]
    log_rows = []
    best_mask, best_fit = population[0], -1.0

    for gen in range(config.n_generations):
        fits = [fitness(ind) for ind in population]
        gen_best_i = int(np.argmax(fits))
        if fits[gen_best_i] > best_fit:
            best_fit, best_mask = fits[gen_best_i], population[gen_best_i]
        log_rows.append({
            "generation": gen,
            "best_fitness": float(np.max(fits)),
            "mean_fitness": float(np.mean(fits)),
            "best_n_features": int(sum(population[gen_best_i])),
        })
        logger.info(
            f"[GA] gen {gen:3d}  best={np.max(fits):.4f}  mean={np.mean(fits):.4f}  "
            f"n_features(best)={sum(population[gen_best_i])}"
        )

        new_pop = [population[gen_best_i]]  # elitism
        while len(new_pop) < config.population_size:
            p1, p2 = tournament(population, fits), tournament(population, fits)
            if rng.random() < config.crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            new_pop.append(mutate(c1))
            if len(new_pop) < config.population_size:
                new_pop.append(mutate(c2))
        population = new_pop

    best_features = [f for f, keep in zip(candidate_features, best_mask) if keep]
    log_df = pd.DataFrame(log_rows)
    logger.info(f"[GA] final: {len(best_features)} features, fitness={best_fit:.4f}")
    return best_features, log_df


# ===========================================================================
# Stability wrapper — re-run Stages 1-3 across resamples, report frequency
# ===========================================================================

@dataclass
class StabilityResult:
    frequency: pd.DataFrame  # one row per feature: times_selected, frequency, mean_importance
    stable_features: list[str]  # features selected in >= min_frequency of runs
    per_run_features: list[list[str]]  # raw per-run RFECV output, for auditing


def stability_select(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    w: Optional[pd.Series] = None,
    n_runs: int = 8,
    min_frequency: float = 0.75,
    block_frac: float = 0.85,
    corr_threshold: float = 0.90,
    boruta_iterations: int = 100,
    boruta_alpha: float = 0.05,
    rfecv_min_features: int = 8,
    rfecv_step: int = 5,
    random_state: int = 42,
) -> StabilityResult:
    """
    Answer "which features should I actually run" directly, instead of
    eyeballing tallies across separately-launched pipeline runs by hand.

    Re-runs Stages 1-3 (correlation clustering -> Boruta -> RFECV) `n_runs`
    times, each time on a different contiguous block bootstrap of the
    training window (resample rows in blocks, not i.i.d., to avoid
    shuffling apart the time-series structure the walk-forward CV logic
    elsewhere in this file relies on). The GA stage (4) is intentionally
    excluded here — it's the noisiest stage and the one this function
    exists to route around; run genetic_search() separately afterward on
    `stable_features` if you still want a GA polish pass on a pre-vetted,
    stable pool.

    A feature's `frequency` is the fraction of the `n_runs` in which it
    survived all the way to that run's RFECV output. `mean_importance` is
    its average _quick_model_importance() score across the runs it
    appeared in (NaN if it never appeared) — use this to break ties among
    equally-stable features, the same way the retrained feature_importance
    ranking was used earlier to prioritize among the old GA "stable 4".

    Returns features with frequency >= min_frequency as `stable_features`,
    but the full `frequency` table is the more useful artifact: it lets you
    manually pick a cutoff instead of trusting a single hard-coded one.
    """
    rng = np.random.RandomState(random_state)
    Xc, yv_full = _prep_xy(X, y)
    n = len(Xc)
    block_size = max(int(n * block_frac), 1)

    counts: dict[str, int] = {}
    importance_sums: dict[str, float] = {}
    importance_counts: dict[str, int] = {}
    per_run_features: list[list[str]] = []

    for run_i in range(n_runs):
        # Contiguous block resample: pick a random start, take a
        # contiguous slice, wrapping around if needed. Preserves local
        # time-ordering (required by time_aware_splits downstream) while
        # still perturbing which rows/period the run sees, so cluster
        # membership and Boruta/RFECV outcomes can actually vary run to
        # run — that variation is the whole point of this function.
        start = rng.randint(0, n)
        idx = (np.arange(block_size) + start) % n
        idx = np.sort(idx)

        X_run = X.iloc[idx].reset_index(drop=True)
        y_run = y.iloc[idx].reset_index(drop=True)
        dates_run = dates.iloc[idx].reset_index(drop=True)
        w_run = w.iloc[idx].reset_index(drop=True) if w is not None else None

        logger.info(f"[stability] run {run_i + 1}/{n_runs} (n={len(X_run)})")

        corr_features, _ = correlation_cluster_selection(
            X_run, y_run, corr_threshold=corr_threshold,
        )
        boruta_result = boruta_select(
            X_run[corr_features], y_run, w=w_run,
            n_iterations=boruta_iterations, alpha=boruta_alpha,
            random_state=random_state + run_i,
        )
        boruta_features = boruta_result.confirmed
        if len(boruta_features) < rfecv_min_features:
            boruta_features = boruta_result.confirmed + boruta_result.tentative

        rfecv_features, _ = rfecv_time_aware(
            X_run[boruta_features], y_run, dates_run, w=w_run,
            min_features=rfecv_min_features, step=rfecv_step,
            random_state=random_state + run_i,
        )
        per_run_features.append(rfecv_features)

        Xc_run, yv_run = _prep_xy(X_run[rfecv_features], y_run)
        run_importance = _quick_model_importance(Xc_run, yv_run, random_state=random_state + run_i)

        for feat in rfecv_features:
            counts[feat] = counts.get(feat, 0) + 1
            importance_sums[feat] = importance_sums.get(feat, 0.0) + float(run_importance.get(feat, 0.0))
            importance_counts[feat] = importance_counts.get(feat, 0) + 1

    rows = []
    for feat, k in counts.items():
        mean_imp = importance_sums[feat] / importance_counts[feat]
        rows.append({
            "feature": feat,
            "times_selected": k,
            "frequency": round(k / n_runs, 3),
            "mean_importance": round(mean_imp, 5),
        })
    frequency = pd.DataFrame(rows).sort_values(
        ["frequency", "mean_importance"], ascending=[False, False]
    ).reset_index(drop=True)

    stable_features = frequency.loc[
        frequency["frequency"] >= min_frequency, "feature"
    ].tolist()

    logger.info(
        f"[stability] {len(stable_features)}/{len(frequency)} features "
        f"selected in >= {min_frequency:.0%} of {n_runs} runs"
    )
    return StabilityResult(
        frequency=frequency,
        stable_features=stable_features,
        per_run_features=per_run_features,
    )


# ===========================================================================
# Orchestrator
# ===========================================================================

@dataclass
class PipelineResult:
    stage0_features: list[str]
    stage1_corr_features: list[str]
    stage2_boruta_features: list[str]
    stage3_rfecv_features: list[str]
    stage4_ga_features: Optional[list[str]]
    final_features: list[str]
    artifacts_dir: Path


def run_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    w: Optional[pd.Series] = None,
    output_dir: str | Path = "ml_models/feature_selection",
    corr_threshold: float = 0.90,
    boruta_iterations: int = 100,
    boruta_alpha: float = 0.05,
    # FIX: 20 was too close to typical Boruta output (~24 in practice), so
    # RFECV took a single elimination step and produced a 2-point "curve"
    # instead of an actual elbow. Lower default so it walks a real range.
    rfecv_min_features: int = 8,
    rfecv_step: int = 5,
    run_genetic_polish: bool = True,
    ga_config: Optional[GAConfig] = None,
    run_stability_check: bool = True,
    stability_n_runs: int = 8,
    stability_min_frequency: float = 0.75,
    exclude_features: Optional[list[str]] = None,
) -> PipelineResult:
    """
    Run the pipeline and persist artifacts to `output_dir`.

    CALLER CONTRACT — X/y/dates must already be train-only: this function
    (and every stage it calls) fits directly on whatever rows it's given.
    Stages 1 (correlation-cluster representative pick) and 2 (Boruta) have
    no internal time split of their own, so if X/y/dates here still include
    rows from the classifier's val / blind-calibration window, this pipeline
    will pick features using label information from that "held out" period
    before the classifier ever sees it — inflating its val AUC and
    blind_cal_auc regardless of how clean ml_retrain_model.py's own split is.
    The `_cli()` entry point below carves out a train-only slice (mirroring
    ml_retrain_model.py's _compute_val_cutoff()/VAL_WEEKS) before calling
    this function; any other caller must do the same.

    `run_stability_check` defaults to True: Stages 1-3 are first repeated
    `stability_n_runs` times over block-bootstrapped resamples (see
    stability_select()) and the per-feature selection frequency +
    mean importance is written to `stability_frequency.csv`. This does not
    change what feeds the rest of the pipeline (Boruta/RFECV/GA below still
    run once on the full data, as before) — it's a diagnostic to check
    whether that single run's output is actually stable, so you don't have
    to eyeball tallies across separately-launched runs by hand.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if exclude_features:
        X = apply_feature_exclusions(X, exclude_features)

    if run_stability_check:
        logger.info("STAGE 0.5: stability check (bootstrapped Stages 1-3)")
        stability = stability_select(
            X, y, dates, w=w,
            n_runs=stability_n_runs,
            min_frequency=stability_min_frequency,
            corr_threshold=corr_threshold,
            boruta_iterations=boruta_iterations,
            boruta_alpha=boruta_alpha,
            rfecv_min_features=rfecv_min_features,
            rfecv_step=rfecv_step,
        )
        stability.frequency.to_csv(out / "stability_frequency.csv", index=False)
        logger.info(
            f"[stability] wrote {out / 'stability_frequency.csv'} — "
            f"{len(stability.stable_features)} features stable at "
            f">= {stability_min_frequency:.0%} frequency"
        )

    logger.info("=" * 70)
    logger.info(f"STAGE 0: starting from {X.shape[1]} features")
    logger.info("=" * 70)

    logger.info("STAGE 1: correlation clustering")
    corr_features, corr_report = correlation_cluster_selection(X, y, corr_threshold=corr_threshold)
    corr_report.to_csv(out / "stage1_correlation_clusters.csv", index=False)

    logger.info("STAGE 2: Boruta shadow-feature test")
    boruta_result = boruta_select(X[corr_features], y, w=w, n_iterations=boruta_iterations, alpha=boruta_alpha)
    boruta_result.history.to_csv(out / "stage2_boruta_history.csv", index=False)
    boruta_features = boruta_result.confirmed
    if len(boruta_features) < rfecv_min_features:
        logger.warning(
            f"[boruta] only {len(boruta_features)} confirmed features (< "
            f"rfecv_min_features={rfecv_min_features}) — including tentative features too."
        )
        boruta_features = boruta_result.confirmed + boruta_result.tentative

    logger.info("STAGE 3: RFECV (time-aware walk-forward CV)")
    rfecv_features, rfecv_curve = rfecv_time_aware(
        X[boruta_features], y, dates, w=w,
        min_features=rfecv_min_features, step=rfecv_step,
    )
    rfecv_curve.to_csv(out / "stage3_rfecv_curve.csv", index=False)

    ga_features = None
    if run_genetic_polish:
        logger.info("STAGE 4: genetic-algorithm polish")
        # FIX: previously `min_features=max(10, rfecv_min_features // 2)` meant
        # the GA's floor was mechanically derived from the RFECV parameter, so
        # a run that produced ~24 RFECV survivors with rfecv_min_features=20
        # would force the GA floor to exactly 10 — and the GA would obediently
        # converge there every time, looking like a discovery when it was
        # really just the wall it was given. GAConfig's own default (5) is
        # independent of rfecv_min_features, so the GA can actually explore
        # down to a genuinely small subset instead of a derived one.
        cfg = ga_config or GAConfig()
        ga_features, ga_log = genetic_search(X[rfecv_features], y, dates, rfecv_features, w=w, config=cfg)
        ga_log.to_csv(out / "stage4_ga_log.csv", index=False)

    final_features = ga_features if ga_features is not None else rfecv_features

    result = PipelineResult(
        stage0_features=list(X.columns),
        stage1_corr_features=corr_features,
        stage2_boruta_features=boruta_features,
        stage3_rfecv_features=rfecv_features,
        stage4_ga_features=ga_features,
        final_features=final_features,
        artifacts_dir=out,
    )

    summary = {
        "stage0_count": len(result.stage0_features),
        "stage1_corr_count": len(result.stage1_corr_features),
        "stage2_boruta_count": len(result.stage2_boruta_features),
        "stage3_rfecv_count": len(result.stage3_rfecv_features),
        "stage4_ga_count": len(ga_features) if ga_features is not None else None,
        "final_count": len(final_features),
        "final_features": final_features,
        "corr_threshold": corr_threshold,
        "boruta_iterations": boruta_iterations,
        "boruta_alpha": boruta_alpha,
        "rfecv_min_features": rfecv_min_features,
        "excluded_features": list(exclude_features) if exclude_features else [],
    }
    with open(out / "selected_features.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(
        f"PIPELINE COMPLETE: {summary['stage0_count']} -> {summary['stage1_corr_count']} "
        f"(corr) -> {summary['stage2_boruta_count']} (boruta) -> {summary['stage3_rfecv_count']} "
        f"(rfecv)" + (f" -> {summary['stage4_ga_count']} (GA)" if ga_features is not None else "")
    )
    logger.info(f"Artifacts written to: {out}/")
    logger.info("=" * 70)
    return result


# ===========================================================================
# CLI — pulls the same training data ml_retrain_model.py uses
# ===========================================================================

def _cli() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--boruta-iterations", type=int, default=100)
    parser.add_argument("--boruta-alpha", type=float, default=0.05)
    parser.add_argument("--rfecv-min-features", type=int, default=8)
    parser.add_argument("--rfecv-step", type=int, default=5)
    parser.add_argument("--skip-genetic", action="store_true")
    parser.add_argument(
        "--no-stability-check", action="store_true",
        help="Skip the stability check (on by default): repeating Stages 1-3 "
             "over block-bootstrapped resamples first and writing "
             "stability_frequency.csv (per-feature selection frequency + "
             "mean importance) before running the normal single-pass pipeline.",
    )
    parser.add_argument("--stability-n-runs", type=int, default=8)
    parser.add_argument("--stability-min-frequency", type=float, default=0.75)
    parser.add_argument(
        "--exclude-features-file",
        default=DEFAULT_EXCLUDED_FEATURES_PATH,
        help="JSON file listing feature names to exclude from selection entirely "
             f"(default: {DEFAULT_EXCLUDED_FEATURES_PATH}). Either a plain JSON "
             "list of names, or a dict with an 'excluded_features' key. Missing "
             "file is treated as an empty list (no error).",
    )
    parser.add_argument(
        "--no-exclude-features", action="store_true",
        help="Disable the feature blocklist entirely, even if --exclude-features-file exists "
             "(the blocklist is applied by default).",
    )
    parser.add_argument("--output-dir", default="ml_models/feature_selection")
    parser.add_argument(
        "--lookback-days", type=int, default=365,
        help="Cap how much T-1/base history is fetched for feature selection "
             "(server-side, via the same fetch_table_paginated() date filter "
             "ml_retrain_model.py uses for its own --lookback-days). Deliberately "
             "much wider than retrain's window (currently 180d) so feature "
             "selection still sees more market-regime diversity than any single "
             "retrain does -- this is a ceiling to stop egress growing "
             "unboundedly as the T-1 tables accumulate, not a tight budget. "
             "Pass 0 to disable and fetch full history.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Reuse the exact same data-loading / feature-prep code path as the
    # production retrain script, so the feature set this pipeline evaluates
    # is identical to what train_model() actually sees.
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import ml_retrain_model as rt  # noqa: E402

    client = rt.get_supabase_client()
    lookback_days = args.lookback_days or None  # 0 → None → unbounded fetch
    base_df = rt.load_base_training_data(client, lookback_days=lookback_days)
    t1_df = rt.load_t1_data(client, lookback_days=lookback_days)
    combined_df = rt.combine_datasets(base_df, t1_df)
    X, y, w = rt.prepare_features(combined_df)

    date_col = "detection_date" if "detection_date" in combined_df.columns else "event_date"
    dates = combined_df[date_col] if date_col in combined_df.columns else pd.Series(pd.NaT, index=combined_df.index)

    # ── LEAK FIX: restrict feature selection to train-only rows ─────────────
    # Stages 1 (correlation-cluster representative pick) and 2 (Boruta) each
    # fit directly on whatever X/y they're given, with no internal time split
    # of their own — unlike Stage 3 (RFECV) and Stage 4 (GA), which already do
    # walk-forward CV internally. If X/y here include the same rows that
    # ml_retrain_model.py later holds out as val / blind-calibration data,
    # those stages pick features using label information from the "held out"
    # period before ml_retrain_model.py ever sees it — so its val AUC and
    # blind_cal_auc both come out inflated regardless of how clean the
    # downstream split is. This mirrors ml_retrain_model.py's own
    # _compute_val_cutoff()/VAL_WEEKS logic so feature selection is blind to
    # exactly the same window the classifier is evaluated on.
    cutoff = rt._compute_val_cutoff(combined_df)
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    # Rows with unparseable dates (e.g. mistake samples) are kept in the
    # train-only pool, matching ml_retrain_model.py's FIX 2 (NaT -> train).
    fs_train_mask = parsed_dates.isna() | (parsed_dates < cutoff)
    n_excluded = int((~fs_train_mask).sum())
    logging.getLogger(__name__).info(
        f"Feature-selection train-only cutoff: {cutoff.date()} — "
        f"excluding {n_excluded} row(s) at/after cutoff from feature selection "
        f"({int(fs_train_mask.sum())} rows remain for Stages 1-4)."
    )
    X = X.loc[fs_train_mask]
    y = y.loc[fs_train_mask]
    if w is not None:
        w = w.loc[fs_train_mask]
    dates = dates.loc[fs_train_mask]

    exclude_features: list[str] = []
    if not args.no_exclude_features:
        exclude_features = load_excluded_features(args.exclude_features_file)
    else:
        logging.getLogger(__name__).info(
            "[exclude] --no-exclude-features set — ignoring any blocklist file"
        )

    run_pipeline(
        X, y, dates, w=w,
        output_dir=args.output_dir,
        corr_threshold=args.corr_threshold,
        boruta_iterations=args.boruta_iterations,
        boruta_alpha=args.boruta_alpha,
        rfecv_min_features=args.rfecv_min_features,
        rfecv_step=args.rfecv_step,
        run_genetic_polish=not args.skip_genetic,
        run_stability_check=not args.no_stability_check,
        stability_n_runs=args.stability_n_runs,
        stability_min_frequency=args.stability_min_frequency,
        exclude_features=exclude_features,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
