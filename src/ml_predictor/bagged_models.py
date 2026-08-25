"""
bagged_models.py
-----------------
Shared home for BaggedXGBClassifier / BaggedXGBRegressor so that joblib/pickle
can always resolve them by fully-qualified name
(src.ml_predictor.bagged_models.BaggedXGBClassifier / BaggedXGBRegressor)
regardless of which script is __main__ at load time — same reasoning as
_PriorCorrectedModel in prior_corrected_model.py (see that file's docstring
for the joblib/__main__ pitfall this avoids).

WHY THIS EXISTS (seed-variance bagging)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XGBOOST_PARAMS fixes random_state=42, but subsample=0.8 / colsample_bytree=0.8
still make each tree's row/feature sample a random draw. On a dataset this
small, a meaningful slice of a single model's val AUC (classifier) or val
R²/MAE (regressor) is just "which random draw did tree-building make" —
not signal about the features or hyperparameters.

Fix: train N models that are identical except for random_state, and average
their predictions. This is bagging over the seed rather than over rows
(row/feature subsampling already happens per-tree inside each model), and it
directly cancels the seed-dependent component of variance while leaving the
signal (which is common across seeds) intact.

Both classes are deliberately thin — they hold a list of already-fitted
estimators and average predict() / predict_proba() across them. Nothing here
performs the training loop itself; that stays in ml_retrain_model.py so the
existing early-stopping / eval_set / sample_weight logic is reused unchanged
for each seed.
"""

import numpy as np


class BaggedXGBClassifier:
    """
    Averages predict_proba() across N XGBClassifier models trained with
    different random_state values (same params, same data, same eval_set —
    only the seed driving subsample/colsample_bytree row & feature draws
    differs).

    Exposes predict_proba / predict / classes_ / best_iteration / best_score
    / get_booster()-compatible feature importance so it's a drop-in
    replacement for a single XGBClassifier everywhere downstream:
      - CalibratedClassifierCV(..., cv="prefit") only needs predict_proba,
        so calibration fits on top of the ensemble's averaged raw scores.
      - best_iteration / best_score are averaged across seeds for logging
        and model_metadata.json.
      - get_feature_importance() averages each seed's gain-importance dict
        (see compute_feature_importance() in ml_retrain_model.py, which
        checks for this method before falling back to get_booster()).

    Must be defined at module level (not inside a function) so joblib can
    pickle it via its fully-qualified name.
    """

    def __init__(self, estimators, consensus_iteration=None, per_seed_peaks=None,
                 consensus_score=None):
        """
        Args:
            estimators: fitted XGBClassifier seeds (see class docstring).
            consensus_iteration: optional int. When each seed is independently
                early-stopped, its "best round" is picked off that seed's own
                noisy per-round validation curve — on a val set with only a
                few hundred positives, that curve can swing wildly from one
                random_state to the next even with identical train/val data
                (observed in practice: best_iteration ranging ~6-33 across 7
                otherwise-identical seeds). Averaging those noisy per-seed
                *picks* just averages the noise.
                Callers that instead capture each seed's FULL per-round eval
                curve (no early stopping), average the curves themselves
                across seeds first, and THEN pick the argmax of that smoothed
                curve, should pass the resulting round here. It overrides the
                per-seed-average fallback below as self.best_iteration, and
                every predict_proba() call is truncated to it via
                iteration_range so inference actually uses the consensus
                number of trees from every seed, not each seed's own (noisy)
                stopping point.
            per_seed_peaks: optional list[int], each seed's own curve-argmax
                (for logging/diagnostics only — shows how much noisier the
                un-averaged per-seed picks were than the consensus).
            consensus_score: optional float, the averaged eval-metric curve's
                value AT consensus_iteration (i.e. mean_curve[consensus_iteration
                - 1]). Overrides self.best_score the same way consensus_iteration
                overrides self.best_iteration. Pass the metric you actually
                want reported (e.g. true ROC-AUC, not aucpr) — see
                ml_retrain_model.py's seed-training loop.
        """
        if not estimators:
            raise ValueError("BaggedXGBClassifier needs at least one fitted estimator")
        self.estimators_ = list(estimators)
        self.classes_ = self.estimators_[0].classes_
        self.n_seeds_ = len(self.estimators_)
        self.seeds_ = [
            getattr(e, "random_state", None) for e in self.estimators_
        ]
        # Aggregate early-stopping diagnostics across seeds so downstream
        # logging / model_metadata.json still get a single representative
        # number instead of needing to know about bagging.
        _iters = [int(e.best_iteration) for e in self.estimators_
                  if getattr(e, "best_iteration", None) is not None]
        _scores = [float(e.best_score) for e in self.estimators_
                   if getattr(e, "best_score", None) is not None]
        self.best_iteration = int(round(np.mean(_iters))) if _iters else None
        self.best_score = float(np.mean(_scores)) if _scores else None
        self.best_iteration_std_ = float(np.std(_iters)) if _iters else None
        self.best_score_std_ = float(np.std(_scores)) if _scores else None

        # CONSENSUS STOPPING FIX: prefer the curve-averaged round when the
        # caller supplies one — it directly replaces the noisy
        # average-of-independent-early-stops number above.
        self.per_seed_peaks_ = list(per_seed_peaks) if per_seed_peaks else None
        self.consensus_iteration_ = int(consensus_iteration) if consensus_iteration is not None else None
        if self.consensus_iteration_ is not None:
            self.best_iteration = self.consensus_iteration_
        if consensus_score is not None:
            self.best_score = float(consensus_score)
        # predict_proba() below truncates every seed to this many trees
        # (1-indexed tree count) via iteration_range=(0, N). None means "use
        # each estimator's own full/early-stopped tree count" (old behaviour).
        self._iteration_range = (
            (0, self.consensus_iteration_) if self.consensus_iteration_ is not None else None
        )

    def predict_proba(self, X):
        if self._iteration_range is not None:
            probs = np.mean(
                [e.predict_proba(X, iteration_range=self._iteration_range)
                 for e in self.estimators_],
                axis=0,
            )
        else:
            probs = np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # FEATURE-METADATA PROXY FIX: every seed is trained on the same params,
    # data, and columns (only random_state differs — see class docstring),
    # so estimators_[0] is representative of them all. Without these,
    # getattr(self, "feature_names_in_", ...) / "n_features_in_" on the
    # wrapper silently fall back to their default (missing/None) instead of
    # raising OR returning the real values, which lets callers that build
    # extra regressor-only columns based on these attributes (see
    # BaggedXGBRegressor below, and explosion_predictor.py's
    # predict_with_targets()) silently skip columns the underlying model
    # actually needs.
    @property
    def feature_names_in_(self):
        return self.estimators_[0].feature_names_in_

    @property
    def n_features_in_(self):
        return self.estimators_[0].n_features_in_

    def get_booster(self):
        return self.estimators_[0].get_booster()

    def get_feature_importance(self, feature_names=None):
        """Average each seed's gain-importance scores (raw XGBoost 'fN' keys,
        or feature_names[N] if provided). Used by compute_feature_importance()
        in ml_retrain_model.py instead of a single get_booster() call."""
        totals = {}
        counts = {}
        for est in self.estimators_:
            scores = est.get_booster().get_score(importance_type="gain")
            for feat, score in scores.items():
                totals[feat] = totals.get(feat, 0.0) + score
                counts[feat] = counts.get(feat, 0) + 1
        # Average over the seeds where that feature actually appeared in a
        # split (not over n_seeds_) — a feature unused by some seeds isn't
        # penalised just for those seeds' trees, matching how gain importance
        # already only counts realised splits within a single model.
        return {feat: totals[feat] / counts[feat] for feat in totals}

    # Explicit pickle support: without these, pickle can call __getattr__
    # during state restoration before __dict__ is populated, causing
    # AttributeError (same pattern as _PriorCorrectedModel).
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class BaggedXGBRegressor:
    """
    Averages predict() across N XGBRegressor models trained with different
    random_state values (same params, same data, same eval_set — only the
    seed differs). Same seed-variance-cancelling rationale as
    BaggedXGBClassifier; see that class's docstring.

    Exposes predict() / best_iteration / best_score so it's a drop-in
    replacement for a single XGBRegressor in train_gain_regressor() and at
    inference time in explosion_predictor.py.
    """

    def __init__(self, estimators):
        if not estimators:
            raise ValueError("BaggedXGBRegressor needs at least one fitted estimator")
        self.estimators_ = list(estimators)
        self.n_seeds_ = len(self.estimators_)
        self.seeds_ = [
            getattr(e, "random_state", None) for e in self.estimators_
        ]
        _iters = [int(e.best_iteration) for e in self.estimators_
                  if getattr(e, "best_iteration", None) is not None]
        _scores = [float(e.best_score) for e in self.estimators_
                   if getattr(e, "best_score", None) is not None]
        self.best_iteration = int(round(np.mean(_iters))) if _iters else None
        self.best_score = float(np.mean(_scores)) if _scores else None
        self.best_iteration_std_ = float(np.std(_iters)) if _iters else None
        self.best_score_std_ = float(np.std(_scores)) if _scores else None

    def predict(self, X):
        preds = np.mean([e.predict(X) for e in self.estimators_], axis=0)
        return preds

    def get_feature_importance(self, feature_names=None):
        totals = {}
        counts = {}
        for est in self.estimators_:
            scores = est.get_booster().get_score(importance_type="gain")
            for feat, score in scores.items():
                totals[feat] = totals.get(feat, 0.0) + score
                counts[feat] = counts.get(feat, 0) + 1
        return {feat: totals[feat] / counts[feat] for feat in totals}

    # FEATURE-METADATA PROXY FIX: see BaggedXGBClassifier.feature_names_in_
    # above — same rationale. Without this, explosion_predictor.py can't
    # detect that this regressor was trained with the extra 'log_price' /
    # 'clf_proba' columns (ml_retrain_model.py's "REGRESSOR-ONLY ... FEATURE"
    # blocks), and silently predicts on an incomplete feature row instead of
    # appending them — the root cause of the collapsed/garbage gain outputs.
    @property
    def feature_names_in_(self):
        return self.estimators_[0].feature_names_in_

    @property
    def n_features_in_(self):
        return self.estimators_[0].n_features_in_

    def get_booster(self):
        return self.estimators_[0].get_booster()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
