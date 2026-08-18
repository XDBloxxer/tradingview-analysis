"""
prior_corrected_model.py
------------------------
Shared home for _PriorCorrectedModel so that joblib/pickle can always
resolve the class by its fully-qualified name
(src.ml_predictor.prior_corrected_model._PriorCorrectedModel) regardless
of which script is running as __main__.

Background
~~~~~~~~~~
joblib (and pickle) serialise a class instance by recording the module path
and class name.  When the model was originally saved by ml_retrain_model.py
it was recorded as ``__main__._PriorCorrectedModel`` because that script was
__main__ at save time.  Later, when ml_screen_and_predict.py calls
``joblib.load()``, Python tries to import ``__main__._PriorCorrectedModel``
— but *now* __main__ is ml_screen_and_predict.py, which has no such class,
causing the error:

    Can't get attribute '_PriorCorrectedModel' on
    <module '__main__' from '.../ml_screen_and_predict.py'>

Fix: define the class here (a stable, importable location) and import it
into both ml_retrain_model.py and ml_screen_and_predict.py (the latter via
explosion_predictor.py).  joblib will then always find it via
``src.ml_predictor.prior_corrected_model._PriorCorrectedModel``.

NOTE: Existing .pkl files saved with the old ``__main__`` path will still
fail until the model is retrained (or re-saved) with this module in place.
To migrate without retraining see the one-liner in the project README.
"""

from types import SimpleNamespace

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class _CrossFittedBlendedCalibrator:
    """
    Cross-fitted blend of isotonic regression and Platt (sigmoid) scaling.

    Motivation
    ~~~~~~~~~~
    A single isotonic fit on a small calibration set (e.g. ~2k rows / ~300
    positives) is a nonparametric step function. With that few points it is
    jumpy: a handful of rows landing on one side of a fixed split instead of
    the other can shift a breakpoint enough to leave a wide, unreachable gap
    in the calibrated-probability range (e.g. nothing ever lands in
    0.667-1.0) purely as an artifact of which rows happened to be in the
    calibration split — not a real property of the model.

    This class removes that split-sensitivity two ways:

      1. Cross-fitting: instead of ONE isotonic regressor fit on all of
         X_cal/y_cal, fit K isotonic regressors, each on K-1 folds, and
         average their held-out-style predictions at inference. This acts
         like a bagged calibrator — the ensemble's step breakpoints don't
         depend on a single fixed split, so gaps average out.
      2. Blending: K sigmoid (Platt-scaling) calibrators are cross-fitted
         the same way and averaged with the isotonic ensemble. Sigmoid is a
         smooth parametric curve — it cannot produce unreachable plateaus —
         so blending it in fills in whatever gaps remain from the isotonic
         side while isotonic still contributes most of the weight (it's
         distribution-free and generally more accurate given enough data).

    blend_weight is the weight given to the isotonic ensemble; the sigmoid
    ensemble gets (1 - blend_weight).
    """

    def __init__(self, base_model, iso_models, sig_models, blend_weight=0.7):
        self._base = base_model
        self._iso_models = list(iso_models)
        self._sig_models = list(sig_models)
        self._blend_weight = float(blend_weight)
        self.classes_ = base_model.classes_

        # Downstream code (compute_feature_importance(), the best_iteration/
        # best_val_auc metadata lookup) unwraps calibration wrappers via
        # `model.calibrated_classifiers_[0].estimator` when
        # hasattr(model, "calibrated_classifiers_"). Expose the same shape
        # here (pointing at the raw XGBClassifier/BaggedXGBClassifier) so
        # those call sites keep working unchanged.
        raw_estimator = base_model
        if hasattr(base_model, "calibrated_classifiers_"):
            raw_estimator = base_model.calibrated_classifiers_[0].estimator
        self.calibrated_classifiers_ = [SimpleNamespace(estimator=raw_estimator)]

    def _raw_scores(self, X):
        return np.asarray(self._base.predict_proba(X)[:, 1])

    def predict_proba(self, X):
        raw = self._raw_scores(X)
        iso_pred = np.mean([m.predict(raw) for m in self._iso_models], axis=0)
        sig_pred = np.mean(
            [m.predict_proba(raw.reshape(-1, 1))[:, 1] for m in self._sig_models],
            axis=0,
        )
        blended = self._blend_weight * iso_pred + (1.0 - self._blend_weight) * sig_pred
        blended = np.clip(blended, 0.0, 1.0)
        return np.column_stack([1.0 - blended, blended])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # Explicit pickle support: without these, pickle calls __getattr__ during
    # state restoration before __dict__ is populated, causing AttributeError.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        if "_base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._base, name)


def fit_cross_fitted_blended_calibrator(
    base_model,
    X_cal,
    y_cal,
    n_splits: int = 5,
    blend_weight: float = 0.7,
    random_state: int = 42,
) -> tuple["_CrossFittedBlendedCalibrator", int]:
    """
    Fit a cross-fitted, isotonic/sigmoid-blended calibrator on (X_cal, y_cal).

    Replaces the old single CalibratedClassifierCV(method="isotonic",
    cv="prefit") fit — see _CrossFittedBlendedCalibrator docstring for why.

    n_splits is reduced automatically if the calibration set doesn't have
    enough positives/negatives to support it (at least ~10 examples of the
    rarer class per fold). Returns (calibrator, n_splits_actually_used).
    """
    y_arr = np.asarray(y_cal)
    raw = np.asarray(base_model.predict_proba(X_cal)[:, 1])

    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    max_splits_by_class = max(2, min(n_pos, n_neg) // 10)
    n_splits = int(max(2, min(n_splits, max_splits_by_class)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    iso_models, sig_models = [], []
    for train_idx, _ in skf.split(raw.reshape(-1, 1), y_arr):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw[train_idx], y_arr[train_idx])
        iso_models.append(iso)

        sig = LogisticRegression(solver="lbfgs")
        sig.fit(raw[train_idx].reshape(-1, 1), y_arr[train_idx])
        sig_models.append(sig)

    calibrator = _CrossFittedBlendedCalibrator(
        base_model=base_model,
        iso_models=iso_models,
        sig_models=sig_models,
        blend_weight=blend_weight,
    )
    return calibrator, n_splits


class _PriorCorrectedModel:
    """
    Thin wrapper around a CalibratedClassifierCV that applies Bayes
    prior-probability correction to predict_proba output.

    Must be defined at module level (not inside a function) so that joblib
    can pickle it via its fully-qualified name.

    The correction shifts calibrated probabilities to account for the
    base-rate mismatch between the val/calibration set (positive rate ~10-25%)
    and the screened inference universe (positive rate ~30-50%):

        odds_corrected = odds_calibrated * odds_ratio
        p_corrected    = odds_corrected / (1 + odds_corrected)

    where odds_ratio = (p_inf / (1-p_inf)) / (p_cal / (1-p_cal)).
    """

    def __init__(self, base, odds_ratio: float):
        self._base       = base
        self._odds_ratio = float(odds_ratio)
        self.classes_    = base.classes_
        # Forward attributes needed by CalibratedClassifierCV unwrap logic in
        # explosion_predictor.py and compute_feature_importance.
        if hasattr(base, "calibrated_classifiers_"):
            self.calibrated_classifiers_ = base.calibrated_classifiers_

    def predict_proba(self, X):
        raw    = self._base.predict_proba(X)
        p      = raw[:, 1]
        odds   = p / np.clip(1.0 - p, 1e-9, None) * self._odds_ratio
        p_corr = odds / (1.0 + odds)
        return np.column_stack([1.0 - p_corr, p_corr])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # Explicit pickle support: without these, pickle calls __getattr__ during
    # state restoration before __dict__ is populated, causing AttributeError.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        # Only reached for attributes not in __dict__. After __setstate__
        # runs, _base is in __dict__ so this is only hit for genuine forwards
        # (e.g. feature_importances_, best_iteration).
        if "_base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._base, name)
