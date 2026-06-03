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

import numpy as np


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

    def __getattr__(self, name):
        # Guard against infinite recursion during unpickling: if __dict__ is
        # not yet populated (pickle calls __getattr__ before __setstate__/
        # __init__ has run), raise AttributeError immediately rather than
        # forwarding to self._base (which would call __getattr__ again).
        if name.startswith("_") and "_base" not in self.__dict__:
            raise AttributeError(name)
        # Forward any other attribute lookups to the base model so code that
        # probes e.g. best_iteration, best_score, feature_importances_ works.
        return getattr(self._base, name)
