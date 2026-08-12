"""
Explosion Predictor

FIXES IN THIS VERSION (2026-03-02 v4 + 2026 RC4/RC5/RC6 gain fixes + 2026-06-03 RC7/RC8/RC9):

FIX 1 — Auto-detect which feature prefix the loaded model uses.

FIX 2 — Bimodal/compression detection now also handles narrow probability bands.

FIX 3 — _classify_signals_relative operates on full distribution.

FIX 4 — Gain regressor receives SCALED features.

RC4 FIX — Relaxed std guard:
  The previous `if predicted_gains.std() < 1.0 → disable regressor` guard was
  too aggressive. Because RC1/RC2/RC3 were not yet fixed, the regressor
  legitimately produced narrow ranges (18–22%) and was always disabled,
  falling back to rule-based estimates forever. The guard is now lowered to
  0.5 with a minimum sample count requirement, so a regressor that produces
  even modest variation is kept rather than discarded. Additionally, the
  guard now logs a warning rather than silently nulling self.regressor, so
  the disable is visible in logs.

RC6 FIX — High-probability clustering detection (post-filter calibration problem):
  AUC training + scale_pos_weight pushes probabilities high for most candidates
  that pass the screener filters. The previous _detect_bimodal only caught
  LOW-end compression (>85% below 0.50) and narrow bands (std < 0.02). It
  missed the mirror-image case where probabilities cluster at the HIGH end,
  causing SIGNAL_THRESHOLDS["STRONG BUY"]=0.90 to trigger for 60%+ of stocks
  and become meaningless.

  The root cause was a two-part miscalibration in ml_retrain_model.py:
    (a) SPW_MAX=10.0 over-weighted positives, pushing raw scores too high.
    (b) Isotonic calibration on the val set (positive rate ~10-25%) was not
        adjusted for the screened inference universe's higher positive rate
        (~30-50%), so it under-corrected the systematic over-confidence.

  Both causes are now fixed in ml_retrain_model.py (RC6 revised):
    (a) SPW_MAX restored to 5.0.
    (b) Prior-probability correction (Bayes odds-ratio shift) applied on top
        of isotonic calibration to account for the base-rate mismatch.

  The _detect_bimodal modes D and E below are retained as a runtime safety net
  in case the calibration is misconfigured (e.g. SCREENER_POSITIVE_RATE is
  stale) or the model is loaded from a pkl that pre-dates the RC6 fix. When
  they fire, relative ranking is used as a fallback so that signals remain
  meaningful even with a miscalibrated model.

  Mode D: High-probability clustering — if >=50% of predictions score >=0.90
  (the STRONG BUY threshold), absolute thresholds are no longer discriminating
  and relative ranking is activated.

  Mode E: High mean probability — if mean probability >=0.80, the model's
  decision boundary has shifted so far up that absolute thresholds lose
  meaning even if clustering isn't strictly above 0.90.

RC5 FIX — Isotonic regression calibration for the gain fallback:
  The previous fallback bucketed predictions into Low/Medium/High/Very High
  and used the median per bucket. With most predictions clustering in the
  0.60–0.80 range, all buckets converged on ~20–30% regardless of signal
  strength, giving identical gain estimates across all stocks.

  The new fallback fits an IsotonicRegression on (probability, actual_gain_pct)
  pairs from historical data — this is a monotone, non-parametric curve that
  maps probability → expected gain without the bucket-collapse problem. If
  isotonic regression cannot be fit (too few points), it falls back to a
  simple linear interpolation over the observed quantiles, which at minimum
  preserves rank order across the probability range.

  The rule-based _estimate_target_gain() is retained as the final backstop.

RC7 FIX — Tie-breaking bug in _classify_signals_relative (2026-06-03):
  ROOT CAUSE: When many stocks share the same probability (e.g. 25/44 stocks
  all saturating at 0.9098 due to miscalibrated _odds_ratio in the pkl), the
  default rank(pct=True) uses method='average', which averages the ranks of
  all tied entries. Stocks ranked 20–44 (all with prob=0.9098) receive an
  averaged percentile of ~0.73, placing them between RELATIVE_HOLD_PCT=0.60
  and RELATIVE_BUY_PCT=0.80. Result: 25 STRONG BUY stocks from the probability
  histogram were all demoted to HOLD, and the signal breakdown showed 0 STRONG
  BUY / 0 BUY / 25 HOLD — the opposite of what the distribution implied.

  FIX: Use method='max' so all tied entries receive the HIGHEST rank in their
  tie group. Every stock tied at the top gets percentile=1.0, correctly placing
  them in the STRONG BUY or BUY bucket. This also ensures that when
  probabilities are healthy and ties are rare, behaviour is unchanged.

RC8 FIX — RELATIVE_STRONG_BUY_PCT lowered from 0.95 → 0.90 (2026-06-03):
  With method='average' (the old default) and a 44-stock batch, 0.95 required
  the top 2.2 stocks to qualify — fine in theory but rounding meant only 2
  ever got STRONG BUY, and with ties the count could drop to 0. Lowering to
  0.90 (top 10%) gives ~4–5 STRONG BUY picks in a typical 44-stock batch,
  which is still selective (top decile) while being robust to small batch sizes
  and minor rank ties. RELATIVE_BUY_PCT is unchanged at 0.80 (top 10–20%).

RC9 FIX — Mode D threshold tightened from 0.50 → 0.40 (2026-06-03):
  The previous threshold (>=50% of stocks at STRONG BUY level) was too
  permissive: a model needed to be severely miscalibrated before relative
  ranking activated. Lowering to 0.40 catches the saturated-probability case
  earlier (e.g. 25/44 = 56.8% in the failing run, which should have triggered
  relative ranking but the method='average' tie bug masked the effect). With
  RC7's method='max' fix in place, the two changes together ensure both the
  detection and the classification are correct.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib

# Import _PriorCorrectedModel into this namespace so joblib/pickle can resolve
# the class when loading a best_model.pkl that was saved by ml_retrain_model.py.
# Without this import the load fails with:
#   Can't get attribute '_PriorCorrectedModel' on <module '__main__' ...>
# because pickle looks up the class by module path at unpickling time and
# src.ml_predictor.prior_corrected_model must already be imported.
from src.ml_predictor.prior_corrected_model import _PriorCorrectedModel  # noqa: F401
from src.ml_predictor.symbol_demeaning import (
    demean_live_features,
    load_symbol_baselines,
    DEFAULT_BASELINE_PATH as _SYMBOL_DEMEAN_BASELINE_PATH,
)
from src.ml_predictor.feature_scaling import (
    apply_winsor_bounds,
    scale_with_fitted_scaler,
    normalise_t1_features,
)

_META_COLS = {"symbol", "exchange"}

SIGNAL_THRESHOLDS = {
    "STRONG BUY": 0.90,
    "BUY":        0.70,
    "HOLD":       0.50,
}

# Relative-ranking percentile thresholds (used when _is_bimodal = True)
# RC8 FIX: Lowered RELATIVE_STRONG_BUY_PCT from 0.95 → 0.90.
# With method='average' (old default) and a 44-stock batch, 0.95 meant only
# the top ~2 stocks qualified. Rank ties could reduce this to 0. At 0.90
# (top 10%) a typical 44-stock batch yields ~4 STRONG BUY, which is still
# selective while being robust to small batch sizes and minor rank ties.
RELATIVE_STRONG_BUY_PCT = 0.90   # top 10% (~4-5 stocks per 44-stock batch)
RELATIVE_BUY_PCT        = 0.80   # top 10-20% (~4-5 stocks per 44-stock batch)
RELATIVE_HOLD_PCT       = 0.60   # top 20-40%

# Compression detection thresholds
COMPRESSION_THRESHOLD      = 0.85   # >85% below 0.50 → use relative ranking
NARROW_BAND_STD_THRESHOLD  = 0.02   # std < 0.02 regardless of level → use relative ranking

# RC6: High-probability clustering detection — runtime safety net.
# With a correctly calibrated model (SPW_MAX=5.0 + prior correction in
# ml_retrain_model.py) these thresholds should rarely be hit. They are retained
# to catch models loaded from pre-RC6 pkl files or misconfigured SCREENER_POSITIVE_RATE.
# RC9 FIX: Tightened HIGH_PROB_CLUSTERING_RATE from 0.50 → 0.40.
# The previous 0.50 was too permissive: a run where 56.8% of stocks scored
# ≥0.90 should have been caught much earlier. At 0.40 the safety net
# activates as soon as 40% of the batch saturates the STRONG BUY level,
# switching to relative ranking before the distribution becomes completely
# unusable as an absolute discriminator.
HIGH_PROB_CLUSTERING_RATE  = 0.40   # >=40% at or above STRONG BUY → relative ranking
HIGH_PROB_MEAN_THRESHOLD   = 0.80   # mean >=0.80 → absolute thresholds likely meaningless

# Original mid-range sparsity check (bimodal toward extremes)
BIMODAL_MIDRANGE             = (0.15, 0.85)
BIMODAL_MIN_MIDRANGE_COUNT   = 5

# RC4 FIX: Lowered from 1.0 to 0.5.
# A regressor that produces 0.5% std is still near-useless, but one producing
# e.g. 3–5% std should NOT be discarded. The previous 1.0 threshold was
# disabling competent regressors whenever the gain distribution was moderate.
REGRESSOR_MIN_STD_THRESHOLD = 0.5
# Minimum predictions needed before the std guard fires
REGRESSOR_MIN_PRED_COUNT    = 10

# RC5: minimum historical samples needed to fit isotonic calibration
ISOTONIC_MIN_SAMPLES = 30


def _norm(s: str) -> str:
    """Normalize for matching: lowercase + dots to underscores."""
    return s.lower().replace(".", "_")


# ---------------------------------------------------------------------------
# RC5 FIX: Isotonic calibration of gain estimates
# ---------------------------------------------------------------------------

def _build_isotonic_gain_calibrator(historical_df: pd.DataFrame):
    """
    RC5 FIX: Fit an IsotonicRegression on (predicted_probability, actual_gain_pct)
    from historical accuracy data.

    IsotonicRegression is monotone and non-parametric — it learns the actual
    relationship between probability and gain without assuming linearity, and
    avoids the bucket-collapse problem of the old median-per-bucket approach.

    Returns a callable f(prob_array) → gain_array, or None if not enough data.
    """
    if historical_df is None or historical_df.empty:
        return None

    # Need predicted_probability and actual_gain_pct
    prob_col = next(
        (c for c in ["predicted_probability", "probability", "explosion_probability"]
         if c in historical_df.columns),
        None,
    )
    gain_col = next(
        (c for c in ["actual_gain_pct", "actual_high_pct"] if c in historical_df.columns),
        None,
    )

    if prob_col is None or gain_col is None:
        return None

    pairs = historical_df[[prob_col, gain_col]].copy()
    pairs = pairs.rename(columns={prob_col: "prob", gain_col: "gain"})
    pairs["prob"] = pd.to_numeric(pairs["prob"], errors="coerce")
    pairs["gain"] = pd.to_numeric(pairs["gain"], errors="coerce")
    pairs = pairs.dropna()
    pairs = pairs[(pairs["prob"] > 0) & (pairs["prob"] <= 1) & (pairs["gain"].abs() < 500)]

    if len(pairs) < ISOTONIC_MIN_SAMPLES:
        logging.getLogger(__name__).info(
            f"RC5: Only {len(pairs)} historical pairs — need {ISOTONIC_MIN_SAMPLES} "
            "for isotonic calibration. Falling back to quantile interpolation."
        )
        return _build_quantile_interpolator(pairs)

    try:
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir.fit(pairs["prob"].values, pairs["gain"].values)

        # Sanity check: calibrated std across the probability range
        test_probs = np.linspace(0.05, 0.99, 50)
        test_gains = ir.predict(test_probs)
        calib_std = float(test_gains.std())

        logger = logging.getLogger(__name__)
        logger.info(
            f"RC5: IsotonicRegression fitted on {len(pairs)} pairs. "
            f"Calibrated gain range: {test_gains.min():.1f}%–{test_gains.max():.1f}% "
            f"(std={calib_std:.2f}%)"
        )

        if calib_std < 1.0:
            logger.warning(
                f"RC5: IsotonicRegression std={calib_std:.2f}% is low. "
                "Historical gain data may still be compressed (RC2 not yet taking effect). "
                "Falling back to quantile interpolation for better rank separation."
            )
            return _build_quantile_interpolator(pairs)

        return lambda probs: ir.predict(np.asarray(probs, dtype=float))

    except ImportError:
        logging.getLogger(__name__).warning(
            "RC5: sklearn IsotonicRegression not available — using quantile interpolation"
        )
        return _build_quantile_interpolator(pairs)
    except Exception as e:
        logging.getLogger(__name__).warning(f"RC5: IsotonicRegression failed ({e}) — using quantile interpolation")
        return _build_quantile_interpolator(pairs)


def _build_quantile_interpolator(pairs: pd.DataFrame):
    """
    RC5 fallback: build a piecewise-linear interpolation over observed quantiles
    of (probability, gain) so that rank order is preserved even when isotonic
    regression is unavailable or produces a flat curve.

    At minimum this ensures STRONG BUY stocks get higher gain estimates than
    BUY stocks get higher than HOLD stocks, which was not guaranteed by the
    old median-per-bucket approach.
    """
    if pairs.empty:
        return None

    quantile_points = np.linspace(0.05, 0.95, 19)
    prob_quantiles = np.quantile(pairs["prob"].values, quantile_points)
    gain_quantiles = np.quantile(pairs["gain"].values, quantile_points)

    # Ensure monotone (isotone) by taking cumulative max
    gain_quantiles = np.maximum.accumulate(gain_quantiles)

    logger = logging.getLogger(__name__)
    logger.info(
        f"RC5: Quantile interpolator fitted. "
        f"Gain range: {gain_quantiles.min():.1f}%–{gain_quantiles.max():.1f}% "
        f"(std={gain_quantiles.std():.2f}%)"
    )

    def _interpolate(probs):
        probs_arr = np.asarray(probs, dtype=float).clip(0.01, 0.99)
        return np.interp(probs_arr, prob_quantiles, gain_quantiles)

    return _interpolate


class ExplosionPredictor:

    def __init__(self, model_dir: str = "ml_models"):
        self.logger        = logging.getLogger(__name__)
        self.model_dir     = Path(model_dir)
        self.model         = None
        self.regressor     = None
        self.scaler        = None
        self.feature_names: List[str] = []
        self.metadata:      dict = {}
        self._norm_to_feature: Dict[str, str] = {}
        self._regressor_n_features: Optional[int] = None
        self._is_bimodal   = False
        self._diag_done    = False
        # FIX 1: detected prefix for external callers
        self._model_feature_prefix: str = "t1_close"
        # Symbol-fingerprint demeaning (2026-08-11): baselines written by
        # ml_retrain_model.py's most recent retrain — see symbol_demeaning.py.
        # Loaded once here rather than per-predict() call; a fresh retrain
        # means a fresh ExplosionPredictor instance anyway (new model file).
        self._hv_symbol_baselines: dict = load_symbol_baselines(_SYMBOL_DEMEAN_BASELINE_PATH)

        self._load_model()

    # -------------------------------------------------------------------------
    # FIX 1: Expose which prefix the model uses so screen script can match it
    # -------------------------------------------------------------------------

    @property
    def model_feature_prefix(self) -> str:
        return self._model_feature_prefix

    def _detect_model_prefix(self) -> str:
        if not self.feature_names:
            return "t1_close"

        counts = {
            "t1_close": sum(1 for f in self.feature_names if f.startswith("t1_close_")),
            "t1_open":  sum(1 for f in self.feature_names if f.startswith("t1_open_")),
            "t3":       sum(1 for f in self.feature_names if f.startswith("t3_")),
            "t5":       sum(1 for f in self.feature_names if f.startswith("t5_")),
            "t10":      sum(1 for f in self.feature_names if f.startswith("t10_")),
        }

        total = sum(counts.values())
        if total == 0:
            return "t1_close"

        dominant = max(counts, key=counts.get)

        self.logger.info("Model feature prefix breakdown:")
        for pfx, cnt in counts.items():
            pct = cnt / total * 100 if total else 0
            self.logger.info(f"  {pfx:12s}: {cnt:4d}  ({pct:.1f}%)")
        self.logger.info(f"  → dominant prefix: '{dominant}' — screen script will use this")

        has_t1   = counts["t1_close"] > 0 or counts["t1_open"] > 0
        has_flat = counts["t3"] > 0 or counts["t5"] > 0 or counts["t10"] > 0

        if has_t1 and has_flat:
            self.logger.info(
                "  → HYBRID model detected (t1_ + t3/t5/t10 features). "
                "Returning 't1_close' as primary prefix."
            )
            return "t1_close"

        return dominant

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self):
        model_path     = self.model_dir / "best_model.pkl"
        regressor_path = self.model_dir / "gain_regressor.pkl"
        scaler_path    = self.model_dir / "scaler.pkl"
        metadata_path  = self.model_dir / "model_metadata.json"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model files not found in {self.model_dir}")

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if regressor_path.exists():
            self.regressor = joblib.load(regressor_path)
            try:
                self._regressor_n_features = self.regressor.n_features_in_
            except AttributeError:
                try:
                    self._regressor_n_features = self.regressor.get_booster().num_features()
                except Exception:
                    self._regressor_n_features = None
            self.logger.info(f"Loaded gain regressor (expects {self._regressor_n_features} features)")
        else:
            self.regressor = None
            self.logger.warning(
                "gain_regressor.pkl not found — will use calibrated gain estimates. "
                "Run ml_retrain_model.py once ≥30 rows have gain data."
            )

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        # SPARSE-COLUMN CONSISTENCY FIX: build_scaler() in ml_retrain_model.py
        # determines "sparse" (mostly-missing in the training set — historically
        # t1_ intraday columns back when t1 coverage was inconsistent; now
        # whatever genuinely has low coverage, e.g. long-window indicators for
        # short-history stocks) columns from coverage computed on the *training
        # set*, and that exact list is what
        # decides which columns get NaN restored after scaling (so XGBoost uses
        # its learned missing-value branch instead of a mean-imputed value).
        # Previously _scale_features() below recomputed "sparse" from the
        # coverage of whatever batch was being predicted — for a single-row
        # prediction that degenerates to "columns that are NaN in this row",
        # and for a multi-row screening batch it can disagree with training
        # coverage entirely, giving some features a different representation
        # at inference than the model was trained on. Loading the persisted
        # list here (written by save_outputs() at training time) makes the two
        # paths use identical sparse-column membership.
        self._trained_sparse_cols = list(self.metadata.get("sparse_cols", []))
        if not self._trained_sparse_cols:
            self.logger.warning(
                "model_metadata.json has no 'sparse_cols' (older model or "
                "not yet retrained with the fix) — _scale_features() will "
                "fall back to per-batch sparse-column inference, which can "
                "be inconsistent with how the model was trained. Retrain "
                "with the updated ml_retrain_model.py to populate this."
            )

        # WINSORIZATION CONSISTENCY FIX: build_scaler() in ml_retrain_model.py
        # (via src.ml_predictor.feature_scaling) winsorizes each column to a
        # [0.5th, 99.5th] percentile band fit on X_train BEFORE fitting the
        # scaler, so the scaler's mean_/std_ are never dragged by a handful
        # of outlier rows. Persisted here so _scale_features() applies the
        # SAME train-derived bounds to live features before calling
        # scaler.transform() — otherwise a live outlier value training would
        # have clipped flows through unclipped, landing on a scale the model
        # never saw during training.
        raw_winsor_bounds = self.metadata.get("winsor_bounds", {})
        self._trained_winsor_bounds = {
            col: (float(lo), float(hi)) for col, (lo, hi) in raw_winsor_bounds.items()
        }
        if not self._trained_winsor_bounds:
            self.logger.warning(
                "model_metadata.json has no 'winsor_bounds' (older model or "
                "not yet retrained with the fix) — live features will be "
                "scaled without the outlier clip training used. Retrain "
                "with the updated ml_retrain_model.py to populate this."
            )

        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.scaler.n_features_in_)]
            self.logger.warning("Scaler has no feature_names_in_ - using positional names.")

        self._build_lookup()

        self._model_feature_prefix = self._detect_model_prefix()

        classifier_n = self.scaler.n_features_in_
        self.logger.info(
            f"Classifier/scaler expects {classifier_n} features; "
            f"regressor expects {self._regressor_n_features} features"
        )

        # FEATURE-COUNT CHECK FIX: ml_retrain_model.py deliberately trains the
        # gain regressor on extra features — 'log_price' and 'clf_proba' —
        # that the classifier/scaler never sees (see the "REGRESSOR-ONLY
        # log_price FEATURE" and "REGRESSOR-ONLY clf_proba FEATURE" blocks in
        # ml_retrain_model.py). predict_with_targets() below already knows how
        # to append both before calling self.regressor.predict(), so a
        # regressor with classifier_n + {1 or 2} features (with 'log_price'
        # and/or 'clf_proba' in its feature_names_in_) is EXPECTED, not a
        # mismatch. Only disable the regressor when the count differs in a
        # way that isn't explained by these known, intentional offsets.
        regressor_feature_names_raw = getattr(self.regressor, "feature_names_in_", None)
        regressor_feature_names = (
            list(regressor_feature_names_raw) if regressor_feature_names_raw is not None else []
        )
        regressor_has_log_price = "log_price" in regressor_feature_names
        regressor_has_clf_proba = "clf_proba" in regressor_feature_names
        expected_regressor_n = (
            classifier_n
            + (1 if regressor_has_log_price else 0)
            + (1 if regressor_has_clf_proba else 0)
        )
        _extra_desc = " + ".join(
            n for n, present in
            (("log_price", regressor_has_log_price), ("clf_proba", regressor_has_clf_proba))
            if present
        )

        if (self._regressor_n_features is not None
                and self._regressor_n_features != expected_regressor_n):
            self.logger.warning(
                f"Regressor feature count ({self._regressor_n_features}) != "
                f"expected ({expected_regressor_n}, classifier={classifier_n} "
                f"{('+ ' + _extra_desc) if _extra_desc else ''}). "
                f"Regressor DISABLED — retrain both together."
            )
            self.regressor = None
        elif _extra_desc:
            self.logger.info(
                f"Regressor expects {self._regressor_n_features} features "
                f"({classifier_n} shared with classifier + {_extra_desc}) — this is "
                f"the expected, intentional offset. Regressor ENABLED."
            )

        has_t1   = any("t1_open" in f or "t1_close" in f for f in self.feature_names)
        has_flat = any(
            f.startswith("t3_") or f.startswith("t5_") or f.startswith("t10_")
            for f in self.feature_names
        )
        if has_flat and has_t1:
            self.logger.info("Model type: HYBRID (T-3/T-5/T-10 + T-1 open/close)")
        elif has_flat:
            self.logger.info("Model type: CSV-ONLY (T-3/T-5/T-10)")
        elif has_t1:
            self.logger.info("Model type: DATABASE-ONLY (T-1 open/close)")
        else:
            self.logger.warning("Could not identify model type. First 10 features: %s",
                                self.feature_names[:10])

    def _build_lookup(self):
        self._norm_to_feature = {_norm(f): f for f in self.feature_names}

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def _log_feature_diagnostics(self, feature_df: pd.DataFrame, match_log: dict):
        self.logger.info("")
        self.logger.info("=" * 72)
        self.logger.info("FEATURE DIAGNOSTIC REPORT")
        self.logger.info("=" * 72)

        direct  = sum(1 for v in match_log.values() if v == "direct")
        norm_m  = sum(1 for v in match_log.values() if v == "norm")
        default = sum(1 for v in match_log.values() if v == "default")
        self.logger.info(
            f"Match breakdown: direct={direct}  norm={norm_m}  "
            f"default(constant)={default}  meta={len(match_log)-direct-norm_m-default}"
        )
        self.logger.info("")

        groups = [
            ("t1_close_", [c for c in feature_df.columns if c.startswith("t1_close_")]),
            ("t1_open_",  [c for c in feature_df.columns if c.startswith("t1_open_")]),
            ("t3_",       [c for c in feature_df.columns if c.startswith("t3_")]),
            ("t5_",       [c for c in feature_df.columns if c.startswith("t5_")]),
            ("t10_",      [c for c in feature_df.columns if c.startswith("t10_")]),
        ]

        for group_name, cols in groups:
            num_cols = [c for c in cols
                        if c in feature_df.columns
                        and pd.api.types.is_numeric_dtype(feature_df[c])]
            if not num_cols:
                self.logger.info(f"  {group_name:<14}  0 features")
                continue

            sub        = feature_df[num_cols].astype(float)
            col_stds   = sub.std()
            zero_var   = int((col_stds < 1e-9).sum())
            live_var   = int((col_stds >= 1e-9).sum())
            mean_std   = float(col_stds[col_stds >= 1e-9].mean()) if live_var > 0 else 0.0
            dflt_count = sum(1 for c in num_cols if match_log.get(c) == "default")

            self.logger.info(
                f"  {group_name:<14}  {len(num_cols):>3} cols | "
                f"zero-var: {zero_var:>3} | live-var: {live_var:>3} | "
                f"mean_std: {mean_std:.4f} | from_default: {dflt_count}"
            )

            live_cols = [c for c in num_cols if col_stds.get(c, 0) >= 1e-9][:5]
            if live_cols and len(feature_df) >= 3:
                for col in live_cols:
                    vals = feature_df[col].astype(float).iloc[:3].tolist()
                    self.logger.info(
                        f"    sample {col}: "
                        + "  ".join(f"{v:>10.4f}" for v in vals)
                    )

        self.logger.info("")

        fi_path = self.model_dir / "feature_importance.csv"
        if fi_path.exists():
            try:
                fi = pd.read_csv(fi_path).head(20)
                self.logger.info("  Top 20 model features by importance vs live data:")
                self.logger.info(f"  {'Feature':<45} {'Imp':>6}  {'Std':>8}  {'Src'}")
                self.logger.info("  " + "-" * 72)
                for _, row in fi.iterrows():
                    feat      = row["feature"]
                    imp       = row["importance"]
                    feat_norm = _norm(feat)
                    feat_col  = None
                    if feat in feature_df.columns:
                        feat_col = feat
                    else:
                        for col in feature_df.columns:
                            if _norm(col) == feat_norm:
                                feat_col = col
                                break
                    if feat_col is not None and feat_col in feature_df.columns:
                        std_val = float(feature_df[feat_col].astype(float).std())
                        src     = match_log.get(feat_col, match_log.get(feat, "?"))
                    else:
                        std_val = float("nan")
                        src     = "MISSING"
                    self.logger.info(
                        f"  {feat:<45} {imp:>6.4f}  {std_val:>8.4f}  {src}"
                    )
            except Exception as e:
                self.logger.warning(f"Could not read feature_importance.csv: {e}")

        self.logger.info("=" * 72)
        self.logger.info("")

    # -------------------------------------------------------------------------
    # Feature preparation
    # -------------------------------------------------------------------------

    def prepare_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Preparing features for {len(data_df)} stocks")

        # LEGACY COMPAT ONLY (2026-08-06): has_t1_features was removed from
        # ml_retrain_model.py's prepare_features() as a source-provenance
        # leak risk — new models will not have it in self.feature_names, so
        # this block is a no-op for them. It's kept only so that a model
        # trained BEFORE this change (which still expects the column) doesn't
        # break; once you retrain, this branch stops firing for that model
        # and can be deleted.
        if "has_t1_features" in self.feature_names:
            data_df = data_df.copy()
            data_df["has_t1_features"] = 1.0

        # ── FIX (2026-08-12): t1_ unit normalisation, shared with training ──
        # normalise_t1_features() (src.ml_predictor.feature_scaling) detects,
        # PER ROW, whether each t1_ price/MACD/ATR/volume column is still in
        # raw dollar/cumulative-volume scale and converts it to the same %
        # distance / % of close / volume-ratio scale multiday_feature_collector
        # writes at collection time. ml_retrain_model.py calls this on the
        # RAW loaded DataFrame (before feature selection trims columns down),
        # using t1_close_Close / t1_open_Close as the per-row close anchor.
        # Previously prediction had no equivalent step and relied entirely on
        # the assumption that live data is always already normalised by
        # intraday_data_collector.py. This must run on data_df — NOT on the
        # already-trimmed feature_df built below — because the close anchor
        # column may not itself be a selected model feature and would
        # otherwise be missing by the time normalisation ran, silently
        # disabling the price/MACD/ATR detection groups that need it.
        # Idempotent: a no-op on rows the collector already normalised, and a
        # real safety net if it didn't.
        for _t1_prefix in ("t1_close", "t1_open"):
            if any(c.startswith(f"{_t1_prefix}_") for c in data_df.columns):
                data_df = normalise_t1_features(data_df, prefix=_t1_prefix)

        input_norm_to_col: Dict[str, str] = {_norm(c): c for c in data_df.columns}

        feature_data: dict = {}
        match_log:    dict = {}
        matched       = 0
        missing_names: List[str] = []

        for feature in self.feature_names:
            feature_norm = _norm(feature)

            if feature in data_df.columns:
                feature_data[feature] = data_df[feature].values
                match_log[feature]    = "direct"
                matched += 1
            elif feature_norm in input_norm_to_col:
                feature_data[feature] = data_df[input_norm_to_col[feature_norm]].values
                match_log[feature]    = "norm"
                matched += 1
            elif feature_norm in _META_COLS:
                feature_data[feature] = 0.0
                match_log[feature]    = "meta"
            else:
                feature_data[feature] = self._get_default_value(feature, data_df)
                match_log[feature]    = "default"
                missing_names.append(feature)

        feature_df = pd.DataFrame(feature_data, index=data_df.index)

        for col in ("symbol", "exchange"):
            if col in data_df.columns:
                feature_df[col] = data_df[col].values

        coverage = (matched / len(self.feature_names) * 100) if self.feature_names else 0
        self.logger.info(
            f"Feature coverage: {coverage:.1f}% ({matched}/{len(self.feature_names)})"
        )
        if missing_names:
            self.logger.info(f"First 30 MISSING features: {missing_names[:30]}")
            sample_avail = [c for c in data_df.columns if c not in _META_COLS][:30]
            self.logger.info(f"First 30 AVAILABLE columns: {sample_avail}")
        if coverage < 50:
            self.logger.warning(
                f"LOW feature coverage ({coverage:.1f}%) — model may be using t3_ prefix "
                f"but features were built with '{self._model_feature_prefix}' prefix."
            )

        # Note: has_t1_features is injected into data_df at the top of this
        # method (before the feature loop) so it is picked up via a direct match
        # and never lands in missing_names.  No post-loop override needed.

        if not self._diag_done:
            self._log_feature_diagnostics(feature_df, match_log)
            self._diag_done = True

        # ── FIX (2026-08-11): symbol-fingerprint demeaning for HV_10/20/30 ──
        # Mirrors the training-side transform in ml_retrain_model.py's
        # prepare_features(). A live row has no in-process history to compute
        # a trailing mean from, so this subtracts each symbol's PERSISTED
        # baseline (saved by the last retrain) instead. Symbols with no
        # stored baseline are left RAW — see demean_live_features() docstring.
        if "symbol" in feature_df.columns and self._hv_symbol_baselines:
            feature_df = demean_live_features(
                feature_df, feature_df["symbol"], self._hv_symbol_baselines
            )
        elif "symbol" not in feature_df.columns:
            self.logger.info(
                "[symbol-demean] no 'symbol' column in live features — skipping HV "
                "symbol-fingerprint demeaning for this batch (raw HV values kept as-is)"
            )

        return feature_df

    def _get_default_value(self, feature: str, data: pd.DataFrame):
        f = _norm(feature)
        for pfx in ("t1_open_", "t1_close_", "t3_", "t5_", "t10_"):
            if f.startswith(pfx):
                f = f[len(pfx):]
                break
        if any(x in f for x in ("rsi", "stoch", "willr", "cci")):
            return 50.0
        if any(x in f for x in ("change", "pct", "ratio", "slope", "diff", "roc", "mom", "macd", "ao")):
            return 0.0
        if any(x in f for x in ("above", "below", "cross", "flag", "signal")):
            return 0.0
        if "volume" in f or "obv" in f:
            for col in data.columns:
                if "volume" in col.lower() and col not in _META_COLS:
                    try:
                        return float(data[col].median())
                    except Exception:
                        pass
            return 100_000.0
        if any(x in f for x in ("price", "close", "open", "high", "low", "ema", "sma", "wma", "vwap")):
            for col in data.columns:
                if "close" in col.lower() and col not in _META_COLS:
                    try:
                        return float(data[col].median())
                    except Exception:
                        pass
            return 50.0
        if any(x in f for x in ("atr", "volatility", "hv", "bb")):
            return 1.0
        return 0.0

    # -------------------------------------------------------------------------
    # NaN-safe scaling
    # -------------------------------------------------------------------------

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        expected_n = self.scaler.n_features_in_
        if X.shape[1] != expected_n:
            raise ValueError(
                f"_scale_features: X has {X.shape[1]} cols but scaler expects {expected_n}.")

        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            self.logger.warning(f"Coercing non-numeric columns to NaN: {non_numeric}")
            X = X.copy()
            for c in non_numeric:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        # SHARED-IMPLEMENTATION FIX: this used to be a hand-rolled
        # re-implementation of build_scaler()'s transform path (own
        # mean_series/fillna/sparse-restoration logic, and no winsorization
        # step at all). It now calls the exact same
        # feature_scaling.scale_with_fitted_scaler() that
        # ml_retrain_model.py uses to transform the validation split during
        # training — same winsorization bounds, same sparse-column NaN
        # restoration, same fill/transform order. There is only one
        # implementation of "how a feature gets scaled" in the whole
        # pipeline now; training and prediction both call it.
        #
        # CONSISTENCY FIX: sparse-column membership and winsorization bounds
        # both come from the training set (persisted in model_metadata.json
        # by save_outputs()), not recomputed from this inference batch — see
        # _load_model() above for why recomputing per-batch would be
        # inconsistent with how the model was trained.
        sparse_cols = getattr(self, "_trained_sparse_cols", None) or None
        winsor_bounds = getattr(self, "_trained_winsor_bounds", None) or None

        return scale_with_fitted_scaler(
            self.scaler,
            X,
            sparse_threshold_cols=sparse_cols,
            winsor_bounds=winsor_bounds,
        )

    # -------------------------------------------------------------------------
    # FIX 2: Enhanced bimodal/compression/narrow-band detection
    # -------------------------------------------------------------------------

    def _detect_bimodal(self, probabilities: np.ndarray) -> bool:
        n = len(probabilities)

        # Mode A: Low-probability compression
        below_half = int((probabilities < 0.50).sum())
        compression_rate = below_half / n if n > 0 else 0
        if compression_rate > COMPRESSION_THRESHOLD:
            self.logger.warning(
                f"LOW-PROB COMPRESSION: {compression_rate:.1%} of predictions below 0.50. "
                f"Switching to percentile-based signals."
            )
            return True

        # Mode B: Narrow probability band (FIX 2)
        prob_std = float(np.std(probabilities))
        if prob_std < NARROW_BAND_STD_THRESHOLD:
            self.logger.warning(
                f"NARROW PROBABILITY BAND: std={prob_std:.4f} < {NARROW_BAND_STD_THRESHOLD}. "
                f"Switching to percentile-based signals."
            )
            return True

        # Mode C: Bimodal toward extremes
        mid_lo, mid_hi = BIMODAL_MIDRANGE
        mid_count = int(((probabilities > mid_lo) & (probabilities < mid_hi)).sum())
        if mid_count < BIMODAL_MIN_MIDRANGE_COUNT:
            self.logger.warning(
                f"BIMODAL COLLAPSE: only {mid_count} predictions in mid-range. "
                "Switching to percentile-based signals."
            )
            return True

        # RC6 Mode D: High-probability clustering (safety net).
        # With properly calibrated models (SPW_MAX=5.0 + prior correction) this
        # mode should rarely fire.  It is retained as a runtime safety net for:
        #   - models loaded from pkl files pre-dating the RC6 calibration fix
        #   - misconfigured SCREENER_POSITIVE_RATE causing under-correction
        # When >=50% of stocks hit the STRONG BUY threshold (>=0.90), the
        # absolute threshold is no longer discriminating within today's batch
        # and relative ranking is activated.
        strong_buy_threshold = SIGNAL_THRESHOLDS["STRONG BUY"]
        above_threshold_rate = float((probabilities >= strong_buy_threshold).mean())
        if above_threshold_rate >= HIGH_PROB_CLUSTERING_RATE:
            self.logger.warning(
                f"HIGH-PROB CLUSTERING (RC6 Mode D): {above_threshold_rate:.1%} of predictions "
                f"are at or above the STRONG BUY threshold ({strong_buy_threshold}). "
                f"Absolute thresholds are not discriminating today's batch. "
                f"Switching to percentile-based signals."
            )
            return True

        # RC6 Mode E: High mean probability
        # Even without clustering above 0.90, a mean >=0.80 indicates the model's
        # decision boundary has shifted far from training calibration and
        # absolute thresholds have become nearly trivially satisfied.
        prob_mean = float(np.mean(probabilities))
        if prob_mean >= HIGH_PROB_MEAN_THRESHOLD:
            self.logger.warning(
                f"HIGH-PROB MEAN (RC6 Mode E): mean probability={prob_mean:.3f} "
                f">= {HIGH_PROB_MEAN_THRESHOLD}. Absolute thresholds have lost "
                f"discriminative power. Switching to percentile-based signals."
            )
            return True

        return False

    def _classify_signal_absolute(self, probability: float) -> str:
        if probability >= SIGNAL_THRESHOLDS["STRONG BUY"]: return "STRONG BUY"
        elif probability >= SIGNAL_THRESHOLDS["BUY"]:       return "BUY"
        elif probability >= SIGNAL_THRESHOLDS["HOLD"]:      return "HOLD"
        return "AVOID"

    def _classify_signals_relative(self, probabilities: pd.Series) -> pd.Series:
        """FIX 3: Rank-based signal classification on the FULL distribution.

        RC7 FIX: Use method='max' for rank() so that all tied entries receive
        the HIGHEST rank in their tie group instead of the average.

        With method='average' (the old default), stocks tied at probability
        0.9098 (e.g. 25 out of 44 stocks due to saturated calibration) all
        receive the averaged percentile rank of ~0.73, placing every one of
        them in the HOLD bucket (0.60–0.80) and producing 0 STRONG BUY / 0 BUY
        despite the model clearly preferring them. method='max' assigns all tied
        stocks percentile=1.0 (the rank of the last tie), so they correctly fall
        into STRONG BUY, with the lower-probability stocks correctly ranked below.
        """
        n = len(probabilities)
        signals = pd.Series("AVOID", index=probabilities.index)

        if n == 0:
            return signals

        # RC7 FIX: method='max' — tied entries receive the highest rank in
        # their tie group, not the average. This ensures saturated-probability
        # clusters are not erroneously demoted to HOLD.
        ranks = probabilities.rank(pct=True, method='max')

        signals[ranks >= RELATIVE_STRONG_BUY_PCT] = "STRONG BUY"
        signals[(ranks >= RELATIVE_BUY_PCT) & (ranks < RELATIVE_STRONG_BUY_PCT)] = "BUY"
        signals[(ranks >= RELATIVE_HOLD_PCT) & (ranks < RELATIVE_BUY_PCT)] = "HOLD"

        strong_buy_n = (signals == "STRONG BUY").sum()
        buy_n        = (signals == "BUY").sum()
        hold_n       = (signals == "HOLD").sum()

        self.logger.info(
            f"Relative signals (percentile-based, method=max): STRONG BUY={strong_buy_n}, BUY={buy_n}, "
            f"HOLD={hold_n}, AVOID={n - strong_buy_n - buy_n - hold_n}"
        )

        return signals

    # -------------------------------------------------------------------------
    # PSI drift detection â compares live feature distributions to training
    # -------------------------------------------------------------------------

    def _check_feature_drift(self, X: pd.DataFrame) -> None:
        """Compute PSI for top-10 features and warn if any exceed threshold.

        Training distributions (percentile buckets) are stored in
        model_metadata.json under the key ``top10_feature_distribution`` by
        ml_retrain_model.py.  If that key is absent (older model), this method
        is a no-op so callers need not guard against it.

        PSI formula per bucket:
            psi_i = (actual_pct - expected_pct) * ln(actual_pct / expected_pct)
        Total PSI = sum over all buckets.  PSI > 0.2 indicates significant
        distribution shift.
        """
        PSI_WARN_THRESHOLD = 0.2
        PSI_MIN_ROWS = 10  # skip check if live batch is too small to be meaningful

        train_stats: dict = self.metadata.get("top10_feature_distribution", {})
        if not train_stats:
            return  # older model without saved distribution â skip silently

        if len(X) < PSI_MIN_ROWS:
            self.logger.info(
                f"PSI drift check skipped â only {len(X)} live rows "
                f"(minimum {PSI_MIN_ROWS} required)."
            )
            return

        high_psi_features = []
        for feat, stats in train_stats.items():
            if feat not in X.columns:
                continue
            live_col = X[feat].dropna()
            if len(live_col) < PSI_MIN_ROWS:
                continue

            percentiles = stats["percentiles"]  # 11 values â 10 equal-frequency train buckets
            n_buckets = len(percentiles) - 1

            # Clip live values to the training range edges so they fall in the
            # first / last bucket rather than creating out-of-range spill.
            live_clipped = live_col.clip(percentiles[0], percentiles[-1])

            # BUGFIX: degenerate buckets (lo == hi, e.g. all-zero / heavily
            # tied features) used to be silently skipped, which dropped their
            # share of expected_pct (1/n_buckets) from the running total while
            # the live mass sitting exactly on that shared edge still landed in
            # the neighboring bucket's actual_pct (because of the clip above).
            # That desynced expected (< 1.0 total) from actual (~1.0 total),
            # producing large spurious PSI even when the live batch is drawn
            # from the same distribution as training.
            #
            # Fix: merge consecutive degenerate buckets into their neighbor so
            # each merged bucket's expected_pct reflects its combined weight
            # (merged_count / n_buckets), keeping expected and actual on a
            # consistent 0..1 scale.
            merged_buckets = []  # list of (lo, hi, weight)
            i = 0
            while i < n_buckets:
                lo, hi = percentiles[i], percentiles[i + 1]
                weight = 1
                while lo == hi and i + 1 < n_buckets:
                    i += 1
                    hi = percentiles[i + 1]
                    weight += 1
                merged_buckets.append((lo, hi, weight))
                i += 1

            psi_total = 0.0
            n_merged = len(merged_buckets)
            for idx, (lo, hi, weight) in enumerate(merged_buckets):
                if lo == hi:
                    # Entire feature is a single constant value in training
                    # (all buckets collapsed) — no meaningful PSI possible.
                    continue

                # Expected fraction: this bucket's share of training data,
                # preserving the mass of any buckets merged into it.
                expected_pct = weight / n_buckets

                # Actual fraction of live values falling in this bucket
                if idx < n_merged - 1:
                    actual_count = int(((live_clipped >= lo) & (live_clipped < hi)).sum())
                else:
                    # Last bucket is inclusive on the right edge
                    actual_count = int(((live_clipped >= lo) & (live_clipped <= hi)).sum())
                actual_pct = actual_count / len(live_clipped)

                # Smoothing: avoid log(0) â floor at a small epsilon
                actual_pct   = max(actual_pct,   1e-6)
                expected_pct = max(expected_pct, 1e-6)

                psi_total += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

            if psi_total > PSI_WARN_THRESHOLD:
                high_psi_features.append((feat, psi_total))

        if high_psi_features:
            for feat, psi in sorted(high_psi_features, key=lambda x: -x[1]):
                self.logger.warning(
                    f"FEATURE DRIFT DETECTED â PSI={psi:.3f} (threshold {PSI_WARN_THRESHOLD}) "
                    f"for feature '{feat}'. Live distribution differs significantly from "
                    f"training distribution. Predictions may be degraded. Consider retraining."
                )
        else:
            self.logger.info(
                f"PSI drift check passed â all {len(train_stats)} top features "
                f"within acceptable range (threshold {PSI_WARN_THRESHOLD})."
            )

    # -------------------------------------------------------------------------
    # Prediction — internal, returns (result_df, features_df, X_scaled)
    # -------------------------------------------------------------------------

    def _predict_internal(
        self, data_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        features_df = self.prepare_features(data_df)
        X           = features_df[self.feature_names].copy()
        self._check_feature_drift(X)  # PSI drift check on raw (pre-scaled) features

        # TEMP DEBUG (2026-07-24) — print raw live values for the
        # persistently-drifted features next to their training percentile
        # range, to see actual numbers instead of just the PSI score.
        _debug_feats = ["t5_hv_30", "t10_hv_30", "t10_hv_10", "t5_kcle_20_2"]
        _train_dist = self.metadata.get("top10_feature_distribution", {})
        for _df_name in _debug_feats:
            if _df_name in X.columns:
                _vals = X[_df_name].dropna()
                _tp = _train_dist.get(_df_name, {}).get("percentiles", [])
                self.logger.warning(
                    f"RAW LIVE {_df_name}: min={_vals.min():.2f} "
                    f"p25={_vals.quantile(0.25):.2f} median={_vals.median():.2f} "
                    f"p75={_vals.quantile(0.75):.2f} max={_vals.max():.2f} | "
                    f"TRAIN p10-p90=[{_tp[1] if len(_tp)>1 else 'NA'}, {_tp[9] if len(_tp)>9 else 'NA'}]"
                )

        X_scaled    = self._scale_features(X)

        scaled_std    = X_scaled.std()
        zero_var_cols = int((scaled_std < 1e-9).sum())
        self.logger.info(
            f"Scaled matrix: {X_scaled.shape} | "
            f"zero-var cols: {zero_var_cols} | "
            f"mean_std: {scaled_std.mean():.4f}"
        )
        if zero_var_cols > 0:
            zero_var_names = scaled_std[scaled_std < 1e-9].index.tolist()
            self.logger.warning(f"ZERO-VAR COLUMN NAMES: {zero_var_names}")
            for zv_name in zero_var_names:
                if zv_name in X.columns:
                    raw_vals = X[zv_name]
                    self.logger.warning(
                        f"  {zv_name}: raw live values -> "
                        f"min={raw_vals.min()}, max={raw_vals.max()}, "
                        f"unique={raw_vals.nunique()}, sample={raw_vals.head(5).tolist()}"
                    )
        # Flag the columns with the LARGEST post-scaling std too — these are
        # most likely driving an inflated mean_std (e.g. a 689016-style figure),
        # since one wildly out-of-distribution column dominates the average
        # far more than several merely "drifted" ones do.
        top_std = scaled_std.sort_values(ascending=False).head(5)
        self.logger.warning(f"TOP 5 HIGHEST post-scaling std columns: {top_std.to_dict()}")
        if zero_var_cols > len(self.feature_names) * 0.5:
            self.logger.warning(
                "MORE THAN HALF of features are zero-variance after scaling. "
                "Check TV_TO_MODEL_BASE builds features with the correct prefix."
            )

        predictions   = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        self.logger.info(
            f"Probability spread: min={probabilities.min():.4f}  "
            f"max={probabilities.max():.4f}  "
            f"std={probabilities.std():.6f}"
        )

        # Detect whether the probability distribution is degenerate (compressed,
        # narrow-band, bimodal, or high-probability clustered).  When it is,
        # absolute thresholds lose discriminative power and we fall back to
        # percentile-based (relative) ranking.  When the distribution is healthy
        # — which should become the common case once non-winners are drawn from
        # the same screened universe — absolute SIGNAL_THRESHOLDS are used so
        # that a stock genuinely needs to clear the calibrated bar to be called
        # STRONG BUY / BUY / HOLD.
        self._is_bimodal = self._detect_bimodal(probabilities)

        prob_series = pd.Series(probabilities, index=data_df.index)
        if self._is_bimodal:
            self.logger.info("Using percentile-based (relative) signal classification.")
            signals = self._classify_signals_relative(prob_series)
        else:
            self.logger.info("Distribution looks healthy — using absolute probability thresholds.")
            signals = prob_series.map(self._classify_signal_absolute)

        result_df = pd.DataFrame({
            "explosion_probability": probabilities,
            "prediction":            predictions,
            "signal":                signals.values,
        }, index=data_df.index)

        for col in ("symbol", "exchange"):
            if col in features_df.columns:
                result_df.insert(0, col, features_df[col].values)

        result_df = result_df.sort_values(
            "explosion_probability", ascending=False
        ).reset_index(drop=True)

        result_df["_orig_idx"] = range(len(result_df))
        X_scaled_sorted = X_scaled.reset_index(drop=True).iloc[
            result_df["_orig_idx"].values
        ].reset_index(drop=True)
        result_df = result_df.drop(columns=["_orig_idx"])

        return result_df, features_df, X_scaled_sorted

    def predict(self, data_df: pd.DataFrame) -> pd.DataFrame:
        result_df, _, _ = self._predict_internal(data_df)
        return result_df

    # -------------------------------------------------------------------------
    # Prediction with gain targets — RC4 + RC5 fixes applied here
    # -------------------------------------------------------------------------

    def predict_with_targets(
        self,
        data_df: pd.DataFrame,
        historical_gains_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        predictions, features_df, X_scaled = self._predict_internal(data_df)

        # ------------------------------------------------------------------
        # FIX 4 (original) + RC4 FIX: Regressor receives X_scaled.
        # RC4: Relaxed std guard from 1.0 → REGRESSOR_MIN_STD_THRESHOLD (0.5)
        #      and added minimum sample count guard.
        # ------------------------------------------------------------------
        if self.regressor is not None:
            try:
                # Mirror the regressor-only log_price feature added at training
                # time (ml_retrain_model.py). Built in-memory from data_df's
                # t1_close_Close / t1_open_Close — never persisted, no DB change.
                # Backward-compatible: only appended if the loaded regressor was
                # actually trained with it (older saved regressors won't have
                # 'log_price' in feature_names_in_, so they get X_scaled unchanged).
                regressor_expects_log_price = "log_price" in getattr(
                    self.regressor, "feature_names_in_", []
                )
                # Mirror the regressor-only clf_proba feature added at training
                # time (ml_retrain_model.py). This is the SAME calibrated
                # classifier probability already computed above in
                # _predict_internal() (the "explosion_probability" column in
                # `predictions`, which is row-aligned with X_scaled since both
                # were sorted/reset together) — not a leaky re-derivation, and
                # not an extra classifier call. Backward-compatible: only
                # appended if the loaded regressor was actually trained with
                # it (older saved regressors won't have 'clf_proba' in
                # feature_names_in_, so they get X_scaled unchanged).
                regressor_expects_clf_proba = "clf_proba" in getattr(
                    self.regressor, "feature_names_in_", []
                )

                extra_feats = {}
                if regressor_expects_log_price:
                    price_source = data_df.get("t1_close_Close")
                    if price_source is None:
                        price_source = pd.Series(np.nan, index=data_df.index)
                    price_fallback = data_df.get("t1_open_Close")
                    if price_fallback is not None:
                        price_source = price_source.fillna(price_fallback)
                    price_source = pd.to_numeric(price_source, errors="coerce").clip(lower=0)
                    log_price = np.log1p(price_source).reindex(X_scaled.index)
                    log_price = log_price.fillna(log_price.mean() if log_price.notna().any() else 0.0)
                    extra_feats["log_price"] = log_price
                if regressor_expects_clf_proba:
                    clf_proba = pd.Series(
                        predictions["explosion_probability"].values,
                        index=X_scaled.index,
                    )
                    extra_feats["clf_proba"] = clf_proba

                X_scaled_for_regressor = (
                    X_scaled.assign(**extra_feats) if extra_feats else X_scaled
                )

                predicted_gains_raw = self.regressor.predict(X_scaled_for_regressor)

                # RC7 FIX: If the regressor was trained on a log1p-transformed
                # target (flagged by _log_transformed_target=True), invert the
                # transform so predictions are back in % space.  Models saved
                # before RC7 lack this attribute and are treated as raw-% models
                # for backward compatibility.
                if getattr(self.regressor, "_log_transformed_target", False):
                    import numpy as _np
                    predicted_gains = _np.expm1(predicted_gains_raw)
                    self.logger.info(
                        "RC7: Applied expm1() to convert log-space predictions → % space"
                    )
                else:
                    predicted_gains = predicted_gains_raw

                gain_std  = float(predicted_gains.std())
                gain_mean = float(predicted_gains.mean())
                n_preds   = len(predicted_gains)

                self.logger.info(
                    f"Gain regressor: {n_preds} predictions  "
                    f"range=[{predicted_gains.min():.1f}%, {predicted_gains.max():.1f}%]  "
                    f"mean={gain_mean:.1f}%  std={gain_std:.2f}%"
                )

                # RC4 FIX: Use relaxed threshold and minimum sample count
                if n_preds < REGRESSOR_MIN_PRED_COUNT:
                    self.logger.warning(
                        f"RC4: Regressor made only {n_preds} predictions "
                        f"(need {REGRESSOR_MIN_PRED_COUNT}). "
                        "Falling back to calibrated gain estimates."
                    )
                    self.regressor = None
                elif gain_std < REGRESSOR_MIN_STD_THRESHOLD:
                    self.logger.warning(
                        f"RC4: Gain regressor std={gain_std:.4f}% < {REGRESSOR_MIN_STD_THRESHOLD}% "
                        f"(was 1.0 in previous version — now using relaxed threshold). "
                        f"Gain predictions are too flat to be useful. "
                        f"This likely means RC2/RC3 fixes haven't propagated yet "
                        f"(model trained before fixes). Falling back to calibrated estimates."
                    )
                    self.regressor = None
                else:
                    self.logger.info(
                        f"RC4: Regressor std={gain_std:.2f}% passes threshold "
                        f"({REGRESSOR_MIN_STD_THRESHOLD}%). Using regressor predictions."
                    )
                    predictions["target_gain_pct"]  = predicted_gains
                    predictions["target_gain_low"]   = predicted_gains * 0.8
                    predictions["target_gain_high"]  = predicted_gains * 1.2
                    predictions["gain_source"]        = "model"
            except Exception as e:
                self.logger.warning(f"Regressor predict failed ({e}) — falling back")
                self.regressor = None

        # ------------------------------------------------------------------
        # RC5 FIX: Calibrated fallback when regressor is unavailable/disabled
        # ------------------------------------------------------------------
        if self.regressor is None:
            gain_calibrator = None

            if historical_gains_df is not None and not historical_gains_df.empty:
                # RC5: Try to build an isotonic/quantile calibrator first
                gain_calibrator = _build_isotonic_gain_calibrator(historical_gains_df)

                if gain_calibrator is not None:
                    self.logger.info(
                        "RC5: Using isotonic/quantile calibration for gain estimates."
                    )
                    probs = predictions["explosion_probability"].values
                    calibrated_gains = gain_calibrator(probs)

                    predictions["target_gain_pct"]  = calibrated_gains
                    predictions["target_gain_low"]   = calibrated_gains * 0.7
                    predictions["target_gain_high"]  = calibrated_gains * 1.3
                    predictions["gain_source"]        = "isotonic_fallback"

                    gain_std_cal = float(np.std(calibrated_gains))
                    self.logger.info(
                        f"RC5: Calibrated gains range: "
                        f"{calibrated_gains.min():.1f}%–{calibrated_gains.max():.1f}%  "
                        f"std={gain_std_cal:.2f}%"
                    )

                    if gain_std_cal < 1.0:
                        self.logger.warning(
                            f"RC5: Calibrated gain std={gain_std_cal:.2f}% still low. "
                            "Historical gain data may be compressed (RC2 not yet in effect). "
                            "Falling back to rank-adjusted rule-based estimates."
                        )
                        gain_calibrator = None  # Will use rule-based below

                else:
                    # Old bucket approach as intermediate fallback
                    # (kept for compatibility but with rank-adjustment to avoid collapse)
                    self.logger.info(
                        "RC5: Isotonic calibration unavailable. "
                        "Using rank-adjusted bucket estimates."
                    )
                    gain_calibrator = None

            if gain_calibrator is None:
                # Final fallback: rule-based but with rank adjustment to ensure
                # spread across the prediction pool (avoids flat 20–30% for all)
                self.logger.info(
                    "RC5: Using rank-adjusted rule-based gain estimates."
                )
                probs = predictions["explosion_probability"].values
                base_gains = np.array([self._estimate_target_gain(p) for p in probs])

                # Apply rank-based spread: boost high-probability stocks,
                # reduce low-probability stocks relative to their rule-based estimate
                if len(probs) > 1:
                    prob_ranks = pd.Series(probs).rank(pct=True).values
                    # Spread multiplier: ranges from 0.6 (bottom) to 1.4 (top)
                    spread_multiplier = 0.6 + 0.8 * prob_ranks
                    adjusted_gains = base_gains * spread_multiplier
                else:
                    adjusted_gains = base_gains

                predictions["target_gain_pct"]  = adjusted_gains
                predictions["target_gain_low"]   = adjusted_gains * 0.5
                predictions["target_gain_high"]  = adjusted_gains * 1.5
                predictions["gain_source"]        = "rule_based_fallback"

                self.logger.info(
                    f"RC5: Rank-adjusted gains: "
                    f"{adjusted_gains.min():.1f}%–{adjusted_gains.max():.1f}%  "
                    f"std={adjusted_gains.std():.2f}%"
                )

        # Fill any remaining NaN gains
        nan_gain = predictions["target_gain_pct"].isna()
        if nan_gain.any():
            predictions.loc[nan_gain, "target_gain_pct"] = (
                predictions.loc[nan_gain, "explosion_probability"]
                .apply(self._estimate_target_gain))
            predictions.loc[nan_gain, "target_gain_low"]  = (
                predictions.loc[nan_gain, "target_gain_pct"] * 0.5)
            predictions.loc[nan_gain, "target_gain_high"] = (
                predictions.loc[nan_gain, "target_gain_pct"] * 1.5)

        # Current price lookup
        if "symbol" in predictions.columns:
            current_price = self._extract_current_price(data_df, predictions)
            if current_price is not None:
                predictions["current_price"]     = current_price
                predictions["target_price"]      = current_price * (1 + predictions["target_gain_pct"] / 100)
                predictions["target_price_low"]  = current_price * (1 + predictions["target_gain_low"] / 100)
                predictions["target_price_high"] = current_price * (1 + predictions["target_gain_high"] / 100)
            else:
                self.logger.warning(
                    "Could not determine current_price — target_price columns will be missing."
                )

        return predictions

    def _extract_current_price(
        self, data_df: pd.DataFrame, predictions: pd.DataFrame
    ) -> Optional[pd.Series]:
        if "symbol" not in data_df.columns:
            return None

        price_candidates = ["current_price", "t3_Close", "t1_close_Close", "Close"]
        for col in data_df.columns:
            for pfx in ("t3_", "t1_close_", "t1_open_"):
                if col.startswith(pfx) and col.lower().endswith("_close"):
                    if col not in price_candidates:
                        price_candidates.append(col)

        price_col = next(
            (c for c in price_candidates if c in data_df.columns), None
        )

        if price_col is None:
            for col in data_df.columns:
                for pfx in ("t3_", "t1_close_", "t1_open_"):
                    if col.startswith(pfx) and any(x in col.lower() for x in ("close", "price")):
                        price_col = col
                        break
                if price_col:
                    break

        if price_col is None:
            return None

        if price_col != "current_price":
            self.logger.info(f"Using '{price_col}' as current price source")

        price_map = data_df.set_index("symbol")[price_col]
        aligned   = predictions["symbol"].map(price_map)

        if aligned.isna().all():
            self.logger.warning(f"Price column '{price_col}' produced all NaN after symbol join")
            return None

        return aligned.values

    # -------------------------------------------------------------------------
    # Compatibility shims
    # -------------------------------------------------------------------------

    def _classify_signal(self, probability: float) -> str:
        return self._classify_signal_absolute(probability)

    def _estimate_target_gain(self, probability: float) -> float:
        """
        Rule-based gain estimate backstop.  Values are uncapped because
        STRONG BUY stocks historically produce 100%+ intraday highs.
        """
        if probability >= 0.95: return 150.0
        if probability >= 0.90:  return 80.0
        if probability >= 0.85:  return 55.0
        if probability >= 0.80:  return 40.0
        if probability >= 0.75:  return 30.0
        if probability >= 0.70:  return 23.0
        if probability >= 0.65:  return 17.0
        if probability >= 0.60:  return 13.0
        if probability >= 0.55:  return 10.0
        if probability >= 0.50:   return 7.0
        return 4.0
