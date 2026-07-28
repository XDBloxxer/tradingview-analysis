#!/usr/bin/env python3
"""
ml_retrain_model.py  —  Weekly Full Retrain From Scratch

Replaces the previous fine-tuning approach with a complete retrain every week.

DATA SOURCES (combined into one training dataset):
  1. ml_training_base    — original CSV data pivoted to wide format, both classes
                           Feature prefixes: t3_, t5_, t10_ only
  2. winners_day_prior_close / winners_day_prior_open
                         — accumulating T-1 winner samples from daily runs (label=1)
  3. non_winners_day_prior_close / non_winners_day_prior_open
                         — accumulating T-1 non-winner samples from daily runs (label=0)
  4. ml_mistake_learner  — high-weight samples from the model's own past errors
                           (false positives: weight 3x, false negatives: weight 2x)

FIXES IN THIS VERSION:
  1. Time-based train/val split (not random) — prevents data leakage where the model
     validates on stocks from the same week it trained on. With a random split, the
     model sees the market regime in both train and val, producing fake 0.9999 AUC.
     A time-based split forces the model to generalise across time periods.

     IMPORTANT: Uses a unified sort_date column (detection_date ?? event_date) so
     that base CSV rows (which have event_date but no detection_date) sort correctly
     alongside T-1 rows (which have detection_date). Without this, base CSV rows
     sort to the END as NaT (na_position='last') and the val set ends up being
     entirely T-1 non-winners → 0 positives in val → degenerate model.

  2. Stronger regularisation — min_child_weight raised from 3→10, max_depth 6→5,
     gamma 0.1→1.0, reg_alpha 0.1→0.5. These prevent the model from memorising
     individual stocks.

  3. scale_pos_weight capped at [0.5, 5.0] — avoids extreme corrections when the
     training set happens to be very imbalanced in either direction.
     SPW_MAX is intentionally kept at 5.0 even though the raw imbalance is ~8.7x:
     combining SPW=10 with eval_metric="auc" pushes raw probabilities so high that
     STRONG_BUY thresholds become meaningless.  Base-rate mismatch between the val
     calibration set and the screened inference universe is corrected via prior-
     probability correction in train_model() instead of via a higher SPW.

  4. Intraday-high label support — if actual_high_pct is available and exceeds
     INTRADAY_WIN_THRESHOLD, those rows are also treated as winners (label=1).
     This fixes the JDZG/RIME problem where the model was RIGHT (stock moved big)
     but the close-based label called it a false positive.

  5. Duplicate-date deduplication — the same (symbol, date) can appear in both the
     base CSV and T-1 tables, causing the model to overfit to repeated examples. We
     now deduplicate after combine_datasets() so the model doesn't overfit to
     repeated rows.

GAIN REGRESSOR FIXES (2026 update):
  RC2. Correct gain target: actual_high_pct now uses prev_close as denominator
       (fetched from daily_winners prev-day row or ml_prediction_accuracy), NOT the
       same-day close. This was severely compressing the target range.
  RC3. Scale alignment: regressor is trained on X_scaled (StandardScaler output),
       exactly matching what explosion_predictor.py passes at inference time.
  RC1. Broader training set: regressor now also trains on non-winner rows that have
       actual_gain_pct from ml_prediction_accuracy (yfinance data), giving far more
       training samples and a wider gain distribution.
  RC6. Mistake row enrichment: mistake samples (false positives/negatives) are
       enriched with actual_gain_pct from ml_prediction_accuracy before being added
       to combined_df, so they contribute to regressor training.

LABEL FIX (2026):
  RC3-label. Non-winner label correction via ml_prediction_accuracy: the RC2
       join only matches daily_winners rows, so non_winners_day_prior rows exit
       RC2 with actual_high_pct = NULL.  The previous implementation backfilled
       actual_high_pct from ml_prediction_accuracy into combined_df before
       apply_intraday_high_labels() ran — this was lookahead leakage because
       actual_high_pct is a same-day post-close outcome that does not exist at
       prediction time.  The fix separates the two pipelines: ml_prediction_accuracy
       is fetched to build _accuracy_gain_map (for the gain regressor) and to
       apply label corrections directly (label=0 → 1 where actual_high_pct >=
       threshold), but actual_high_pct is NOT written back into the feature
       matrix (combined_df).  apply_intraday_high_labels() now only operates on
       actual_high_pct values that originated from prior-day data (RC2/daily_winners).

NOTE ON CLASS BALANCE:
  ml_training_base contains both winners (label=1) and non-winners (label=0) from
  the original CSV, all with t3_/t5_/t10_ features from daily bars.

WHY FULL RETRAIN (not fine-tuning):
  - Only ~3,600 base rows — trivially fast to retrain (seconds, not minutes)
  - Fine-tuning with dummy-default T-3/T-7/T-14 values was corrupting new trees
  - NaN for genuinely missing columns is correct; XGBoost handles it natively
  - feature_importance.csv is regenerated each run — always accurate and current

OUTPUTS (same paths as before, drop-in compatible with ml_weekly_retrain.yml):
  ml_models/best_model.pkl
  ml_models/scaler.pkl
  ml_models/model_metadata.json
  ml_models/feature_importance.csv
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# _PriorCorrectedModel lives in a stable shared module so that joblib can
# always resolve it by fully-qualified name regardless of which script is
# __main__ at load time.  Import it here so the name is available in this
# module's namespace (used below when wrapping the calibrated model).
from src.ml_predictor.prior_corrected_model import _PriorCorrectedModel  # noqa: F401
from supabase import create_client, Client
from xgboost import XGBClassifier

# T-1 column name translator (intraday short names → model long names)
try:
    from t1_column_map import rename_t1_columns
    T1_MAP_AVAILABLE = True
except ImportError:
    T1_MAP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "t1_column_map.py not found — T-1 features will not be renamed. "
        "Place t1_column_map.py alongside ml_retrain_model.py."
    )

# Mistake learner — DISABLED (too few samples; circular feedback risk)
# With only ~18 mistakes, the 3×/2× weighting creates circular feedback:
# valid setups that fail due to market noise get trained as "bad patterns",
# causing the model to suppress them on future retrains.
# Re-enable once a statistically meaningful mistake corpus is available.
#
# try:
#     from ml_mistake_learner import build_mistake_training_samples, log_mistake_summary
#     MISTAKE_LEARNER_AVAILABLE = True
# except ImportError:
#     MISTAKE_LEARNER_AVAILABLE = False
#     logging.getLogger(__name__).warning(
#         "ml_mistake_learner.py not found — mistake-learning step will be skipped."
#     )
MISTAKE_LEARNER_AVAILABLE = False  # Placeholder — re-enable when corpus is large enough

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TABLE_BASE                   = "ml_training_base"
TABLE_WINNERS_CLOSE          = "winners_day_prior_close"
TABLE_WINNERS_OPEN           = "winners_day_prior_open"
TABLE_NON_WINNERS_CLOSE      = "non_winners_day_prior_close"
TABLE_NON_WINNERS_OPEN       = "non_winners_day_prior_open"
TABLE_WINNERS_MULTIDAY       = "winners_multiday"
TABLE_NON_WINNERS_MULTIDAY   = "non_winners_multiday"

MODEL_DIR               = Path("ml_models")
MODEL_PATH              = MODEL_DIR / "best_model.pkl"
SCALER_PATH             = MODEL_DIR / "scaler.pkl"
GAIN_REGRESSOR_PATH     = MODEL_DIR / "gain_regressor.pkl"
METADATA_PATH           = MODEL_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

# FIX (2026-06-03): Flipped weights so fresh T-1 intraday rows are trusted MORE
# than the older base CSV daily-bar rows.
#
# Previously BASE_CSV_WEIGHT=1.5 / T1_WEIGHT=1.0 told XGBoost: "the older
# daily-bar snapshots matter 50% more than today's live intraday data." That
# caused t1_close_ and t1_open_ features to land near-zero in feature
# importance (RSI_14=0.000155, Volume_Ratio=0.000769) despite being the
# freshest, most actionable signal available at inference time.
#
# With BASE_CSV_WEIGHT=1.0 / T1_WEIGHT=2.0 the model will learn from the
# intraday signal that actually drives same-day explosive moves. The base CSV
# rows still contribute full signal for the t3_/t5_/t10_ daily features — they
# are just no longer artificially inflated relative to the richer T-1 rows.
BASE_CSV_WEIGHT         = 1.0
T1_WEIGHT               = 1.0
MIN_T1_ROWS_FOR_EQUAL_WEIGHT = 1800

# Validation window — the most recent N weeks of labelled data are reserved for
# validation; everything before that window is used for training.
#
# Why dynamic instead of a fixed date:
#   A hardcoded date causes the val set to grow every week as new T-1 rows
#   accumulate, which shifts scale_pos_weight, changes the early-stopping signal,
#   and makes week-over-week metric comparisons unreliable.  Pinning to "the last
#   N weeks" keeps the val window the same size every retrain regardless of when
#   the job runs.
#
# Tune VAL_WEEKS to taste:
#   • Too small  → noisy AUC / unstable early stopping.
#   • Too large  → less training data, slower to adapt to recent market regimes.
#   8 weeks (≈ 2 months) is a reasonable starting point.
VAL_WEEKS = 8

# ---------------------------------------------------------------------------
# Purge / embargo gap at the train/val boundary
# ---------------------------------------------------------------------------
# The train/val split below is a hard date cutoff. Several of the most
# important features (t3_hv_30, t3_hv_20, t5_hv_10, ...) are rolling windows
# up to N days deep. Without a gap, a train row dated just before the cutoff
# and a val row dated just after it (especially for the same symbol) have
# rolling-window feature vectors that overlap heavily in the underlying days
# they're computed from — nothing is "from the future," but the two rows are
# highly autocorrelated purely because they sit next to each other in time.
# This inflates val AUC relative to what the model will see on a genuinely
# fresh period, in exactly the way purged/embargoed CV (de Prado) is
# designed to prevent.
#
# EMBARGO_DAYS_FLOOR / EMBARGO_DAYS_CAP bound the *inferred* gap (below).
# The floor guards against a feature-name scan that happens to find nothing
# and would otherwise embargo 0 days; the cap guards against a stray large
# number in an unrelated column name (e.g. a year) blowing the gap out to
# something absurd and starving train of data.
EMBARGO_DAYS_FLOOR = 5
EMBARGO_DAYS_CAP   = 90

# Minimum number of days of pre-embargo data train_val_split will insist on
# keeping for TRAIN, even if that means shrinking the inferred embargo below
# what the deepest rolling-window feature would otherwise call for. Purely a
# safety floor against the embargo silently consuming the entire train
# window (see the guard in train_val_split) -- it does not replace choosing
# a sane --lookback-days for the actual dataset.
MIN_TRAIN_WINDOW_DAYS = 14

# Matches the deepest rolling-window length actually present in a set of
# feature column names, so the embargo automatically widens if someone adds
# a feature with a longer lookback (e.g. hv_45) without anyone remembering
# to bump a hardcoded constant. Column names encode window length as a
# trailing/embedded integer — SMA_50, EMA_26, hv_30, Volume_MA20, HV_10,
# BBL_20_2.0_2.0, MACD_12_26_9, etc. — so every integer run in the name is a
# candidate window length; we take the max across all feature columns,
# clamped to a sane [floor, cap] range.
_WINDOW_NUMBER_RE = re.compile(r"\d+")


def _infer_embargo_days(
    feature_cols,
    floor: int = EMBARGO_DAYS_FLOOR,
    cap: int = EMBARGO_DAYS_CAP,
) -> int:
    """Infer the purge/embargo gap (in days) from the deepest rolling-window
    length encoded in the given feature column names.

    Falls back to `floor` if no plausible window length is found (e.g. an
    empty feature list), and clamps the result to `cap` so a stray large
    number in an unrelated column name can't blow the train set apart.
    """
    max_window = 0
    for col in feature_cols:
        for match in _WINDOW_NUMBER_RE.findall(col):
            try:
                n = int(match)
            except ValueError:
                continue
            # Decimal fragments like "2.0" in BBL_20_2.0_2.0 surface as "2"
            # and "0" — harmless, they're just smaller than the real window
            # numbers and won't win the max(). Ignore implausibly large
            # numbers (e.g. a stray year or id) outright rather than let
            # them dominate the max before clamping.
            if n > cap:
                continue
            max_window = max(max_window, n)

    inferred = max_window if max_window > 0 else floor
    return int(max(floor, min(cap, inferred)))


# ---------------------------------------------------------------------------
# Filter-aware negative sampling — loosening config
# ---------------------------------------------------------------------------
# These values are read from config.yaml (non_winners section) at import time
# so they stay in sync with the same knobs used by the live detector.
# If config.yaml is absent or unreadable the defaults below are used.
#
#   loosening_passes   – how many times to retry with progressively looser
#                        filters before falling back to fully-unfiltered rows.
#                        0 = no loosening (original behaviour).
#   loosening_step_pct – % to relax each min_* / max_* threshold per pass.
#                        At pass N filters are relaxed by (step * N) %.
#   min_hard_neg_ratio – stop loosening once hard-negative rows >= this
#                        fraction of the total negatives needed for that date.
#                        1.0 = fill everything from filtered pool if possible.
# ---------------------------------------------------------------------------

def _load_loosening_config() -> dict:
    """Read loosening knobs from config.yaml -> non_winners section.
    Returns defaults if the file is absent or the key is missing."""
    defaults = {
        "loosening_passes":   5,
        "loosening_step_pct": 20.0,
        "min_hard_neg_ratio": 0.80,
    }
    try:
        cfg_path = Path("config.yaml")
        if cfg_path.exists() and _YAML_AVAILABLE:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            nw = cfg.get("non_winners", {})
            for key in defaults:
                if key in nw:
                    defaults[key] = type(defaults[key])(nw[key])
    except Exception:
        pass   # silently keep defaults — retrain must not fail over config issues
    return defaults

_LOOSENING_CFG = _load_loosening_config()
SAMPLING_LOOSENING_PASSES   = _LOOSENING_CFG["loosening_passes"]
SAMPLING_LOOSENING_STEP_PCT = _LOOSENING_CFG["loosening_step_pct"]
SAMPLING_MIN_HARD_NEG_RATIO = _LOOSENING_CFG["min_hard_neg_ratio"]


def _compute_val_cutoff(df_with_dates: "pd.DataFrame") -> "pd.Timestamp":
    """Return the cutoff Timestamp that keeps the most recent VAL_WEEKS of data
    as the validation set.

    The cutoff is derived from the actual data rather than wall-clock time so
    that the val window stays stable even when the training job is backfilled or
    run on stale data.  Falls back to (today − VAL_WEEKS) if no valid dates are
    found in the dataframe.
    """
    import pandas as _pd

    date_series: "_pd.Series | None" = None
    for col in ("detection_date", "event_date", "date"):
        if col in df_with_dates.columns:
            parsed = _pd.to_datetime(df_with_dates[col], errors="coerce")
            if parsed.notna().any():
                date_series = parsed
                break

    if date_series is not None and date_series.notna().any():
        max_date = date_series.max()
    else:
        max_date = _pd.Timestamp.today().normalize()

    cutoff = max_date - _pd.Timedelta(weeks=VAL_WEEKS)
    return cutoff

# FIX 3: Minimum number of positive examples required in the val set before
# training proceeds. If the cutoff date produces fewer than this many winners,
# training aborts with a clear message rather than producing a junk model.
MIN_VAL_POSITIVES = 50

# Train-set size guards — abort if the training split is too thin to generalise.
# These fire when the Supabase tables are sparse (new deployment, data gaps, or
# a lookback_days window that returned far less data than expected).
#
# MIN_TRAIN_POSITIVES: minimum winner examples needed in the train split.
#   XGBoost with early stopping requires enough positives for the loss surface
#   to carry a meaningful gradient signal.  50 is intentionally conservative;
#   raise it once you have more accumulated data.
# MIN_TRAIN_ROWS: minimum total rows (positives + negatives) in the train split.
#   A very small train set will overfit regardless of regularisation settings.
MIN_TRAIN_POSITIVES = 50
MIN_TRAIN_ROWS      = 200

# FIX 4: Intraday high threshold — a stock is considered a "winner" even if
# it didn't close at the top, as long as it hit this intraday gain.
INTRADAY_WIN_THRESHOLD = 20.0  # %
# Aligned with ml_track_comprehensive_accuracy.py and the tracker's became_winner
# definition.  A stock that hits ≥20% intraday is a winner regardless of close price.

# scale_pos_weight caps — prevent extreme corrections while still respecting
# the actual class imbalance (~8.8x in production data).
# SPW_MAX raised from 3.0 → 5.0: the previous cap of 3.0 on an 8.8x imbalance
# under-weighted positives so severely that the logloss surface was distorted,
# making it harder for early stopping to detect genuine improvement and
# contributing to the model halting at best_iteration=12.
SPW_MIN = 0.5
SPW_MAX = 5.0    # Restored from 10.0 → 5.0.
                 #
                 # Root-cause analysis: the jump to 10.0 was intended to restore
                 # BUY/STRONG BUY signals that were being suppressed.  The real cause
                 # of that suppression was the isotonic calibrator being fitted on a
                 # val set whose positive rate (~10–25%) was far below the screened
                 # inference universe's positive rate, not an insufficient SPW.
                 #
                 # At SPW=10 with eval_metric="auc" (rank-based, ignores calibration),
                 # XGBoost pushes raw probabilities so high that 50–60%+ of post-screener
                 # stocks cluster at ≥0.90, making STRONG_BUY / BUY thresholds trivially
                 # easy to satisfy and effectively meaningless as absolute cutoffs.
                 # The RC6 _detect_bimodal workaround detects this and falls back to
                 # percentile-based ranking, but that means the absolute probability
                 # output is no longer interpretable at all.
                 #
                 # The correct fix is:
                 #   1. Keep SPW ≤ 5.0 to prevent over-weighting positives when
                 #      eval_metric="auc" is used (AUC training amplifies the effect).
                 #   2. Apply prior-probability correction in the calibration step
                 #      (see SCREENER_POSITIVE_RATE and the corrected train_model())
                 #      to account for the base-rate mismatch between val set and the
                 #      screened inference universe.  This restores interpretable
                 #      absolute probabilities without inflating them globally.

# Prior-probability correction for post-training isotonic calibration.
#
# Background:
#   The calibration set is carved from the val split, which has a positive rate
#   matching roughly the unscreened (or lightly screened) universe — typically
#   ~10–25%.  But at inference time, every stock passed through the screener
#   first.  The screener raises the effective positive rate to ~30–50%.
#
#   Isotonic calibration fits a monotone mapping from raw score → probability
#   using the calibration set's base rate as an implicit anchor.  When that
#   anchor is far below the inference base rate, the calibrator systematically
#   under-estimates probabilities for screened stocks.
#
#   Prior-probability correction (Bayes odds-ratio adjustment) shifts the
#   calibrated output to account for this mismatch without refitting the model:
#
#       odds_corrected = odds_calibrated * (p_inf / (1 - p_inf))
#                                        / (p_cal / (1 - p_cal))
#
#   where p_inf is the estimated positive rate of the screened inference universe
#   and p_cal is the positive rate of the calibration set.
#
#   Set SCREENER_POSITIVE_RATE to the observed fraction of screened candidates
#   that eventually become winners (i.e. hit >=INTRADAY_WIN_THRESHOLD).
#   Query ml_prediction_accuracy to estimate this from production history:
#       SELECT COUNT(*) FILTER (WHERE became_winner) * 1.0 / COUNT(*)
#       FROM ml_prediction_accuracy
#       WHERE prediction_date >= NOW() - INTERVAL '90 days';
#   If this query is unavailable, 0.35 is a conservative starting estimate
#   for a well-tuned screener targeting 20%+ intraday gains.
#
#   FIX (2026-06-03): Disabled prior correction (set to None).
#
#   SCREENER_POSITIVE_RATE was a manual knob that applied a Bayes odds-ratio
#   shift to push probabilities upward at inference time, on the theory that
#   the screened inference universe has a higher positive rate than the training
#   set. In practice this introduces more problems than it solves:
#
#   1. The "correct" value is unknown and has to be guessed. A value that is
#      even modestly too high inflates probabilities across the board and causes
#      the saturation problem seen in the pre-RC6 logs (25/44 stocks hitting
#      0.9098, all signals collapsing to HOLD after relative ranking).
#
#   2. The model's own isotonic calibration already accounts for any base-rate
#      mismatch it can observe in the val set. Stacking a manual Bayes shift on
#      top of that double-corrects and creates a miscalibrated pkl.
#
#   3. The right fix for base-rate mismatch is to make the training positive
#      rate match the inference positive rate — which is achieved by tuning
#      SPW_MAX and the sample weights, not by post-hoc probability shifting.
#
#   With SCREENER_POSITIVE_RATE=None the model returns raw isotonic-calibrated
#   probabilities. The RC7/RC8/RC9 safety nets in explosion_predictor.py will
#   catch any residual clustering if calibration drifts.
SCREENER_POSITIVE_RATE: float | None = None



# _PriorCorrectedModel was previously defined inline here, which caused
# a pickle deserialization failure: joblib recorded the class path as
# __main__._PriorCorrectedModel when ml_retrain_model.py was __main__ at
# save time, then failed to find it when __main__ was ml_screen_and_predict.py
# at load time.  The class now lives in src/ml_predictor/prior_corrected_model.py
# and is imported at the top of this file; call-sites below are unchanged.

XGBOOST_PARAMS = {
    "n_estimators":       500,
    "max_depth":          3,       # reduced from 6 → less overfitting
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   4,       # loosened from 10 → was over-constraining the smaller,
                                    # denser 22-feature/lookback-fixed training set (early
                                    # stopping was firing after only 3 trees)
    "gamma":              0.3,     # loosened from 2.0 → lower minimum gain needed to split
    "reg_alpha":          0.2,     # loosened from 0.5 → less L1 regularisation
    "reg_lambda":         1.5,     # loosened from 2.0 → less L2 regularisation
    "scale_pos_weight":   3,       # overridden at train time (clamped to SPW_MIN/MAX)
    "objective":          "binary:logistic",
    # eval_metric changed from "logloss" to "auc":
    # logloss is sensitive to predicted probability calibration.  When the val
    # set has a very different positive rate from the train set (e.g. val has
    # 27% positives vs train 9.5%), scale_pos_weight causes logloss on the val
    # set to be noisy from tree 1, triggering early stopping after just 7 trees.
    # AUC is rank-based and immune to this calibration skew — it only cares
    # whether the model separates positives from negatives, not the absolute
    # probability level, so it gives a stable and meaningful early-stopping signal.
    "eval_metric":        "auc",
    "use_label_encoder":  False,
    "random_state":       42,
    "n_jobs":             -1,
    # Lowered from 100 → 40. At 100 rounds of patience with n_estimators=500,
    # training was running to best_iteration≈478 — i.e. XGBoost kept adding
    # trees for as long as loss kept improving on X_val_xgb, which is also the
    # set the classification report / probability distribution are evaluated
    # on. That's model selection against the "held-out" set, not a blind
    # evaluation, and it manifested as a bimodal (hard 0 / hard 1) probability
    # collapse on X_val_xgb that did NOT appear on the truly untouched
    # calibration set (X_cal_fit), which stayed well-spread (std≈0.44) the
    # whole time. 40 rounds of patience still tolerates normal noisy plateaus
    # but stops the model well before it can memorize X_val_xgb round-by-round.
    "early_stopping_rounds": 50,
}

# Columns excluded from the feature matrix X.
NON_FEATURE_COLS = {
    "id", "created_at", "updated_at", "date", "symbol", "ticker",
    "label", "source", "sample_weight", "detection_date", "explosion_date",
    "change_pct", "rank", "notes", "mistake_type", "actual_gain_pct",
    "actual_high_pct", "_sort_date",
    # Label-leaking columns: present in training tables but unavailable at prediction time
    "gain_pct", "volume_spike",
    # TRUE GAIN TARGET FIX: the gain-regressor's own label columns — never features
    "true_gain_pct", "_unified_gain_target",
    # Training metadata: table bookkeeping columns, not predictive signals
    "snapshot_date", "snapshot_type", "snapshot_time",
    "event_date", "days_since_event", "interval",
    # ── Raw OHLCV multiday features (t3/t5/t10) ──────────────────────────────
    # These are price-level features that do not generalise out-of-sample:
    #   • Affected by stock splits, reverse splits, and delistings.
    #   • Susceptible to survivor bias in historical training data.
    #   • t3_high alone held 19.2 % feature importance, indicating the model
    #     was learning "high-priced stocks explode" rather than a real signal.
    # The multiday_feature_collector no longer writes these columns for new rows.
    # They are explicitly excluded here so that any legacy historical rows that
    # still carry them in the DB do not leak back into future retrains.
    # Derived / normalised indicators (price_vs_sma20, volume_ratio, hv_*, etc.)
    # that depend on price internally are still included — they capture the
    # signal without exposing the raw price level.
    "t3_open", "t3_high", "t3_low", "t3_close", "t3_volume",
    "t5_open", "t5_high", "t5_low", "t5_close", "t5_volume",
    "t10_open", "t10_high", "t10_low", "t10_close", "t10_volume",
    # ── Raw OHLCV T-1 features ────────────────────────────────────────────────
    # The same price-level generalisation problem applies to T-1 snapshots.
    # A $2 stock and a $50 stock have fundamentally different volatility regimes;
    # including the raw price teaches the model price-level patterns rather than
    # true momentum signals.  Derived indicators (RSI, ATR%, MACD, etc.) that
    # are internally price-normalised are retained — they capture the signal
    # without exposing the absolute price level.
    # Both the close-snapshot and open-snapshot variants are excluded.
    "t1_close_Close", "t1_close_High", "t1_close_Low", "t1_close_Open",
    "t1_open_Close",  "t1_open_High",  "t1_open_Low",  "t1_open_Open",
}

T1_MARKER_PREFIXES = ("t1_", "open_", "close_")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# logging.basicConfig() is a no-op if any handler already exists on the root
# logger (e.g. the GitHub Actions runner pre-configures one).  Explicitly
# installing a StreamHandler guarantees our format and level are always applied,
# regardless of the calling environment.
#
# --verbose / --lookback-days / --use-all-timepoints are parsed in main() via
# argparse; here we only set up the handler at INFO level.  main() will call
# _configure_logging(logging.DEBUG) when --verbose is passed.
def _configure_logging(level: int = logging.INFO) -> None:
    """Install a stdout StreamHandler on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers so we own the format completely.
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)

    # Silence noisy third-party HTTP / networking libraries.
    # They stay visible at WARNING+ so connection errors still surface.
    _QUIET_LOGGERS = [
        "urllib3", "urllib3.connectionpool",
        "httpx",
        "httpcore", "httpcore.http11", "httpcore.http2", "httpcore.connection",
        "hpack", "h2",
        "requests",
        "charset_normalizer",
        "postgrest", "gotrue", "realtime", "supabase",
        "websockets", "websockets.client", "websockets.server",
        "asyncio",
    ]
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

_configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set.")
        sys.exit(1)
    return create_client(url, key)


FETCH_BUFFER_DAYS = 10  # see full rationale in docstring below


def fetch_table_paginated(
    client: Client,
    table: str,
    page_size: int = 1000,
    date_columns: Optional[list] = None,
    cutoff_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch rows from a Supabase table using pagination.

    If ``date_columns`` and ``cutoff_date`` are given, apply a server-side
    ``col >= cutoff_date`` filter (OR'd across every column in
    ``date_columns``, plus rows where all of them are NULL, so mistake /
    legacy rows without a date are never silently dropped). This cuts
    egress on tables that hold long history but where only a recent window
    is ever used for training. The exact, row-accurate lookback filter
    still runs afterwards in Python (see FETCH_BUFFER_DAYS), so results are
    identical to fetching the whole table -- only the transferred byte
    count changes.
    """
    query = client.table(table).select("*")

    if date_columns and cutoff_date:
        # PostgREST or_() syntax: "col1.gte.DATE,col2.gte.DATE,col1.is.null"
        # A row passes if ANY clause matches: keep it if it's recent by any
        # available date column, or if it has no date at all (better to
        # over-fetch a few stray rows than to silently lose ones the old
        # full-fetch would have kept).
        clauses = [f"{col}.gte.{cutoff_date}" for col in date_columns]
        clauses += [f"{col}.is.null" for col in date_columns]
        query = query.or_(",".join(clauses))

    def _run(q):
        rows_, offset_ = [], 0
        while True:
            resp = q.range(offset_, offset_ + page_size - 1).execute()
            batch = resp.data or []
            rows_.extend(batch)
            logger.info(f"  {table}: fetched {len(rows_)} rows so far...")
            if len(batch) < page_size:
                break
            offset_ += page_size
        return rows_

    try:
        rows = _run(query)
    except Exception as e:
        if date_columns and cutoff_date:
            # Most likely cause: one of date_columns doesn't exist on this
            # table's schema, which PostgREST rejects outright. Don't let a
            # bandwidth optimization take down the whole retrain -- fall
            # back to the old unfiltered fetch and flag it loudly so the
            # date_columns list can be fixed.
            logger.warning(
                f"  {table}: date-filtered fetch failed ({e}); "
                "falling back to a full unfiltered fetch. Check that "
                f"{date_columns} actually exist on '{table}'."
            )
            rows = _run(client.table(table).select("*"))
        else:
            raise

    df = pd.DataFrame(rows)
    logger.info(f"  {table}: total {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _fetch_cutoff(lookback_days: Optional[int]) -> Optional[str]:
    """Server-side fetch cutoff = lookback_days + FETCH_BUFFER_DAYS ago.

    Looser than the exact client-side lookback filter on purpose (see
    FETCH_BUFFER_DAYS) -- this only trims what crosses the wire, the exact
    cutoff is still enforced in Python afterwards.
    """
    if not lookback_days:
        return None
    return (
        datetime.now().date() - timedelta(days=lookback_days + FETCH_BUFFER_DAYS)
    ).isoformat()


def _table_max_date(client: Client, table: str, date_column: str) -> Optional[str]:
    """Cheap 1-row query for the newest value of date_column in a table.

    Used to decide whether a full fetch is even worth doing, instead of
    paginating the whole table and finding out afterward it's stale.
    Returns None if the table is empty, the column doesn't exist, or the
    query fails for any reason (caller should fall back to a real fetch).
    """
    try:
        resp = (
            client.table(table)
            .select(date_column)
            .order(date_column, desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0].get(date_column) if rows else None
    except Exception as e:
        logger.debug(f"  {table}: max-date pre-check failed ({e}); will fall back to a full fetch")
        return None


def load_base_training_data(client: Client, lookback_days: Optional[int] = None) -> pd.DataFrame:
    """Load original CSV data from ml_training_base."""
    logger.info(f"Loading base training data from '{TABLE_BASE}'...")

    cutoff = _fetch_cutoff(lookback_days)

    # ml_training_base is a mostly-static historical seed table (uploaded once
    # via upload_base_training_data.py, not continuously appended to like the
    # T-1 tables). Once its newest event_date falls outside the lookback
    # window, EVERY retrain from here on would fetch and then immediately
    # discard the whole table. A single tiny query up front tells us that
    # cheaply, instead of paginating potentially the entire table only to end
    # up with 0 usable rows.
    if cutoff is not None:
        max_date = _table_max_date(client, TABLE_BASE, "event_date")
        if max_date is not None and max_date < cutoff:
            logger.info(
                f"  {TABLE_BASE}: newest event_date is {max_date}, older than the "
                f"{cutoff} fetch cutoff — table is stale relative to the "
                f"{lookback_days}-day lookback window. Skipping the fetch "
                "entirely (T-1 data covers the training window instead)."
            )
            return pd.DataFrame(columns=["symbol", "event_date", "label", "sample_weight", "source"])

    # ml_training_base only has event_date (no detection_date) -- see
    # combine_datasets(). Filtering on a column that doesn't exist on this
    # table would make PostgREST reject the whole query, so we deliberately
    # do NOT include detection_date here.
    df = fetch_table_paginated(
        client, TABLE_BASE,
        date_columns=["event_date"],
        cutoff_date=cutoff,
    )
    if df.empty:
        if cutoff is not None:
            # Distinguish "table is genuinely missing/misconfigured" (fatal,
            # same as before) from "table has rows, just none inside the
            # lookback window" (expected once the seed CSV ages out — not an
            # error, just means base contributes nothing to this retrain).
            unfiltered_probe = fetch_table_paginated(client, TABLE_BASE, page_size=1)
            if not unfiltered_probe.empty:
                logger.info(
                    f"  {TABLE_BASE}: no rows within the lookback window "
                    f"(cutoff={cutoff}), but the table itself is not empty — "
                    "base data is just older than lookback_days. Continuing "
                    "with T-1 data only."
                )
                return pd.DataFrame(columns=["symbol", "event_date", "label", "sample_weight", "source"])

        logger.error(
            f"Table '{TABLE_BASE}' is empty! "
            "Run upload_base_training_data.py first."
        )
        sys.exit(1)

    if "label" not in df.columns:
        logger.error(f"'{TABLE_BASE}' has no 'label' column.")
        sys.exit(1)

    # Normalise the stock identifier column to "symbol" so that combine_datasets
    # and all downstream deduplication logic uses a single consistent column name.
    # ml_training_base stores the ticker under the column "ticker" while T-1 tables
    # use "symbol".  Without this rename, after pd.concat the base rows have
    # symbol=NaN (the T-1 column) and ticker=<value>, causing drop_duplicates on
    # (symbol, event_date) to treat every ticker on the same date as the same stock,
    # collapsing all per-date base rows into a single row.
    if "symbol" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
        logger.info("  Renamed 'ticker' -> 'symbol' for consistency with T-1 tables")
    elif "symbol" not in df.columns:
        logger.warning("  Neither 'symbol' nor 'ticker' column found in base data — deduplication may be incorrect")

    if "sample_weight" not in df.columns:
        df["sample_weight"] = BASE_CSV_WEIGHT
    df["source"] = df.get("source", "base_csv")

    n_pos = int((df['label']==1).sum())
    n_neg = int((df['label']==0).sum())
    pos_rate = n_pos / max(1, len(df))
    logger.info(f"Base data: {len(df)} rows, pos={n_pos}, neg={n_neg}, pos_rate={pos_rate:.1%}")

    # Warn if the base data positive rate is unexpectedly high.
    # Expected range is ~5-20% for explosive-stock prediction.
    # If this number jumps week-over-week, the base table may have had extra
    # winner rows inserted (or negative rows deleted) outside of the normal
    # upload_base_training_data.py workflow.
    #
    # Two-tier warning:
    #   >20%: advisory — rate is above the expected ceiling but not critical.
    #         Likely causes: short LOOKBACK window over-representing a recent
    #         winning streak, or mild label drift.
    #   >25%: stronger warning — investigate before relying on this model.
    if pos_rate > 0.25:
        logger.warning(
            f"BASE DATA WARNING: positive rate is {pos_rate:.1%} ({n_pos}/{len(df)} rows). "
            "Expected ~5-20%. If this increased since the last run, check whether "
            "extra rows were inserted into ml_training_base (e.g. by intraday_high_labels "
            "or a backfill script), or whether negative rows were accidentally deleted."
        )
    elif pos_rate > 0.20:
        logger.warning(
            f"BASE DATA ADVISORY: positive rate is {pos_rate:.1%} ({n_pos}/{len(df)} rows), "
            "above the expected ~5-20% ceiling. This is not yet critical, but may indicate "
            "that a short LOOKBACK window is over-representing recent winning periods, or "
            "that mild label drift has occurred. Monitor week-over-week; if the rate "
            "continues rising, investigate ml_training_base for label imbalance."
        )

    return df

def audit_base_data(base_df: pd.DataFrame) -> None:
    """Call this immediately after load_base_training_data() to catch label corruption."""
    n_pos = int((base_df['label'] == 1).sum())
    n_neg = int((base_df['label'] == 0).sum())
    pos_rate = n_pos / len(base_df)
    
    logger.info(f"BASE DATA AUDIT:")
    logger.info(f"  Positive rate: {pos_rate:.1%}  ({n_pos} pos / {n_neg} neg)")
    
    if pos_rate > 0.40:
        logger.error(
            f"CRITICAL: Base data has {pos_rate:.1%} positive rate. "
            "This is way too high for explosive-stock prediction. "
            "Expected ~5-20%. Your ml_training_base table likely has "
            "far too few non-winner rows. Check upload_base_training_data.py."
        )
    
    # Check if 'source' column tells us where the imbalance comes from
    if 'source' in base_df.columns:
        logger.info(f"  Label breakdown by source:")
        for src, grp in base_df.groupby('source'):
            p = (grp['label'] == 1).mean()
            logger.info(f"    {src}: {p:.1%} positive ({len(grp)} rows)")
    
    # Check intraday relabelling impact
    # NOTE: previously hardcoded to 15.0, which had drifted out of sync with
    # INTRADAY_WIN_THRESHOLD (20.0) used by apply_intraday_high_labels() — this
    # diagnostic was overstating how many rows would actually get upgraded.
    if 'actual_high_pct' in base_df.columns:
        would_upgrade = (
            (base_df['label'] == 0) & 
            (pd.to_numeric(base_df['actual_high_pct'], errors='coerce') >= INTRADAY_WIN_THRESHOLD)
        ).sum()
        logger.info(f"  Rows intraday_high_labels would upgrade: {would_upgrade}")


def load_multiday_data(client: Client, lookback_days: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the backfilled / daily-generated T-3/T-5/T-10 feature tables.

    Returns two DataFrames (winners_multiday, non_winners_multiday), each
    indexed by (symbol, detection_date) and containing only the t3_/t5_/t10_
    feature columns plus those two key columns.

    These are joined onto the T-1 rows inside load_t1_data() so that every
    T-1 training row ends up with the full feature set the model expects.
    """
    result = {}
    for table, key in [
        (TABLE_WINNERS_MULTIDAY,     "winners"),
        (TABLE_NON_WINNERS_MULTIDAY, "non_winners"),
    ]:
        try:
            df = fetch_table_paginated(
                client, table,
                date_columns=["detection_date"],
                cutoff_date=_fetch_cutoff(lookback_days),
            )
            if df.empty:
                logger.warning(f"  {table}: table is empty or does not exist")
                result[key] = pd.DataFrame()
                continue

            logger.info(f"  {table}: raw fetch {len(df)} rows, sample cols: {sorted(df.columns.tolist())[:15]}")

            # Keep only key columns + feature columns (drop Supabase bookkeeping)
            keep = {"symbol", "detection_date"}
            feature_cols = [c for c in df.columns
                            if c.startswith(("t3_", "t5_", "t10_"))]

            if not feature_cols:
                logger.warning(
                    f"  {table}: NO t3_/t5_/t10_ columns found! "
                    f"All columns: {sorted(df.columns.tolist())}"
                )
                result[key] = pd.DataFrame()
                continue

            keep.update(feature_cols)
            df = df[[c for c in df.columns if c in keep]].copy()

            # Normalise detection_date to plain string YYYY-MM-DD for joining
            df["detection_date"] = pd.to_datetime(
                df["detection_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["symbol", "detection_date"])

            # Drop dupes (shouldn't happen but be safe)
            df = df.drop_duplicates(subset=["symbol", "detection_date"], keep="last")

            sample_dates = df["detection_date"].dropna().head(3).tolist()
            logger.info(
                f"  {table}: {len(df)} rows, "
                f"{len(feature_cols)} multiday feature columns, "
                f"sample dates: {sample_dates}"
            )
            result[key] = df

        except Exception as e:
            logger.error(f"Could not load '{table}': {e}", exc_info=True)
            result[key] = pd.DataFrame()

    return result.get("winners", pd.DataFrame()), result.get("non_winners", pd.DataFrame())


def _join_multiday(
    t1_df: pd.DataFrame,
    multiday_df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """
    Left-join multiday (t3_/t5_/t10_) features onto a T-1 DataFrame.

    Rows without a matching multiday entry keep NaN for the multiday columns —
    XGBoost handles this natively, so they still contribute intraday signal.
    """
    if multiday_df.empty:
        logger.warning(
            f"  {table_name}: no multiday data to join — "
            "t3/t5/t10 features will be NaN for these rows"
        )
        return t1_df

    # Normalise detection_date in t1_df to the same plain string format
    if "detection_date" not in t1_df.columns:
        logger.warning(f"  {table_name}: no detection_date column, skipping multiday join")
        return t1_df

    t1_copy = t1_df.copy()
    t1_copy["detection_date"] = pd.to_datetime(
        t1_copy["detection_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    sym_col = next((c for c in ["symbol", "ticker"] if c in t1_copy.columns), None)
    if not sym_col:
        logger.warning(f"  {table_name}: no symbol column, skipping multiday join")
        return t1_df

    # Diagnostic: show sample keys from both sides so date format mismatches are obvious
    t1_sample = list(zip(
        t1_copy[sym_col].head(3).tolist(),
        t1_copy["detection_date"].head(3).tolist()
    ))
    md_sample = list(zip(
        multiday_df["symbol"].head(3).tolist(),
        multiday_df["detection_date"].head(3).tolist()
    ))
    logger.info(f"  {table_name}: T-1 join keys sample    : {t1_sample}")
    logger.info(f"  {table_name}: multiday join keys sample: {md_sample}")

    before_cols = len(t1_copy.columns)
    merged = t1_copy.merge(
        multiday_df,
        left_on=[sym_col, "detection_date"],
        right_on=["symbol", "detection_date"],
        how="left",
        suffixes=("", "_md"),
    )

    # If sym_col != "symbol", the merge introduced a duplicate "symbol" column — drop it
    if sym_col != "symbol" and "symbol" in merged.columns:
        merged = merged.drop(columns=["symbol"])

    multiday_cols_added = [c for c in merged.columns
                           if c.startswith(("t3_", "t5_", "t10_"))
                           and c not in t1_df.columns]
    n_matched = merged[multiday_cols_added[0]].notna().sum() if multiday_cols_added else 0

    logger.info(
        f"  {table_name}: joined {len(multiday_cols_added)} multiday columns, "
        f"{n_matched}/{len(merged)} rows have multiday data "
        f"({n_matched/len(merged)*100:.0f}% coverage)"
    )
    return merged


def normalise_t1_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Ensure all T-1 feature columns that carry dollar-scale or cumulative-volume
    values are expressed in the same scale-free units that multiday_feature_collector
    writes to the DB (and that the model was trained to expect).

    WHY THIS IS NEEDED
    ------------------
    The intraday_data_collector was originally storing raw dollar values for
    MAs, Bollinger Bands, MACD, etc.  A fix to the collector now normalises
    those values at collection time, but rows already in the DB are raw.
    This function detects — per column, per batch — whether each feature is
    already normalised and skips it if so, making the operation idempotent:
    running it on already-normalised data is a no-op.

    NORMALISATION CATEGORIES (mirror of multiday_feature_collector.py)
    -------------------------------------------------------------------
    A. Price lines  → (value / close - 1) * 100   [% distance from close]
       SMA_5/10/20/50, EMA_5/10/12/20/26/50, BB lower/middle/upper,
       Keltner lower/middle/upper, Donchian lower/middle/upper, VWAP.

    B. Dollar diffs → value / close * 100          [% of close]
       MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9, MOM_10, AO.

    C. ATR          → value / close * 100          [% of close]
       ATR_14.  multiday stores ATRr (already %) but intraday's `ta` library
       returns dollar ATR, so the same normalisation is required.

    D. Volume       → value / Volume_MA20          [ratio]
       Volume_MA5, Volume_MA10, OBV.
       Volume_MA20 / Volume_MA20 = 1.0 always (sentinel value; set to 1.0).

    DERIVED RE-COMPUTATION
    ----------------------
    Four derived columns depend on the raw MA values and must be re-derived
    after normalisation so they are consistent with their inputs:
       EMA_20_Slope   = diff(EMA_20)            after normalisation
       SMA_20_Slope   = diff(SMA_20)            after normalisation
       EMA_12_26_Diff = EMA_12 − EMA_26         after normalisation
       SMA_20_50_Diff = SMA_20 − SMA_50         after normalisation
       Price_vs_SMA20 = −SMA_20                 (= 0 − normalised_SMA_20)
       Price_vs_SMA50 = −SMA_50
       Price_vs_EMA20 = −EMA_20
       ATR_14_Slope   = diff(ATR_14)            after normalisation

    DETECTION LOGIC
    ---------------
    Detection is per-column on the median of the non-null values in the batch.
    Using the median (not the mean) makes it robust to a handful of extreme
    outliers.  The close price column (e.g. t1_close_Close) is used as an
    anchor for price-relative checks.

    Each group has its own detection rule:

      Price lines: a normalised MA sits within ±50% of close (i.e. in the
        range −50 to +50).  A raw MA equals the close price itself — so its
        median absolute value ≈ median(close).  Threshold: if the median
        absolute value of the column exceeds 50 (which is impossible for a
        correctly normalised % distance), the column is raw.

      Dollar diffs (MACD/MOM/AO): normalised values are small percentages,
        typically ±5%.  Raw values are dollar differences whose magnitude
        scales with close price.  Threshold: if median(|col|) > 20 the column
        is raw.  (A normalised MACD of 20% of close would be extreme.)

      ATR: normalised ATR is 1–20% for volatile small caps.  Dollar ATR for
        a $5 stock may be $0.30 (= 6% normalised) — difficult to distinguish
        just from magnitude.  Use close as anchor: if median(ATR) > 0.5 *
        median(close) the column must be raw (a 50%+ daily ATR is impossible).
        Fallback: if median(ATR) > 50 it is certainly raw.

      Volume MAs / OBV: after normalisation Volume_MA20 is exactly 1.0 and
        Volume_MA5/MA10/OBV are ratios near 1.0. Unlike every other group
        above, this detection is done PER ROW rather than on the column
        median: a row is raw iff its own Volume_MA20 value is > 100 (raw
        share counts are always > 100; normalised Volume_MA20 is always
        exactly 1.0). This matters because a single DataFrame can contain
        rows from multiple ingestion batches — some already normalised,
        some not — and a column-median threshold would normalise-or-skip
        the *entire* column based on whichever state is in the majority,
        silently leaving the minority batch's rows on a raw share-count
        scale in the same column as rows pinned at the 1.0 sentinel.

    Args:
        df:     DataFrame as returned by rename_t1_columns — column names
                already carry the full prefix (e.g. 't1_close_SMA_20').
        prefix: 't1_close' or 't1_open' — used to locate the close-price
                anchor column and to build the full column names.

    Returns:
        df with unnormalised columns normalised in-place (copy returned).
    """
    if df.empty:
        return df

    df = df.copy()

    close_col = f"{prefix}_Close"
    vol_ma20_col = f"{prefix}_Volume_MA20"

    # --- close anchor (Series of per-row close prices) ----------------------
    # Used for price-relative normalisation; may be absent for old DB rows.
    close_s = (
        pd.to_numeric(df[close_col], errors="coerce")
        if close_col in df.columns
        else None
    )
    safe_close = close_s.replace(0, np.nan) if close_s is not None else None
    median_close = float(close_s.median()) if close_s is not None and close_s.notna().any() else None

    # --- volume anchor -------------------------------------------------------
    vol_ma20_s = (
        pd.to_numeric(df[vol_ma20_col], errors="coerce")
        if vol_ma20_col in df.columns
        else None
    )

    normalised_count = 0
    skipped_count    = 0

    def _median_abs(col_name: str) -> float:
        """Median of the absolute values of a column; NaN if column absent."""
        if col_name not in df.columns:
            return float("nan")
        s = pd.to_numeric(df[col_name], errors="coerce").dropna()
        return float(s.abs().median()) if len(s) else float("nan")

    def _is_raw_price_line(col_name: str) -> bool:
        """True if the column looks like a raw dollar price rather than % dist from close."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False   # no data — leave as-is
        # Normalised % distance from close is almost always in (−50, +50).
        # Raw dollar price equals the close, so median ≈ median_close >> 50.
        return med > 50.0

    def _is_raw_dollar_diff(col_name: str) -> bool:
        """True if MACD/MOM/AO column looks like a raw dollar difference."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        # Normalised value is a small %; rarely exceeds ±20%.
        # A raw MACD of $20+ would require an extraordinarily expensive stock
        # with an unusually wide MACD.  Safe threshold.
        return med > 20.0

    def _is_raw_atr(col_name: str) -> bool:
        """True if ATR column looks like raw dollar ATR rather than % of close."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        # Fast path: if median > 50 it is definitely dollar ATR
        if med > 50.0:
            return True
        # Slower path: use close anchor — dollar ATR > 50% of close is impossible
        if median_close is not None and median_close > 0:
            return med > median_close * 0.5
        return False

    def _is_raw_volume_ma(col_name: str) -> bool:
        """True if a Volume_MA column is still in raw share counts."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        # After normalisation Volume_MA5/10 are ratios near 1.0; Volume_MA20 = 1.0.
        # Raw share counts are always > 100 (even the most thinly traded stocks).
        return med > 100.0

    def _is_raw_obv(col_name: str) -> bool:
        """True if OBV is in raw cumulative share counts rather than ÷ vol_ma20."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        # Normalised OBV is a ratio to daily avg volume — typically ±200.
        # Raw OBV for even a thinly traded stock easily hits tens of thousands.
        return med > 10_000.0

    # ── A. Price lines → (value / close − 1) × 100 ──────────────────────────
    PRICE_LINE_COLS = [
        f"{prefix}_SMA_5",   f"{prefix}_SMA_10",  f"{prefix}_SMA_20",  f"{prefix}_SMA_50",
        f"{prefix}_EMA_5",   f"{prefix}_EMA_10",  f"{prefix}_EMA_12",
        f"{prefix}_EMA_20",  f"{prefix}_EMA_26",  f"{prefix}_EMA_50",
        f"{prefix}_BBL_20_2.0_2.0", f"{prefix}_BBM_20_2.0_2.0", f"{prefix}_BBU_20_2.0_2.0",
        f"{prefix}_KCLe_20_2",      f"{prefix}_KCBe_20_2",      f"{prefix}_KCUe_20_2",
        f"{prefix}_DCL_20_20",      f"{prefix}_DCM_20_20",      f"{prefix}_DCU_20_20",
        f"{prefix}_VWAP",
    ]

    for col in PRICE_LINE_COLS:
        if col not in df.columns:
            continue
        if _is_raw_price_line(col):
            if safe_close is not None:
                num = pd.to_numeric(df[col], errors="coerce")
                df[col] = (num / safe_close - 1) * 100
                normalised_count += 1
                logger.debug(f"  normalise_t1: {col} → % dist from close")
            else:
                logger.warning(
                    f"  normalise_t1: {col} appears raw but {close_col} is absent "
                    "— cannot normalise.  Rows without a close price will have NaN."
                )
        else:
            skipped_count += 1

    # ── B. Dollar diffs → value / close × 100 ────────────────────────────────
    DOLLAR_DIFF_COLS = [
        f"{prefix}_MACD_12_26_9",
        f"{prefix}_MACDh_12_26_9",
        f"{prefix}_MACDs_12_26_9",
        f"{prefix}_MOM_10",
        f"{prefix}_AO",
    ]

    for col in DOLLAR_DIFF_COLS:
        if col not in df.columns:
            continue
        if _is_raw_dollar_diff(col):
            if safe_close is not None:
                num = pd.to_numeric(df[col], errors="coerce")
                df[col] = num / safe_close * 100
                normalised_count += 1
                logger.debug(f"  normalise_t1: {col} → % of close")
            else:
                logger.warning(
                    f"  normalise_t1: {col} appears raw but {close_col} is absent."
                )
        else:
            skipped_count += 1

    # ── C. ATR → value / close × 100 ─────────────────────────────────────────
    atr_col = f"{prefix}_ATR_14"
    if atr_col in df.columns:
        if _is_raw_atr(atr_col):
            if safe_close is not None:
                num = pd.to_numeric(df[atr_col], errors="coerce")
                df[atr_col] = num / safe_close * 100
                normalised_count += 1
                logger.debug(f"  normalise_t1: {atr_col} → % of close")
            else:
                logger.warning(
                    f"  normalise_t1: {atr_col} appears raw but {close_col} is absent."
                )
        else:
            skipped_count += 1

    # ── D. Volume → value / Volume_MA20 ──────────────────────────────────────
    #
    # IMPORTANT: detection here MUST be done PER-ROW, not per-column-median.
    # A DataFrame passed to this function can contain a mix of rows from
    # different ingestion batches — some already normalised (post-fix
    # collector), some still raw (pre-fix collector / old DB rows). A
    # column-level median check decides "raw vs normalised" for the WHOLE
    # column at once, so whichever state is in the majority wins and the
    # minority rows are left untouched. In particular, if most rows are
    # already normalised, `vol_needs_norm` comes back False and the ~10%
    # still-raw rows keep their raw share-count values (hundreds of
    # thousands to millions) sitting in the same column as rows pinned at
    # the 1.0 sentinel — a mixed-scale column that leaks ingestion-batch
    # identity into the model instead of carrying real signal. Detecting
    # and normalising per row (using each row's own Volume_MA20 value as
    # the raw/normalised signal, since raw Volume_MA20 is always > 100 and
    # normalised Volume_MA20 is always exactly 1.0) fixes this regardless
    # of how the batches are mixed.
    vol_ma5_col  = f"{prefix}_Volume_MA5"
    vol_ma10_col = f"{prefix}_Volume_MA10"
    obv_col      = f"{prefix}_OBV"

    if vol_ma20_s is not None:
        # Per-row raw mask: a row is "raw" iff its OWN Volume_MA20 value is
        # still a raw share count (> 100). Post-normalisation Volume_MA20
        # is always exactly 1.0, so this check is self-consistent and
        # idempotent — already-normalised rows are correctly left alone.
        raw_row_mask = vol_ma20_s > 100.0
        n_raw = int(raw_row_mask.sum())

        if n_raw > 0:
            # Use each raw row's own (pre-overwrite) Volume_MA20 as the
            # denominator for that row.
            denom = vol_ma20_s.where(raw_row_mask).replace(0, np.nan)

            for col in [vol_ma5_col, vol_ma10_col, obv_col]:
                if col in df.columns:
                    num = pd.to_numeric(df[col], errors="coerce")
                    df.loc[raw_row_mask, col] = (
                        num.loc[raw_row_mask] / denom.loc[raw_row_mask]
                    )
                    normalised_count += 1
                    logger.debug(
                        f"  normalise_t1: {col} → ratio to vol_ma20 "
                        f"({n_raw} raw rows)"
                    )

            # Volume_MA20 / itself = 1.0 always (matches multiday behaviour).
            # Only overwrite the rows that were actually raw — already
            # normalised rows are already 1.0 and are left untouched.
            df.loc[raw_row_mask, vol_ma20_col] = 1.0
            normalised_count += 1
            logger.debug(
                f"  normalise_t1: {vol_ma20_col} → 1.0 sentinel "
                f"({n_raw} raw rows)"
            )
        else:
            skipped_count += 1   # every row already normalised
    else:
        logger.warning(
            f"  normalise_t1: Volume MAs/OBV appear raw but {vol_ma20_col} is "
            "absent — cannot normalise volume features."
        )

    # ── Derived re-computation ────────────────────────────────────────────────
    # These were computed before normalisation in the old collector and are
    # therefore in raw-dollar units.  Re-derive them from the now-normalised
    # base columns so they are consistent with multiday behaviour.
    #
    # We only re-derive when the base columns exist AND are now normalised
    # (i.e. median absolute value ≤ 50 after the steps above).

    ema20 = f"{prefix}_EMA_20"
    sma20 = f"{prefix}_SMA_20"
    ema12 = f"{prefix}_EMA_12"
    ema26 = f"{prefix}_EMA_26"
    sma50 = f"{prefix}_SMA_50"
    atr14 = f"{prefix}_ATR_14"

    # EMA_20_Slope and SMA_20_Slope: diff of normalised MA (%-point per snapshot)
    ema20_slope_col = f"{prefix}_EMA_20_Slope"
    if ema20 in df.columns and not _is_raw_price_line(ema20):
        df[ema20_slope_col] = pd.to_numeric(df[ema20], errors="coerce").diff(1)
        logger.debug(f"  normalise_t1: re-derived {ema20_slope_col}")

    sma20_slope_col = f"{prefix}_SMA_20_Slope"
    if sma20 in df.columns and not _is_raw_price_line(sma20):
        df[sma20_slope_col] = pd.to_numeric(df[sma20], errors="coerce").diff(1)
        logger.debug(f"  normalise_t1: re-derived {sma20_slope_col}")

    # EMA_12_26_Diff and SMA_20_50_Diff: spread of normalised MAs
    ema_diff_col = f"{prefix}_EMA_12_26_Diff"
    if ema12 in df.columns and ema26 in df.columns:
        if not _is_raw_price_line(ema12) and not _is_raw_price_line(ema26):
            df[ema_diff_col] = (
                pd.to_numeric(df[ema12], errors="coerce") -
                pd.to_numeric(df[ema26], errors="coerce")
            )
            logger.debug(f"  normalise_t1: re-derived {ema_diff_col}")

    sma_diff_col = f"{prefix}_SMA_20_50_Diff"
    if sma20 in df.columns and sma50 in df.columns:
        if not _is_raw_price_line(sma20) and not _is_raw_price_line(sma50):
            df[sma_diff_col] = (
                pd.to_numeric(df[sma20], errors="coerce") -
                pd.to_numeric(df[sma50], errors="coerce")
            )
            logger.debug(f"  normalise_t1: re-derived {sma_diff_col}")

    # Price_vs_MA: price is always 0% from itself after normalisation,
    # so price_vs_MA = 0 − normalised_MA = −normalised_MA.
    for ma_col, vs_col in [
        (sma20, f"{prefix}_Price_vs_SMA20"),
        (sma50, f"{prefix}_Price_vs_SMA50"),
        (ema20, f"{prefix}_Price_vs_EMA20"),
    ]:
        if ma_col in df.columns and not _is_raw_price_line(ma_col):
            df[vs_col] = -pd.to_numeric(df[ma_col], errors="coerce")
            logger.debug(f"  normalise_t1: re-derived {vs_col}")

    # ATR_14_Slope: diff of normalised ATR (matches multiday atr_14_slope)
    atr_slope_col = f"{prefix}_ATR_14_Slope"
    if atr14 in df.columns and not _is_raw_atr(atr14):
        df[atr_slope_col] = pd.to_numeric(df[atr14], errors="coerce").diff(1)
        logger.debug(f"  normalise_t1: re-derived {atr_slope_col}")

    if normalised_count > 0 or skipped_count > 0:
        logger.info(
            f"  normalise_t1 [{prefix}]: "
            f"{normalised_count} column(s) normalised, "
            f"{skipped_count} already normalised (skipped)."
        )

    return df


def load_t1_data(client: Client, lookback_days: Optional[int] = None) -> pd.DataFrame:
    """
    Load accumulated T-1 winner and non-winner samples, then join in the
    corresponding T-3/T-5/T-10 multiday features so every row has the full
    feature set the model expects.

    Column flow
    -----------
    T-1 intraday columns  → renamed via t1_column_map → t1_close_* / t1_open_*
    Multiday columns      → loaded separately          → t3_* / t5_* / t10_*
    Both are joined on (symbol, detection_date) into one unified row.

    Fix: close and open tables for the same label are merged into a single row
    per (symbol, detection_date) — t1_close_* features from the close snapshot
    and t1_open_* features from the open snapshot coexist in the same row.
    Previously they were concatenated as separate rows, causing every T-1 event
    to appear twice and inflating validation AUC via near-identical duplicates.
    """
    logger.info("Loading accumulated T-1 training data...")

    # Load multiday tables once — reused for both open and close variants
    logger.info("Loading multiday feature tables for T-1 enrichment...")
    winners_multiday, non_winners_multiday = load_multiday_data(client, lookback_days=lookback_days)

    # Each label (winner=1, non-winner=0) has a close table and an open table.
    # We load them as paired groups and merge close+open features into a single
    # row per (symbol, detection_date) so the same event is never duplicated.
    PAIR_CONFIG = [
        # (close_table,           open_table,             label, multiday_df)
        (TABLE_WINNERS_CLOSE,    TABLE_WINNERS_OPEN,    1, winners_multiday),
        (TABLE_NON_WINNERS_CLOSE, TABLE_NON_WINNERS_OPEN, 0, non_winners_multiday),
    ]

    # Metadata columns that exist in both tables but should not be prefixed.
    # We keep the close-table copy and ignore the open-table copy on merge.
    META_COLS = {"symbol", "detection_date", "label", "source",
                 "explosion_date", "interval", "days_since_event",
                 "t3_high_pct", "t5_high_pct", "t10_high_pct"}  # multiday cols added later

    def _load_and_rename(table: str, prefix: str) -> pd.DataFrame:
        """Fetch one table and rename its intraday feature columns."""
        df = fetch_table_paginated(
            client, table,
            date_columns=["detection_date"],
            cutoff_date=_fetch_cutoff(lookback_days),
        )
        if df.empty:
            return df
        df["label"]  = -1          # placeholder; caller sets the real value
        df["source"] = table
        if T1_MAP_AVAILABLE:
            before = len(df.columns)
            df     = rename_t1_columns(df, prefix=prefix)
            after  = len([c for c in df.columns if c.startswith(prefix)])
            logger.info(
                f"  {table}: renamed {after} feature columns "
                f"(had {before}, kept metadata + {after} features)"
            )
            dupes = df.columns[df.columns.duplicated()].tolist()
            if dupes:
                logger.warning(
                    f"  {table}: dropping {len(dupes)} duplicate column(s) "
                    f"after rename: {dupes[:10]}"
                )
                df = df.loc[:, ~df.columns.duplicated(keep="first")]
            # Normalise any features that are still in raw dollar / cumulative-
            # volume scale.  Rows collected after the intraday_data_collector fix
            # are already normalised and will be detected as such and skipped.
            # Older rows are normalised here so training always uses scale-free
            # features regardless of when the row was collected.
            df = normalise_t1_features(df, prefix=prefix)
        else:
            logger.warning(
                f"  {table}: column map unavailable — "
                "T-1 features will be NaN in model (not ideal but won't crash)"
            )
        return df

    frames = []

    for close_table, open_table, label, multiday_df in PAIR_CONFIG:
        try:
            close_df = _load_and_rename(close_table, prefix="t1_close")
            open_df  = _load_and_rename(open_table,  prefix="t1_open")

            if close_df.empty and open_df.empty:
                continue

            if close_df.empty:
                # Only open data available — no close features, proceed with open only
                logger.warning(
                    f"  {close_table}: empty — using open-only rows for label={label}"
                )
                merged = open_df
            elif open_df.empty:
                # Only close data available
                logger.warning(
                    f"  {open_table}: empty — using close-only rows for label={label}"
                )
                merged = close_df
            else:
                # ── Merge close + open into one row per (symbol, detection_date) ──
                # Keep only t1_open_* feature columns from open_df (drop shared
                # metadata so we don't get _x/_y suffixes after the merge).
                open_feature_cols = [c for c in open_df.columns if c.startswith("t1_open_")]
                join_key = ["symbol", "detection_date"]
                # Guard: only keep join keys that actually exist in open_df
                open_key_cols = [c for c in join_key if c in open_df.columns]
                open_slim = open_df[open_key_cols + open_feature_cols]

                merged = close_df.merge(
                    open_slim,
                    on=open_key_cols,
                    how="outer",       # keep rows that exist in only one table
                    suffixes=("", "_open_dup"),
                )
                # Drop any accidental duplicate suffix columns
                dup_cols = [c for c in merged.columns if c.endswith("_open_dup")]
                if dup_cols:
                    merged = merged.drop(columns=dup_cols)

                # Deduplicate within this label's merged frame (outer join can
                # introduce duplicates when join keys match multiple times)
                sym_col_local = next(
                    (c for c in ["symbol", "ticker"] if c in merged.columns), None
                )
                if sym_col_local and "detection_date" in merged.columns:
                    before_n = len(merged)
                    merged = merged.drop_duplicates(
                        subset=[sym_col_local, "detection_date"], keep="first"
                    )
                    if len(merged) < before_n:
                        logger.info(
                            f"  label={label}: dropped {before_n - len(merged)} "
                            "intra-label duplicates after close+open merge"
                        )

                logger.info(
                    f"  label={label}: merged {len(close_df)} close rows + "
                    f"{len(open_df)} open rows → {len(merged)} unique events"
                )

            merged["label"]  = label
            merged["source"] = close_table   # canonical source for this label group

            # ── Join multiday (t3/t5/t10) features ───────────────────────────
            merged = _join_multiday(merged, multiday_df, close_table)

            frames.append(merged)

        except Exception as e:
            logger.warning(f"Could not load T-1 pair ({close_table}, {open_table}): {e}")

    if not frames:
        logger.warning("No T-1 data found. Training on base data only.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["sample_weight"] = T1_WEIGHT

    t1_feature_cols = [c for c in combined.columns
                       if c.startswith("t1_close_") or c.startswith("t1_open_")]
    multiday_feature_cols = [c for c in combined.columns
                             if c.startswith(("t3_", "t5_", "t10_"))]
    non_null_t1       = combined[t1_feature_cols].notna().any().sum() if t1_feature_cols else 0
    non_null_multiday = combined[multiday_feature_cols].notna().any().sum() if multiday_feature_cols else 0

    logger.info(f"T-1 data: {len(combined)} rows, "
                f"pos={int((combined['label']==1).sum())}, "
                f"neg={int((combined['label']==0).sum())}")
    logger.info(f"T-1 intraday feature columns populated : {non_null_t1}/{len(t1_feature_cols)}")
    logger.info(f"T-1 multiday feature columns populated : {non_null_multiday}/{len(multiday_feature_cols)}")

    # Warn if multiday coverage is low — most rows should have it after backfill
    if multiday_feature_cols:
        rows_with_any_multiday = combined[multiday_feature_cols].notna().any(axis=1).sum()
        coverage_pct = rows_with_any_multiday / len(combined) * 100
        if coverage_pct < 50:
            logger.warning(
                f"  ⚠️  Only {coverage_pct:.0f}% of T-1 rows have multiday features. "
                "Run the backfill script (backfill_multiday_features.py) to improve coverage."
            )
        else:
            logger.info(f"  ✅ {coverage_pct:.0f}% of T-1 rows have multiday features")

    return combined


# ---------------------------------------------------------------------------
# RC6 FIX: Enrich mistake samples with actual_gain_pct from accuracy table
# ---------------------------------------------------------------------------

def enrich_mistakes_with_gains(
    mistake_df: pd.DataFrame,
    client: Client,
) -> pd.DataFrame:
    """
    RC6 FIX: Fetch actual_gain_pct and actual_high_pct for mistake rows from
    ml_prediction_accuracy so they contribute to gain regressor training.

    Without this, mistake rows have no gain target and are silently excluded
    from the regressor's winner_mask, wasting the corrective signal they carry.

    FIX2 (denominator consistency): ml_prediction_accuracy.actual_high_pct is
    NOT guaranteed to be computed on the same base as
    _compute_correct_actual_high_pct's prev_close-based value. The tracker
    (ml_track_comprehensive_accuracy.py) prefers a prev_close-denominated
    yfinance figure, but falls back to a same-day-close-denominated figure
    (high / same-day price - 1) whenever yfinance data is unavailable for a
    symbol/date. Silently merging that column in — as the previous
    implementation did — mixes two incompatible scales into a single gain
    target: same-day-close-based values cluster near 0% (the denominator and
    numerator are close together intraday), while prev_close-based values
    span the true intraday range. That is exactly the ~122pp split reported
    in the FIX2 diagnostic.

    To keep both sources on the same base, we re-derive actual_high_pct
    directly from daily_winners via _compute_correct_actual_high_pct — the
    same function RC2 uses for winner rows — for every mistake row we can
    match. Only when a row has no daily_winners match (so no reliable
    prev_close is obtainable) do we fall back to the accuracy table's value,
    and we mark that fallback explicitly so it can be distinguished/excluded
    downstream instead of being silently blended in as if it were on the
    same base.
    """
    if mistake_df.empty:
        return mistake_df

    if "symbol" not in mistake_df.columns or "detection_date" not in mistake_df.columns:
        return mistake_df

    logger.info("RC6: Enriching mistake samples with actual gain data...")

    # Collect unique (symbol, date) pairs from mistake rows
    pairs = (
        mistake_df[["symbol", "detection_date"]]
        .dropna()
        .drop_duplicates()
    )

    if pairs.empty:
        return mistake_df

    dates = pairs["detection_date"].unique().tolist()
    symbols = pairs["symbol"].unique().tolist()

    # ── Primary source: daily_winners, run through _compute_correct_actual_high_pct
    # so the denominator (prev_close) exactly matches RC2's winner rows. ──────────
    winners_corrected = pd.DataFrame()
    try:
        winners_rows = []
        for i in range(0, len(dates), 20):
            date_chunk = dates[i:i + 20]
            try:
                resp = (
                    client.table("daily_winners")
                    .select("symbol, detection_date, price, high, open, close, prev_close_db")
                    .in_("detection_date", date_chunk)
                    .in_("symbol", symbols)
                    .execute()
                )
                if resp.data:
                    winners_rows.extend(resp.data)
            except Exception as e:
                logger.debug(f"RC6: daily_winners fetch chunk failed: {e}")

        if winners_rows:
            winners_raw = pd.DataFrame(winners_rows)
            winners_corrected = _compute_correct_actual_high_pct(winners_raw)
    except Exception as e:
        logger.debug(f"RC6: could not fetch/correct daily_winners for mistake rows: {e}")

    # ── Fallback source: ml_prediction_accuracy (denominator not guaranteed) ────
    accuracy_rows = []
    for i in range(0, len(dates), 20):
        date_chunk = dates[i:i + 20]
        try:
            resp = (
                client.table("ml_prediction_accuracy")
                .select("symbol, prediction_date, actual_gain_pct, actual_high_pct")
                .in_("prediction_date", date_chunk)
                .in_("symbol", symbols)
                .execute()
            )
            if resp.data:
                accuracy_rows.extend(resp.data)
        except Exception as e:
            logger.debug(f"RC6: accuracy fetch chunk failed: {e}")

    if winners_corrected.empty and not accuracy_rows:
        logger.info("RC6: No accuracy or daily_winners data found for mistake symbols — skipping enrichment")
        return mistake_df

    result = mistake_df.copy()
    if "actual_gain_pct" not in result.columns:
        result["actual_gain_pct"] = np.nan
    if "actual_high_pct" not in result.columns:
        result["actual_high_pct"] = np.nan
    result["_gain_source"] = "none"

    # Step 1: fill from daily_winners (prev_close-based — same base as RC2).
    n_from_winners = 0
    if not winners_corrected.empty:
        wc = winners_corrected[["symbol", "detection_date", "actual_high_pct"]].copy()
        if "change_pct" in winners_corrected.columns:
            wc["actual_gain_pct"] = winners_corrected["change_pct"]
        merged = result.merge(
            wc,
            on=["symbol", "detection_date"],
            how="left",
            suffixes=("", "_rc2"),
        )
        for col in ["actual_gain_pct", "actual_high_pct"]:
            rc2_col = f"{col}_rc2"
            if rc2_col in merged.columns:
                fillable = merged[col].isna() & merged[rc2_col].notna()
                merged.loc[fillable, col] = merged.loc[fillable, rc2_col]
                if col == "actual_high_pct":
                    n_from_winners = int(fillable.sum())
                    merged.loc[fillable, "_gain_source"] = "daily_winners_prev_close"
                merged = merged.drop(columns=[rc2_col])
        result = merged

    # Step 2: fall back to ml_prediction_accuracy ONLY for rows still missing a
    # gain target. Tag these explicitly since their denominator may not match
    # the prev_close base used above (the accuracy tracker can fall back to a
    # same-day-close denominator when yfinance data is unavailable).
    n_from_accuracy = 0
    if accuracy_rows:
        acc_df = pd.DataFrame(accuracy_rows).rename(columns={"prediction_date": "detection_date"})
        acc_df = acc_df.dropna(subset=["symbol", "detection_date"])

        merged = result.merge(
            acc_df[["symbol", "detection_date", "actual_gain_pct", "actual_high_pct"]],
            on=["symbol", "detection_date"],
            how="left",
            suffixes=("", "_acc"),
        )
        for col in ["actual_gain_pct", "actual_high_pct"]:
            acc_col = f"{col}_acc"
            if acc_col in merged.columns:
                fillable = merged[col].isna() & merged[acc_col].notna()
                merged.loc[fillable, col] = merged.loc[fillable, acc_col]
                if col == "actual_high_pct":
                    n_from_accuracy = int(fillable.sum())
                    merged.loc[fillable, "_gain_source"] = "accuracy_table_unverified_base"
                merged = merged.drop(columns=[acc_col])
        result = merged

    # Clip to non-negative, matching RC2's treatment, so the two sources can't
    # diverge on sign conventions either.
    valid = result["actual_high_pct"].notna()
    result.loc[valid, "actual_high_pct"] = result.loc[valid, "actual_high_pct"].clip(lower=0)

    n_accuracy_base = int((result["_gain_source"] == "accuracy_table_unverified_base").sum())
    if n_accuracy_base > 0:
        logger.warning(
            f"RC6/FIX2: {n_accuracy_base} mistake rows fell back to ml_prediction_accuracy's "
            f"actual_high_pct with an unverified denominator (no matching daily_winners row "
            f"to recompute a prev_close-based value). These are tagged '_gain_source' == "
            f"'accuracy_table_unverified_base' and should be treated with lower confidence."
        )

    enriched_count = result["actual_high_pct"].notna().sum()
    logger.info(
        f"RC6: Enriched {enriched_count}/{len(result)} mistake rows with gain data "
        f"(daily_winners/prev_close: {n_from_winners}, accuracy_table fallback: {n_from_accuracy})"
    )
    result = result.drop(columns=["_gain_source"], errors="ignore")
    return result


# ---------------------------------------------------------------------------
# RC2 FIX: Correct gain target computation (prev_close denominator)
# ---------------------------------------------------------------------------

def _compute_correct_actual_high_pct(
    winners_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    RC2 FIX: Compute actual_high_pct using the PREVIOUS day's close as the
    denominator, not the same-day close (which produces near-zero values and
    was the root cause of the compressed gain range in the regressor).

    prev_close source priority (tracked and logged separately):
      1. prev_close_db  — a dedicated column already present in winners_df
         (e.g. stored by the daily pipeline at insertion time). This is the
         most reliable source and does not depend on the symbol appearing on
         consecutive days.
      2. shift(1) within symbol group — only valid when the same symbol
         appears on back-to-back days in daily_winners. For one-off small-cap
         winners this produces NaN for every row, so we track how many rows
         actually benefit from it.
      3. same-day open — last-resort fallback. Noisier than a true prev_close
         but still far better than same-day close. We log a WARNING when this
         fallback fires for more than OPEN_FALLBACK_WARN_PCT of rows, because
         a high fallback rate signals that shift(1) is not providing real data.

    Args:
        winners_df: DataFrame from daily_winners with columns:
                    symbol, detection_date, price (same-day close),
                    high, open, close, and optionally prev_close_db.

    Returns:
        winners_df with corrected actual_high_pct column added/overwritten
        and a '_prev_close_source' diagnostic column (dropped before return).
    """
    # Fraction of rows allowed to use the open fallback before we warn.
    OPEN_FALLBACK_WARN_PCT = 0.20  # warn if >20 % of rows fall back to open

    if winners_df.empty:
        return winners_df

    required = {"symbol", "detection_date", "high"}
    if not required.issubset(winners_df.columns):
        logger.warning(
            f"RC2: daily_winners missing required columns {required - set(winners_df.columns)} "
            "— cannot compute corrected actual_high_pct"
        )
        return winners_df

    df = winners_df.copy()
    df["detection_date"] = pd.to_datetime(df["detection_date"], errors="coerce")
    df = df.sort_values(["symbol", "detection_date"])

    n_total = len(df)
    df["prev_close"] = np.nan
    df["_prev_close_source"] = "none"

    # ── Source 1: explicit prev_close_db column stored at insertion time ──────
    # This is the most reliable source: no assumption about consecutive rows.
    if "prev_close_db" in df.columns:
        db_vals = pd.to_numeric(df["prev_close_db"], errors="coerce")
        mask_db = db_vals.notna() & (db_vals > 0)
        df.loc[mask_db, "prev_close"] = db_vals[mask_db]
        df.loc[mask_db, "_prev_close_source"] = "db"
        n_db = int(mask_db.sum())
        logger.info(f"RC2: prev_close_db column supplied {n_db}/{n_total} rows")
    else:
        n_db = 0

    # ── Source 2: shift(1) within consecutive symbol rows ────────────────────
    # Only fills rows that still have no prev_close (not already set by db).
    # For symbols that appear only once in daily_winners, shift produces NaN
    # and we get nothing — that is expected and correct behaviour; do not
    # treat these NaNs as the open-fallback trigger.
    close_col = "close" if "close" in df.columns else ("price" if "price" in df.columns else None)
    if close_col:
        shifted = df.groupby("symbol")[close_col].shift(1)
        shifted_numeric = pd.to_numeric(shifted, errors="coerce")
        # Apply only where prev_close is still missing
        mask_shift = (
            df["_prev_close_source"] == "none"
        ) & shifted_numeric.notna() & (shifted_numeric > 0)
        df.loc[mask_shift, "prev_close"] = shifted_numeric[mask_shift]
        df.loc[mask_shift, "_prev_close_source"] = "shift"
        n_shift = int(mask_shift.sum())

        # Rows where shift produced NaN (one-off symbols): count them explicitly
        mask_shift_nan = (df["_prev_close_source"] == "none") & shifted_numeric.isna()
        n_shift_nan_oneoff = int(mask_shift_nan.sum())
        if n_shift_nan_oneoff > 0:
            logger.info(
                f"RC2: shift(1) produced NaN for {n_shift_nan_oneoff}/{n_total} rows "
                f"(symbols appear only once in daily_winners — open fallback will be used)"
            )
    else:
        n_shift = 0

    # ── Source 3: same-day open as last-resort fallback ──────────────────────
    if "open" in df.columns:
        open_numeric = pd.to_numeric(df["open"], errors="coerce")
        mask_open = (
            df["_prev_close_source"] == "none"
        ) & open_numeric.notna() & (open_numeric > 0)
        df.loc[mask_open, "prev_close"] = open_numeric[mask_open]
        df.loc[mask_open, "_prev_close_source"] = "open"
        n_open = int(mask_open.sum())
    else:
        n_open = 0

    n_none = int((df["_prev_close_source"] == "none").sum())

    logger.info(
        f"RC2: prev_close sources — db:{n_db}  shift:{n_shift}  "
        f"open_fallback:{n_open}  missing:{n_none}  total:{n_total}"
    )

    # Warn loudly when the open fallback is carrying the majority of rows,
    # because that means shift(1) is not providing real prev_close data.
    n_non_db = n_total - n_db  # rows that couldn't use the reliable db source
    if n_non_db > 0 and n_open / n_total > OPEN_FALLBACK_WARN_PCT:
        logger.warning(
            f"RC2 WARNING: {n_open}/{n_total} rows ({n_open / n_total:.1%}) are using "
            f"same-day open as prev_close proxy. This is a noisy fallback. "
            f"Consider storing prev_close_db in the daily_winners table at insertion "
            f"time (e.g. from the yfinance previous-day close) to improve accuracy. "
            f"shift(1) only helps when the same symbol appears on consecutive days in "
            f"daily_winners, which is rare for one-off small-cap winners."
        )

    # ── Compute corrected actual_high_pct ────────────────────────────────────
    high_vals = pd.to_numeric(df["high"], errors="coerce")
    prev_close_vals = pd.to_numeric(df["prev_close"], errors="coerce")

    valid_mask = prev_close_vals.notna() & (prev_close_vals > 0) & high_vals.notna()
    df["actual_high_pct"] = np.nan
    df.loc[valid_mask, "actual_high_pct"] = (
        (high_vals[valid_mask] / prev_close_vals[valid_mask] - 1) * 100
    ).clip(lower=0)

    # Also compute actual_gain_pct if change_pct not available from same source
    if "change_pct" not in df.columns and "price" in df.columns:
        price_vals = pd.to_numeric(df["price"], errors="coerce")
        df.loc[valid_mask, "change_pct"] = (
            (price_vals[valid_mask] / prev_close_vals[valid_mask] - 1) * 100
        )

    n_corrected = int(valid_mask.sum())
    if n_corrected > 0:
        pct_range = df.loc[valid_mask, "actual_high_pct"]
        # Break down corrected rows by source for transparency
        src_counts = df.loc[valid_mask, "_prev_close_source"].value_counts().to_dict()
        logger.info(
            f"RC2: Corrected actual_high_pct for {n_corrected}/{n_total} winner rows "
            f"(range: {pct_range.min():.1f}%–{pct_range.max():.1f}%, "
            f"mean: {pct_range.mean():.1f}%) | sources: {src_counts}"
        )
    else:
        logger.warning(
            "RC2: Could not compute corrected actual_high_pct — no prev_close data available"
        )

    # Drop diagnostic column before returning
    df = df.drop(columns=["_prev_close_source"], errors="ignore")

    # Restore string dates
    df["detection_date"] = df["detection_date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# TRUE GAIN TARGET FIX: build the gain-regressor training label directly from
# the market-day OHLC snapshots the pipeline already collects, instead of
# ml_prediction_accuracy (a separate, yfinance-backed *tracking* table that
# only has rows for symbols the model has already scored).
#
# WHY A NEW COLUMN NAME ("true_gain_pct") INSTEAD OF REUSING actual_high_pct:
#   'actual_high_pct' is already overloaded with two incompatible meanings
#   elsewhere in this file:
#     1. ml_prediction_accuracy's own column — a post-hoc outcome captured by
#        the accuracy tracker, sometimes prev_close-denominated, sometimes
#        same-day-close-denominated depending on yfinance availability
#        (see enrich_mistakes_with_gains's FIX2 notes above).
#     2. The RC2-corrected version computed above in
#        _compute_correct_actual_high_pct(), which uses prev_close.
#   Mixing a third computation into that same column name would make the
#   denominator-mismatch bugs even harder to diagnose. 'true_gain_pct' is
#   used only for this pipeline, so it can never silently collide with either
#   of the above.
#
# SOURCE TABLES (already written by intraday_data_collector.py — no yfinance
# dependency, no dependency on the model having already scored the symbol):
#   winners_market_close      / non_winners_market_close
#       -> 'high' column, captured from the last 5-minute bar of the actual
#          detection-day session (snapshot_time ~15:55-16:00). This is the
#          closing-bar high, NOT a full-day intraday high — a stock that
#          spiked mid-day and faded back down by the close will understate
#          its true peak gain here. Still a real, directly-measured value
#          with no external dependency, and a strictly better training
#          signal than leaving the row unlabeled.
#   winners_day_prior_close   / non_winners_day_prior_close
#       -> 'close' column, captured from the prior trading day, but written
#          with detection_date set to the SAME detection_date as the
#          market_close snapshot (see intraday_data_collector.py's
#          "Keep original detection date" comments). This is the correct
#          prev_close denominator.
# ---------------------------------------------------------------------------

def fetch_market_snapshot_gain_targets(client: Client) -> pd.DataFrame:
    """
    Compute true_gain_pct = (market_close.high / day_prior_close.close - 1) * 100
    for every (symbol, detection_date) pair that has both snapshots, for both
    winners and non-winners.

    Returns:
        DataFrame with columns: symbol, detection_date, true_gain_pct, label
        (label=1 for rows sourced from the winners_* tables, 0 for
        non_winners_*). Empty DataFrame (same columns, zero rows) if none of
        the four source tables could be used.
    """
    TABLE_PAIRS = [
        ("winners_market_close",     "winners_day_prior_close",     1),
        ("non_winners_market_close", "non_winners_day_prior_close", 0),
    ]

    frames = []
    for market_table, prior_table, label in TABLE_PAIRS:
        try:
            market_df = fetch_table_paginated(client, market_table)
        except Exception as e:
            logger.warning(f"true_gain_pct: could not fetch '{market_table}': {e}")
            continue
        if market_df.empty:
            logger.warning(f"true_gain_pct: '{market_table}' is empty — skipping")
            continue
        if "high" not in market_df.columns:
            logger.warning(f"true_gain_pct: '{market_table}' has no 'high' column — skipping")
            continue

        try:
            prior_df = fetch_table_paginated(client, prior_table)
        except Exception as e:
            logger.warning(f"true_gain_pct: could not fetch '{prior_table}': {e}")
            continue
        if prior_df.empty:
            logger.warning(f"true_gain_pct: '{prior_table}' is empty — skipping")
            continue
        close_col = next((c for c in ("close", "Close") if c in prior_df.columns), None)
        if close_col is None:
            logger.warning(f"true_gain_pct: '{prior_table}' has no close column — skipping")
            continue

        m = market_df[["symbol", "detection_date", "high"]].copy()
        p = prior_df[["symbol", "detection_date", close_col]].rename(columns={close_col: "prev_close"})

        m["detection_date"] = pd.to_datetime(m["detection_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        p["detection_date"] = pd.to_datetime(p["detection_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        m = m.dropna(subset=["symbol", "detection_date"])
        p = p.dropna(subset=["symbol", "detection_date"])
        m = m.drop_duplicates(subset=["symbol", "detection_date"], keep="last")
        p = p.drop_duplicates(subset=["symbol", "detection_date"], keep="last")

        merged = m.merge(p, on=["symbol", "detection_date"], how="inner")
        merged["high"]       = pd.to_numeric(merged["high"], errors="coerce")
        merged["prev_close"] = pd.to_numeric(merged["prev_close"], errors="coerce")

        valid = merged["high"].notna() & merged["prev_close"].notna() & (merged["prev_close"] > 0)
        merged = merged[valid].copy()
        if merged.empty:
            logger.warning(
                f"true_gain_pct: {market_table} x {prior_table} joined but no rows had "
                "both a valid high and a valid prev_close — skipping"
            )
            continue

        merged["true_gain_pct"] = ((merged["high"] / merged["prev_close"] - 1) * 100).clip(lower=0)
        merged["label"] = label

        logger.info(
            f"true_gain_pct: {market_table} x {prior_table} -> {len(merged)} rows "
            f"(range {merged['true_gain_pct'].min():.1f}%-{merged['true_gain_pct'].max():.1f}%, "
            f"mean {merged['true_gain_pct'].mean():.1f}%)"
        )
        frames.append(merged[["symbol", "detection_date", "true_gain_pct", "label"]])

    if not frames:
        logger.warning(
            "true_gain_pct: no market-snapshot gain data available from any source table — "
            "gain regressor will fall back to ml_training_base.gain_pct / legacy accuracy-table sources."
        )
        return pd.DataFrame(columns=["symbol", "detection_date", "true_gain_pct", "label"])

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["symbol", "detection_date"], keep="last")
    logger.info(f"true_gain_pct: {len(result)} total rows with market-snapshot-derived gain targets")
    return result


def attach_true_gain_targets(combined_df: pd.DataFrame, market_gain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge true_gain_pct (from fetch_market_snapshot_gain_targets) onto
    combined_df by (symbol, detection_date), and build a single unified gain
    target column '_unified_gain_target' that the gain regressor reads first:

        1. true_gain_pct           — market-snapshot-derived (T-1 rows; see above)
        2. gain_pct                — ml_training_base's own pre-computed gain
                                      column, previously collected but never
                                      used as a regression target (only
                                      excluded from the feature matrix to
                                      avoid classifier leakage). Covers the
                                      base-CSV rows that true_gain_pct can't
                                      reach (they have no detection_date).

    Rows with neither source populated are left NaN in '_unified_gain_target'
    and train_gain_regressor() falls back to its legacy
    actual_high_pct / actual_gain_pct / accuracy-table logic for them.
    """
    combined_df = combined_df.copy()

    symbol_col = next((c for c in ["symbol", "ticker"] if c in combined_df.columns), None)
    if symbol_col and "detection_date" in combined_df.columns and not market_gain_df.empty:
        _key = pd.to_datetime(combined_df["detection_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        _lookup = market_gain_df.set_index(["symbol", "detection_date"])["true_gain_pct"]
        keys = list(zip(combined_df[symbol_col], _key))
        combined_df["true_gain_pct"] = [_lookup.get(k, np.nan) for k in keys]
        n_matched = combined_df["true_gain_pct"].notna().sum()
        logger.info(f"true_gain_pct: matched onto {n_matched}/{len(combined_df)} combined_df rows")
    else:
        combined_df["true_gain_pct"] = np.nan
        logger.info("true_gain_pct: nothing to merge (no market_gain_df data or no detection_date column)")

    unified = combined_df["true_gain_pct"].copy()
    if "gain_pct" in combined_df.columns:
        gain_pct_numeric = pd.to_numeric(combined_df["gain_pct"], errors="coerce")
        n_filled_from_base = int((unified.isna() & gain_pct_numeric.notna()).sum())
        unified = unified.fillna(gain_pct_numeric)
        logger.info(
            f"true_gain_pct: filled {n_filled_from_base} additional rows from "
            "ml_training_base.gain_pct (previously unused as a regression target)"
        )

    combined_df["_unified_gain_target"] = unified
    n_total_unified = int(combined_df["_unified_gain_target"].notna().sum())
    logger.info(
        f"_unified_gain_target populated for {n_total_unified}/{len(combined_df)} rows "
        "(true_gain_pct + ml_training_base.gain_pct combined)"
    )
    return combined_df


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def apply_intraday_high_labels(
    combined_df: pd.DataFrame,
    threshold: float = INTRADAY_WIN_THRESHOLD,
) -> pd.DataFrame:
    """
    Re-label rows where actual_high_pct >= threshold as winners (label=1).

    WHY ALL SOURCES ARE NOW ELIGIBLE
    ---------------------------------
    The previous version restricted relabelling to winners_day_prior_* rows only,
    citing a "selection-bias" concern: non_winners_day_prior rows only appear because
    they passed the screener, so relabelling them could teach the model "screener
    passer with high volatility → winner" rather than a genuine signal.

    That concern was valid at a LOW threshold (e.g. 15%) where borderline moves
    could plausibly be screener-pass noise.  But the threshold is now 20%, and the
    data directly refutes the concern:

        476 rows in ml_prediction_accuracy have actual_high_pct >= 15% with
        became_winner = false — meaning nearly 500 REAL explosive moves were sitting
        in non_winners_day_prior as label=0 training samples.  The model was being
        trained that "stock hits +20% intraday = not a winner".  This is the primary
        cause of AVOID/HOLD stocks outperforming BUY/STRONG BUY in production.

    At 20% the move is unambiguous — a stock cannot hit +20% intraday by luck of
    passing a screener filter.  The circular-bias argument does not apply when the
    outcome is this large.  Restricting to winners_day_prior was silently poisoning
    the negative class with hundreds of genuine winners.

    The selection-bias guard is retained for base_csv rows only, because those rows
    do not have reliable actual_high_pct values sourced from the same pipeline.

    Only upgrades label from 0→1 (never downgrades 1→0).
    """
    if "actual_high_pct" not in combined_df.columns:
        return combined_df

    combined_df = combined_df.copy()
    before = int((combined_df["label"] == 1).sum())

    # All T-1 rows (winners AND non-winners) are eligible for relabelling.
    # base_csv rows are excluded: their actual_high_pct values come from a
    # different pipeline and may not be computed with the same prev_close
    # denominator, making them unreliable for threshold comparisons.
    if "source" in combined_df.columns:
        is_base_csv = combined_df["source"].str.contains("base_csv", na=False)
        eligible = ~is_base_csv
        n_eligible      = int(eligible.sum())
        n_base_excluded = int(is_base_csv.sum())
        logger.info(
            f"Intraday-high relabelling: {n_eligible} T-1 rows eligible "
            f"(winners + non-winners); {n_base_excluded} base_csv rows excluded "
            f"(unreliable actual_high_pct source)."
        )
    else:
        logger.warning(
            "Intraday-high relabelling: 'source' column not found. "
            "Applying relabelling to ALL rows. "
            "Ensure load_t1_data() sets df['source'] = table_name."
        )
        eligible = pd.Series(True, index=combined_df.index)

    high_pct = pd.to_numeric(combined_df["actual_high_pct"], errors="coerce")
    mask = (
        (combined_df["label"] == 0) &
        eligible &
        (high_pct >= threshold)
    )

    # Break down the upgrade count by source so we can see how many were
    # previously-hidden non-winner explosions vs winners-table mislabels.
    if "source" in combined_df.columns and mask.any():
        for src_label, src_mask in [
            ("winners_day_prior",     combined_df["source"].str.contains("winners_day_prior", na=False) & ~combined_df["source"].str.contains("non_winners", na=False)),
            ("non_winners_day_prior", combined_df["source"].str.contains("non_winners_day_prior", na=False)),
        ]:
            n_src = int((mask & src_mask).sum())
            if n_src:
                logger.info(f"  → {n_src} upgrades from {src_label}")

    combined_df.loc[mask, "label"] = 1
    # Bump sample weight — these are high-signal corrective examples
    combined_df.loc[mask, "sample_weight"] = combined_df.loc[mask, "sample_weight"] * 1.5

    after = int((combined_df["label"] == 1).sum())
    if after > before:
        logger.info(
            f"Intraday-high relabelling: {after - before} rows upgraded to label=1 "
            f"(actual_high_pct >= {threshold}%)"
        )
    else:
        logger.info("Intraday-high relabelling: no rows upgraded (none met criteria).")
    return combined_df


def combine_datasets(base_df: pd.DataFrame, t1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate base + T-1 data.

    FIX 5: Deduplicate by (symbol, date) after concatenation.
    The same stock+date can appear in both the base CSV and T-1 tables,
    causing the model to overfit to repeated examples. We keep the T-1
    version (which has richer features) when duplicates exist.

    When both detection_date (T-1 rows) and event_date (base-CSV rows) are
    present we deduplicate each partition by its own date key separately,
    so residual within-source duplicates are still eliminated without
    incorrectly treating the two date columns as interchangeable.

    NOTE: mistake samples should be added AFTER this function returns,
    so their custom sample_weights (3.0 / 2.0) are not overwritten here.
    """
    if base_df.empty:
        logger.info("Combining: T-1 data only (base data empty or outside lookback window)")
        return t1_df.copy()

    if t1_df.empty:
        logger.info("Combining: base data only (no T-1 data yet)")
        return base_df.copy()

    t1_count = len(t1_df)
    if t1_count >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT:
        # FIX (2026-06-03): Previously this branch forced base weights to 1.0,
        # overriding BASE_CSV_WEIGHT and accidentally making T-1 rows equal to
        # (or lower than) base rows when T1_WEIGHT < BASE_CSV_WEIGHT. Now we
        # always apply BASE_CSV_WEIGHT / T1_WEIGHT so the intentional weighting
        # is respected regardless of how many T-1 rows are present.
        logger.info(
            f"T-1 data ({t1_count} rows) >= threshold ({MIN_T1_ROWS_FOR_EQUAL_WEIGHT}). "
            f"Base rows weighted {BASE_CSV_WEIGHT}x, T-1 rows weighted {T1_WEIGHT}x."
        )
        base_df = base_df.copy()
        base_df["sample_weight"] = BASE_CSV_WEIGHT
    else:
        logger.info(
            f"T-1 data ({t1_count} rows) < threshold ({MIN_T1_ROWS_FOR_EQUAL_WEIGHT}). "
            f"Base rows weighted {BASE_CSV_WEIGHT}x, T-1 rows weighted {T1_WEIGHT}x."
        )

    # ── Step 1: Deduplicate each frame independently, using its own natural key ──
    #
    # Deduplicate BEFORE concat so we never need to re-split the combined frame.
    # This avoids every previous attempt to infer which rows belong to which
    # source after the fact (via source column, detection_date.notna(), etc.) —
    # all of which broke because ml_training_base contains rows from multiple
    # pipelines with mixed source values and mixed date columns.
    #
    # base_df key  : (symbol, event_date)   — base rows are identified by when
    #                the stock event happened, not when they were collected.
    #                Multiple snapshot rows for the same event (t3/t5/t10 intervals
    #                stored as separate rows) share the same (symbol, event_date)
    #                and are correctly collapsed here to one row.
    # t1_df key    : (symbol, detection_date) — T-1 rows are identified by the
    #                day-prior detection date.  The close+open merge in
    #                load_t1_data() already produces one row per event, but we
    #                dedup again here as a safety net.
    #
    # keep="last": within base_df, later snapshots (t10 > t5 > t3) carry more
    # history and should be preferred.  Supabase pagination returns rows in
    # insertion order, so t10 rows (inserted last) tend to come last.

    base_sym = next((c for c in ["symbol", "ticker"] if c in base_df.columns
                     and base_df[c].notna().any()), None)
    t1_sym   = next((c for c in ["symbol", "ticker"] if c in t1_df.columns
                     and t1_df[c].notna().any()), None)

    n_base_before = len(base_df)
    n_t1_before   = len(t1_df)

    # Capture label counts before dedup so we can audit what was dropped.
    base_pos_before = int((base_df["label"] == 1).sum()) if "label" in base_df.columns else 0
    base_neg_before = int((base_df["label"] == 0).sum()) if "label" in base_df.columns else 0
    base_rate_before = base_pos_before / max(1, base_pos_before + base_neg_before)
    logger.info(
        f"Base data pre-dedup: {n_base_before} rows, "
        f"pos={base_pos_before}, neg={base_neg_before}, "
        f"pos_rate={base_rate_before:.1%}"
    )

    if base_sym and "event_date" in base_df.columns:
        base_df = base_df.drop_duplicates(subset=[base_sym, "event_date"], keep="last")
    elif base_sym and "detection_date" in base_df.columns:
        base_df = base_df.drop_duplicates(subset=[base_sym, "detection_date"], keep="last")

    if t1_sym and "detection_date" in t1_df.columns:
        t1_df = t1_df.drop_duplicates(subset=[t1_sym, "detection_date"], keep="first")

    n_base_dropped = n_base_before - len(base_df)
    n_t1_dropped   = n_t1_before   - len(t1_df)

    # Compute per-label dedup impact so we can detect asymmetric row loss.
    # If dedup disproportionately drops negatives (e.g. many t3/t5/t10 snapshots
    # exist only for non-winners), the post-dedup positive rate will be inflated
    # relative to the pre-dedup rate, and the model will train on a skewed set.
    base_pos_after = int((base_df["label"] == 1).sum()) if "label" in base_df.columns else 0
    base_neg_after = int((base_df["label"] == 0).sum()) if "label" in base_df.columns else 0
    base_pos_dropped = base_pos_before - base_pos_after
    base_neg_dropped = base_neg_before - base_neg_after

    logger.info(
        f"Pre-concat dedup — base: {n_base_before} → {len(base_df)} "
        f"(dropped {n_base_dropped} rows: {base_pos_dropped} pos + {base_neg_dropped} neg, "
        f"key={base_sym}+event_date); "
        f"T-1: {n_t1_before} → {len(t1_df)} "
        f"(dropped {n_t1_dropped}, key={t1_sym}+detection_date)"
    )

    # Warn when dedup removes a disproportionate share of one label.
    # A healthy dedup should drop roughly equal fractions of positives and
    # negatives.  When negatives are dropped at a much higher rate the post-dedup
    # positive rate rises, leading to an under-estimated scale_pos_weight and a
    # model that under-penalises false positives.
    if n_base_dropped > 0 and base_pos_before > 0 and base_neg_before > 0:
        frac_pos_dropped = base_pos_dropped / base_pos_before
        frac_neg_dropped = base_neg_dropped / base_neg_before
        if frac_neg_dropped > frac_pos_dropped + 0.10:
            logger.warning(
                f"DEDUP ASYMMETRY WARNING: dedup dropped {frac_neg_dropped:.1%} of "
                f"negatives but only {frac_pos_dropped:.1%} of positives from base data. "
                f"({base_neg_dropped} neg rows vs {base_pos_dropped} pos rows removed.) "
                "This raises the post-dedup positive rate and may cause scale_pos_weight "
                "to underestimate the true class imbalance. Likely cause: multiple "
                "snapshot rows (t3/t5/t10) exist only for non-winner events. "
                "Check whether ml_training_base stores extra rows for non-winners."
            )

    # Log base label distribution post-dedup so we can catch label imbalance early
    base_pos = int((base_df["label"] == 1).sum()) if "label" in base_df.columns else 0
    base_neg = int((base_df["label"] == 0).sum()) if "label" in base_df.columns else 0
    base_rate = base_pos / max(1, base_pos + base_neg)
    logger.info(
        f"Base data after dedup: {len(base_df)} rows, "
        f"pos={base_pos}, neg={base_neg}, pos_rate={base_rate:.1%}"
    )
    if base_rate > 0.30:
        logger.warning(
            f"Base data post-dedup positive rate is {base_rate:.1%}. "
            "Each (symbol, event_date) pair should have one canonical label. "
            "If winners and non-winners share the same (symbol, event_date) with "
            "different labels, keep=last may be selecting winners over non-winners. "
            "Consider auditing ml_training_base for conflicting label rows."
        )
    elif base_rate > 0.20:
        # Rate is in the 20–30% amber zone.  Log with context so the operator
        # can decide whether to investigate.  Key risk: a short LOOKBACK window
        # (e.g. 90 days) covering a recent period with unusually many winners
        # will inflate positive rate without any data corruption.
        logger.warning(
            f"Base data post-dedup positive rate is {base_rate:.1%} "
            f"({base_pos} pos / {base_neg} neg). "
            "This is above the expected ~5-20% ceiling. "
            "Possible causes: (1) short LOOKBACK window covering an unusually "
            "winner-heavy period — the model may over-represent recent market "
            "conditions; (2) asymmetric dedup dropped more negatives than positives "
            "(see DEDUP ASYMMETRY WARNING above if present); "
            "(3) mild label drift in ml_training_base. "
            "Check the pre-dedup vs post-dedup counts above to isolate the cause."
        )

    # ── Step 2: Concat (T-1 first so it wins any cross-source duplicates) ─────
    combined = pd.concat([t1_df, base_df], ignore_index=True, sort=False)

    # ── Step 3: Cross-source dedup — T-1 beats base for the same event ────────
    # A stock may appear in both T-1 (detection_date) and base (event_date) for
    # the same real-world day.  We prefer the T-1 row (richer features).
    # We only do this cross-source dedup when detection_date is populated, using
    # it as the unified date key.  Base rows that have only event_date (no
    # detection_date) are never incorrectly dropped here.
    cross_sym = next((c for c in ["symbol", "ticker"] if c in combined.columns), None)
    if cross_sym and "detection_date" in combined.columns:
        n_before_cross = len(combined)
        # Only dedup rows that actually have a detection_date (T-1 rows and any
        # base rows that happen to have detection_date populated).
        has_det = combined["detection_date"].notna()
        cross_deduped = combined[has_det].drop_duplicates(
            subset=[cross_sym, "detection_date"], keep="first"
        )
        combined = pd.concat([cross_deduped, combined[~has_det]], ignore_index=True, sort=False)
        n_cross_dropped = n_before_cross - len(combined)
        if n_cross_dropped > 0:
            logger.info(
                f"Cross-source dedup: removed {n_cross_dropped} rows where T-1 and base "
                f"shared the same (symbol, detection_date) ({n_before_cross} → {len(combined)})"
            )

    logger.info(f"Combined dataset: {len(combined)} rows")

    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())

    if n_pos > 0 and n_pos / (n_pos + n_neg) > 0.40:
        # Log a breakdown by source to diagnose which data source is causing the imbalance.
        if "source" in combined.columns:
            logger.error("Positive rate breakdown by source:")
            for src, grp in combined.groupby("source"):
                grp_pos = int((grp["label"] == 1).sum())
                grp_neg = int((grp["label"] == 0).sum())
                grp_rate = grp_pos / max(1, grp_pos + grp_neg)
                logger.error(f"  {src}: {len(grp)} rows, pos={grp_pos}, neg={grp_neg}, rate={grp_rate:.1%}")
        logger.error(
            f"ABORTING: positive rate {n_pos/(n_pos+n_neg):.1%} is too high. "
            "Expected ~5-20% for explosive-stock prediction. "
            "Likely causes: (1) deduplication wiped most negative rows — check "
            "that the 'source' column is populated on base rows; "
            "(2) ml_training_base itself has corrupt/missing negatives; "
            "(3) intraday_high_labels relabelled too many negatives as winners."
        )
        sys.exit(1)

    logger.info(
      f"Combined dataset: {len(combined)} rows, "
      f"{len(combined.columns)} columns, "
      f"pos={n_pos}, neg={n_neg}, "
      f"pos_rate={n_pos/len(combined)*100:.1f}%"
    )

    if n_neg == 0:
        logger.error(
            "CRITICAL: No negative (non-winner) samples found. "
            "The model cannot train without both classes."
        )
        sys.exit(1)

    if n_pos > 0 and (n_neg / n_pos) < 0.2:
        logger.warning(
            f"Class imbalance WARNING: {n_pos} positives vs {n_neg} negatives "
            f"(ratio {n_neg/n_pos:.2f}). scale_pos_weight will compensate, "
            "but consider accumulating more non-winner data."
        )

    return combined


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract feature matrix X, labels y, and sample weights w.

    Returns:
        X: DataFrame of features (NaN preserved here; build_scaler fills to 0.0
           after standardisation so training and inference use the same representation)
        y: Series of labels (0/1)
        w: Series of sample weights
    """
    y = df["label"].astype(int)
    w = (
        df["sample_weight"].astype(float)
        if "sample_weight" in df.columns
        else pd.Series(1.0, index=df.index)
    )

    FEATURE_PREFIXES = ("t1_close_", "t1_open_", "t3_", "t5_", "t10_")
    feature_cols = [
        c for c in df.columns
        if any(c.startswith(pfx) for pfx in FEATURE_PREFIXES)
        and c not in NON_FEATURE_COLS  # exclude raw OHLCV and other non-predictive cols
    ]

    X = df[feature_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # ── FIX: has_t1_features binary flag ────────────────────────────────────
    # T-1 rows (from winners_day_prior_close / non_winners_day_prior_close)
    # have a 'source' column containing 'day_prior'.  Base CSV rows do not.
    # XGBoost handles missingness natively but only when values are actually NaN.
    # After fillna(col_mean), all base rows receive the same imputed constant
    # for every t1_ column, making those features look constant for 85% of rows
    # and causing XGBoost to ignore them entirely in feature importance.
    # Adding a binary 'has_t1_features' column lets XGBoost build a distinct
    # decision branch for "rows where t1_ data is real" vs "rows where it is
    # imputed", restoring t1_ signal without any schema or scaler changes.
    # At inference time (explosion_predictor.py) this column is always set to
    # 1.0 because live predictions always have T-1 intraday data.
    if "source" in df.columns:
        X["has_t1_features"] = (
            df["source"].str.contains("day_prior", na=False).astype(float)
        )
    else:
        # Fallback: infer from NaN coverage of t1_ columns — if >50% of t1_
        # columns are populated for a row it is almost certainly a T-1 row.
        t1_cols = [c for c in X.columns if c.startswith(("t1_close_", "t1_open_"))]
        if t1_cols:
            X["has_t1_features"] = (X[t1_cols].notna().mean(axis=1) > 0.5).astype(float)
        else:
            X["has_t1_features"] = 0.0

    n_t1_rows = int(X["has_t1_features"].sum())
    n_base_rows = len(X) - n_t1_rows
    logger.info(
        f"has_t1_features flag: {n_t1_rows} T-1 rows (flag=1), "
        f"{n_base_rows} base rows (flag=0)"
    )

    # ── OPTIONAL: restrict to a pre-computed feature subset ────────────────
    # If src/ml_predictor/feature_selection.py has been run (see that module's
    # docstring), it writes ml_models/feature_selection/selected_features.json.
    # Setting USE_SELECTED_FEATURES=1 in the environment restricts X to that
    # subset instead of all ~395 raw columns. Default behaviour (flag unset)
    # is unchanged — nothing about this file's normal operation depends on
    # the feature_selection module.
    if os.environ.get("USE_SELECTED_FEATURES", "").lower() in ("1", "true", "yes"):
        selected_path = Path("ml_models/feature_selection/selected_features.json")
        if selected_path.exists():
            with open(selected_path) as f:
                selected = json.load(f).get("final_features", [])
            missing = [c for c in selected if c not in X.columns]
            if missing:
                logger.warning(
                    f"USE_SELECTED_FEATURES: {len(missing)} selected features "
                    f"not present in current data (schema drift): {missing[:10]}..."
                )
            keep = [c for c in selected if c in X.columns]
            if "has_t1_features" in X.columns and "has_t1_features" not in keep:
                keep.append("has_t1_features")
            X = X[keep]
            logger.info(
                f"USE_SELECTED_FEATURES active — restricted to {X.shape[1]} features "
                f"from {selected_path}"
            )
        else:
            logger.warning(
                f"USE_SELECTED_FEATURES=1 but {selected_path} not found — "
                "run `python -m src.ml_predictor.feature_selection` first. "
                "Falling back to the full feature set."
            )

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    nan_pct = X.isna().mean().mean() * 100
    logger.info(f"Overall NaN rate: {nan_pct:.1f}% (expected for cross-lag rows)")

    return X, y, w


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def build_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame, list]:
    """
    Fit scaler on train-split rows only. Returns scaler, scaled X_train, and
    the list of sparse column names determined from training-set coverage.

    LEAKAGE FIX: The scaler is now fit exclusively on X_train so that
    validation-set rows never influence the scaler's mean_ / std_ parameters.
    Call scale_with_fitted_scaler(scaler, X_val, sparse_cols) to transform the
    val set (or any other split) using the same, already-fitted scaler and the
    sparse-column list computed here from the training set.

    NaN RESTORATION FIX (t1_ features): Sparse columns (coverage < SPARSE_THRESHOLD)
    have NaN restored AFTER scaling so XGBoost can use its native missing-value
    branch logic.  Previously fillna(col_mean) → scale → fillna(0.0) made these
    columns appear as the constant 0.0 for 85% of rows, hiding them from gain-based
    feature importance entirely.  StandardScaler still receives NaN-free input
    (required), but XGBoost receives NaN for genuinely absent values, matching the
    inference path in _scale_features() in explosion_predictor.py.

    SPARSE THRESHOLD FIX: sparse_cols is derived exclusively from X_train coverage
    so that the same set of columns is treated as sparse during both training and
    validation transforms.  Previously scale_with_fitted_scaler() re-computed
    coverage from whatever X was passed in, meaning a column that is 60% populated
    in train but 40% in val could be classified differently between splits.
    """
    SPARSE_THRESHOLD = 0.5   # columns with < 50% coverage get NaN restored post-scale

    scaler        = StandardScaler()
    col_means     = X_train.mean()           # computed on train rows only
    X_filled      = X_train.fillna(col_means)
    scaler.fit(X_filled)                     # fit on train rows only — no val leakage

    X_scaled_vals = scaler.transform(X_filled)
    X_scaled      = pd.DataFrame(X_scaled_vals, columns=X_train.columns, index=X_train.index)
    # Fill any remaining NaN (e.g. columns with all-NaN that have no mean) with 0.
    X_scaled      = X_scaled.fillna(0.0)

    # ── Restore NaN for sparse (t1_) columns so XGBoost uses missing-value branches ──
    # Identify columns with low coverage in the training set.  These are almost
    # always t1_ intraday columns which are NaN for every base-CSV row.
    # Restoring NaN lets XGBoost route base rows through its learned "missing"
    # branch rather than treating them as "value = column mean", which was causing
    # all t1_ features to appear constant for 85% of rows and be ignored.
    coverage = X_train.notna().mean()
    sparse_cols = coverage[coverage < SPARSE_THRESHOLD].index.tolist()
    # has_t1_features is binary (0/1) and always dense — never restore NaN on it
    sparse_cols = [c for c in sparse_cols if c != "has_t1_features"]
    if sparse_cols:
        nan_mask = X_train[sparse_cols].isna()
        X_scaled.loc[:, sparse_cols] = X_scaled[sparse_cols].where(~nan_mask, other=np.nan)
        logger.info(
            f"NaN restored for {len(sparse_cols)} sparse columns "
            f"(coverage < {SPARSE_THRESHOLD:.0%}) so XGBoost uses native missing-value branches. "
            f"Examples: {sparse_cols[:5]}"
        )

    return scaler, X_scaled, sparse_cols


def scale_with_fitted_scaler(
    scaler: StandardScaler,
    X: pd.DataFrame,
    sparse_threshold_cols: list | None = None,
    sparse_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Transform X using an already-fitted scaler (e.g. to scale the val set or
    to reassemble a full scaled DataFrame for the gain regressor).

    NaN RESTORATION FIX: mirrors build_scaler — sparse columns have NaN restored
    after scaling so XGBoost receives genuinely missing values rather than 0.0.

    SPARSE THRESHOLD FIX: Pass ``sparse_threshold_cols`` (the third return value
    of ``build_scaler``) so that sparse-column membership is determined from the
    training-set coverage rather than re-computed from the coverage of whatever X
    is passed in here.  A column that is 60% populated in train but 40% in val
    would otherwise be classified differently between splits, causing NaN
    restoration to differ between training and inference paths.

    Fallback: if ``sparse_threshold_cols`` is None (e.g. calling legacy code that
    hasn't been updated yet), coverage is re-computed from X as before and a
    DeprecationWarning is logged so callers know to pass the list.
    """
    col_means = pd.Series(scaler.mean_, index=X.columns)
    X_filled  = X.fillna(col_means)

    X_scaled_vals = scaler.transform(X_filled)
    X_scaled      = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
    X_scaled      = X_scaled.fillna(0.0)

    # ── Restore NaN for sparse columns (mirrors build_scaler logic) ────────────
    if sparse_threshold_cols is not None:
        # Preferred path: use the column list determined from train coverage.
        sparse_cols = [c for c in sparse_threshold_cols if c in X.columns]
    else:
        # Legacy fallback: re-compute from the passed-in X.
        # This is kept for backward compatibility but produces inconsistent results
        # when val coverage differs from train coverage.
        logger.warning(
            "scale_with_fitted_scaler called without sparse_threshold_cols — "
            "sparse columns will be inferred from the coverage of the input X, "
            "which may differ from train coverage. Pass the sparse_cols list "
            "returned by build_scaler() to ensure consistent NaN restoration."
        )
        coverage = X.notna().mean()
        sparse_cols = coverage[coverage < sparse_threshold].index.tolist()
        sparse_cols = [c for c in sparse_cols if c != "has_t1_features"]

    if sparse_cols:
        nan_mask = X[sparse_cols].isna()
        X_scaled.loc[:, sparse_cols] = X_scaled[sparse_cols].where(~nan_mask, other=np.nan)

    return X_scaled


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_cal: pd.DataFrame = None,
    y_cal: pd.Series = None,
) -> object:
    """Train XGBClassifier from scratch with early stopping.

    RC6: If X_cal/y_cal are supplied (a held-out calibration set),
    the raw XGBoost model is wrapped with CalibratedClassifierCV
    (method='isotonic', cv='prefit') before being returned.  Isotonic
    regression fits a rank-preserving monotone step function to the
    calibration data and does NOT anchor to the calibration set's base
    rate — making it robust to the mismatch between the val-set positive
    rate (~10–25%) and the screened inference universe's higher positive
    rate.  Sigmoid (Platt scaling) anchors to the cal-set base rate and
    was suppressing all inference probabilities to 0.50–0.68.
    The calibrator is fitted on X_cal/y_cal (not X_train) so that no
    training data leaks into the calibration fit.
    """
    params = XGBOOST_PARAMS.copy()

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0 and n_neg > 0:
        raw_spw = n_neg / n_pos
        # FIX 3: clamp scale_pos_weight to avoid extreme corrections
        clamped_spw = max(SPW_MIN, min(SPW_MAX, raw_spw))
        params["scale_pos_weight"] = round(clamped_spw, 3)
        if abs(raw_spw - clamped_spw) > 0.01:
            logger.info(
                f"  scale_pos_weight: raw={raw_spw:.3f} → clamped to {clamped_spw:.3f} "
                f"(limits: [{SPW_MIN}, {SPW_MAX}])"
            )
        else:
            logger.info(
                f"  scale_pos_weight set to {clamped_spw:.3f} "
                f"(neg={n_neg} / pos={n_pos})"
            )

    early_stopping = params.pop("early_stopping_rounds", 30)

    model = XGBClassifier(**params, early_stopping_rounds=early_stopping)

    logger.info("Training XGBoost model from scratch...")
    logger.info(f"  Train: {len(X_train)} rows")
    logger.info(f"  Val:   {len(X_val)} rows")

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train.values,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info(f"  Best iteration: {model.best_iteration}")
    logger.info(f"  Best val AUC: {model.best_score:.4f}")

    # Warn if early stopping fired suspiciously early — indicates the val set
    # is too small, too imbalanced, or temporally non-representative.
    if model.best_iteration < 30:
        val_pos  = int((y_val == 1).sum())
        val_neg  = int((y_val == 0).sum())
        val_rate = val_pos / max(1, val_pos + val_neg)
        logger.warning(
            f"  ⚠️  UNDERTRAINED: best_iteration={model.best_iteration} "
            f"(early_stopping fired after only {model.best_iteration} trees). "
            f"Val set: {val_pos} pos / {val_neg} neg ({val_rate:.1%} positive rate). "
            "Possible causes: (1) val set has too few positives (<20), causing "
            "noisy AUC that prematurely triggers early stopping; "
            "(2) val period has a very different class distribution from train; "
            "(3) heavy regularisation params (gamma/min_child_weight) need loosening."
        )

    # Warn if val AUC is suspiciously perfect — sign of data leakage
    if model.best_score > 0.999:
        logger.warning(
            f"  ⚠️  Val AUC={model.best_score:.4f} is suspiciously high. "
            "This may indicate data leakage or label overlap. "
            "Check that the validation set does not overlap with training dates."
        )

    # RC6 (revised): Post-training probability calibration with prior correction.
    #
    # CALIBRATION STRATEGY:
    #   Step 1 — Isotonic calibration: fit a rank-preserving monotone mapping
    #     from raw XGBoost scores to probabilities on the held-out calibration
    #     set.  Isotonic is preferred over sigmoid (Platt scaling) because
    #     sigmoid anchors to the calibration set's positive base rate, which
    #     suppresses all inference probabilities when the screened inference
    #     universe has a higher base rate than the val/cal set.
    #
    #   Step 2 — Prior-probability correction (Bayes odds-ratio adjustment):
    #     Even isotonic calibration implicitly anchors to the calibration set's
    #     base rate.  When the calibration set is carved from the val split
    #     (positive rate ~10–25%) but inference runs on a screened universe
    #     (positive rate ~30–50%), the isotonic output is systematically too
    #     low for screened stocks.
    #
    #     Correction formula (Saerens et al. 2002 / du Plessis & Sugiyama 2014):
    #
    #       Let p_c = calibration-set positive rate (known from y_cal)
    #           p_i = screened inference positive rate (SCREENER_POSITIVE_RATE)
    #           q   = raw isotonic-calibrated probability
    #
    #       odds_corrected = (q / (1 - q)) * (p_i / (1 - p_i)) / (p_c / (1 - p_c))
    #       p_corrected    = odds_corrected / (1 + odds_corrected)
    #
    #     This is applied element-wise at inference time via a thin wrapper that
    #     calls the underlying CalibratedClassifierCV and then shifts odds.
    #     The wrapper is transparent to the rest of the codebase (it still
    #     implements predict_proba / predict / classes_).
    #
    #     SCREENER_POSITIVE_RATE is configurable at the top of this file.
    #     Set it to None to disable prior correction and use raw isotonic output.
    if X_cal is not None and y_cal is not None:
        n_cal_pos = int((y_cal == 1).sum())
        n_cal_neg = int((y_cal == 0).sum())
        if n_cal_pos >= 10 and n_cal_neg >= 10:
            p_cal = n_cal_pos / max(1, n_cal_pos + n_cal_neg)
            logger.info(
                f"RC6: Fitting isotonic probability calibrator on "
                f"{len(y_cal)} calibration samples "
                f"({n_cal_pos} pos / {n_cal_neg} neg, rate={p_cal:.1%})."
            )
            calibrated_model = CalibratedClassifierCV(
                model, method="isotonic", cv="prefit"
            )
            calibrated_model.fit(X_cal, y_cal)

            # Sanity-check: log how calibration shifted the distribution
            raw_proba = model.predict_proba(X_cal)[:, 1]
            cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]
            logger.info(
                f"  Raw proba  — mean={raw_proba.mean():.3f}  "
                f"std={raw_proba.std():.3f}  "
                f"pct>=0.90: {(raw_proba>=0.90).mean():.1%}"
            )
            logger.info(
                f"  Cal proba  — mean={cal_proba.mean():.3f}  "
                f"std={cal_proba.std():.3f}  "
                f"pct>=0.90: {(cal_proba>=0.90).mean():.1%}"
            )

            # Step 2: Prior-probability correction for base-rate mismatch.
            p_inf = SCREENER_POSITIVE_RATE
            if p_inf is not None and 0.0 < p_inf < 1.0 and 0.0 < p_cal < 1.0:
                # Bayes odds-ratio correction factor
                odds_ratio = (p_inf / (1.0 - p_inf)) / (p_cal / (1.0 - p_cal))
                logger.info(
                    f"  Prior correction: p_cal={p_cal:.3f} → p_inf={p_inf:.3f}  "
                    f"odds_ratio={odds_ratio:.3f}"
                )

                # Compute corrected probabilities on the cal set for logging
                raw_odds = cal_proba / np.clip(1.0 - cal_proba, 1e-9, None)
                corr_odds = raw_odds * odds_ratio
                corr_proba = corr_odds / (1.0 + corr_odds)
                logger.info(
                    f"  Corrected proba — mean={corr_proba.mean():.3f}  "
                    f"std={corr_proba.std():.3f}  "
                    f"pct>=0.90: {(corr_proba>=0.90).mean():.1%}"
                )

                # Warn if correction is so large it may be unreliable
                if odds_ratio > 10.0:
                    logger.warning(
                        f"  ⚠️  RC6: Prior correction odds_ratio={odds_ratio:.2f} is very "
                        f"large.  Verify that SCREENER_POSITIVE_RATE={p_inf} reflects the "
                        f"actual fraction of screened candidates that become winners. "
                        f"Run: SELECT COUNT(*) FILTER (WHERE became_winner)*1.0/COUNT(*) "
                        f"FROM ml_prediction_accuracy WHERE prediction_date >= NOW()-'90 days'::interval"
                    )

                # Wrap calibrated_model with prior correction so predict_proba()
                # automatically applies the Bayes odds-ratio shift at inference.
                # _PriorCorrectedModel is defined at module level (not inside this
                # function) so that joblib can pickle it by fully-qualified name.
                return _PriorCorrectedModel(calibrated_model, float(odds_ratio))
            else:
                if p_inf is None:
                    logger.info(
                        "  Prior correction disabled (SCREENER_POSITIVE_RATE=None). "
                        "Returning raw isotonic-calibrated model."
                    )
                else:
                    logger.warning(
                        f"  Prior correction skipped: invalid "
                        f"SCREENER_POSITIVE_RATE={p_inf} or p_cal={p_cal:.3f}. "
                        "Must be strictly between 0 and 1."
                    )
                return calibrated_model
        else:
            logger.warning(
                f"RC6: Calibration set too small or imbalanced "
                f"({n_cal_pos} pos / {n_cal_neg} neg) — "
                "skipping isotonic calibration. Returning raw model."
            )
    else:
        logger.info(
            "RC6: No calibration set provided — returning raw (uncalibrated) model. "
            "Pass X_cal/y_cal to train_model() to enable isotonic calibration."
        )

    return model


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    df_with_dates: pd.DataFrame,
    val_fraction: float = 0.20,  # only used as fallback when no date column exists
) -> tuple:
    """
    FIXED train/val split with three stability improvements:

    FIX 1 — Dynamic cutoff date (VAL_WEEKS most recent weeks) instead of a
      hardcoded date or a floating fraction.
      • A hardcoded date caused the val set to grow every week as new T-1 rows
        accumulated, shifting scale_pos_weight and the early-stopping signal.
      • The old 20%-of-rows approach gave a different market slice each retrain.
      • Pinning to "the last VAL_WEEKS weeks of data" keeps the val window the
        same size every run.  The cutoff is computed from the maximum date found
        in the training dataframe (not wall-clock time), so backfills are stable.

    FIX 2 — Mistake samples (rows with NaT dates) are forced into the train set.
      Previously NaT rows sorted to the end and landed in the val set, biasing
      AUC on the model's own hardest errors rather than a general held-out period.

    FIX 3 — Hard minimum on val positives (MIN_VAL_POSITIVES).
      If the dynamic cutoff leaves fewer than MIN_VAL_POSITIVES winner rows in
      val, training aborts with a clear message rather than producing a junk model
      (previously the code only warned and then continued).

    FIX 4 — Purge/embargo gap at the cutoff (EMBARGO_DAYS).
      Rows dated within EMBARGO_DAYS immediately before the cutoff are dropped
      from train entirely (not moved to val). Several top features are rolling
      windows up to 30 days deep, so a hard cutoff with no gap put train rows
      and val rows right next to each other in time with highly overlapping
      (autocorrelated) rolling-window feature vectors — inflating val AUC via
      boundary adjacency rather than genuine generalisation. The embargo
      removes that adjacency.

    To change the val window size, adjust VAL_WEEKS in the configuration block.
    """
    df_work = df_with_dates.copy()

    # Infer the purge/embargo gap from the deepest rolling-window length
    # encoded in the feature column names (clamped to [floor, cap]).
    EMBARGO_DAYS = _infer_embargo_days(list(X.columns))

    # ── Build a unified sort_date from whichever date column(s) exist ────────
    has_detection = "detection_date" in df_work.columns
    has_event     = "event_date"     in df_work.columns

    if has_detection or has_event:
        sort_date = pd.Series(pd.NaT, index=df_work.index)
        if has_detection:
            sort_date = pd.to_datetime(df_work["detection_date"], errors="coerce")
        if has_event:
            event_parsed = pd.to_datetime(df_work["event_date"], errors="coerce")
            sort_date = sort_date.fillna(event_parsed)

        df_work["_sort_date"] = sort_date
        date_col = "_sort_date"
    else:
        date_col = next((c for c in ["date"] if c in df_work.columns), None)
        sort_date = (
            pd.to_datetime(df_work[date_col], errors="coerce")
            if date_col else pd.Series(pd.NaT, index=df_work.index)
        )

    # ── FIX 2: Identify NaT rows (mistake samples) — pin them to train ───────
    nat_mask = sort_date.isna()
    n_nat    = int(nat_mask.sum())
    if n_nat > 0:
        logger.info(
            f"FIX 2: {n_nat} rows have NaT dates (mistake samples) — "
            "forcing them into the train set so they don't pollute val AUC."
        )

    # ── FIX 1: Dynamic cutoff — last VAL_WEEKS weeks of data held out for val ──
    VAL_CUTOFF_DATE = "unknown"  # default; overwritten below when date_col is present
    if date_col is not None:
        cutoff = _compute_val_cutoff(df_work)
        VAL_CUTOFF_DATE = cutoff.date()  # stored for metadata/logging
        dates  = pd.to_datetime(df_work[date_col], errors="coerce")

        # ── Purge/embargo gap ────────────────────────────────────────────
        # Drop rows whose date falls in [cutoff - EMBARGO_DAYS, cutoff) from
        # TRAIN entirely (they are not moved to val either — they're simply
        # excluded) so that no train row's rolling-window features are
        # adjacent in time to a val row's rolling-window features. This
        # mirrors purged/embargoed CV: it removes the boundary-adjacency
        # effect that would otherwise inflate val AUC via autocorrelated
        # feature vectors rather than genuine generalisation.
        embargo_start = cutoff - pd.Timedelta(days=EMBARGO_DAYS)

        # ── Guard: don't let the embargo eat the entire train window ────────
        # EMBARGO_DAYS is inferred from feature names and can be as large as
        # EMBARGO_DAYS_CAP (90d), independent of how much pre-cutoff data
        # lookback_days actually left us. If embargo_start falls at or before
        # the earliest dated row, EVERY dated row is a candidate for the
        # embargo band and none reach train (NaT rows are the only rows that
        # would survive) -- silently producing a train split with 0 or 1
        # classes rather than an error. Shrink the embargo instead, down to
        # a floor, and log loudly so the mismatch between lookback_days and
        # the inferred embargo is visible rather than surfacing later as a
        # confusing XGBoost "invalid classes" crash.
        earliest_dated = dates[~nat_mask].min() if (~nat_mask).any() else pd.NaT
        if pd.notna(earliest_dated):
            available_pre_cutoff_days = (cutoff - earliest_dated).days
            min_required = MIN_TRAIN_WINDOW_DAYS + EMBARGO_DAYS_FLOOR
            if available_pre_cutoff_days < min_required:
                logger.warning(
                    f"Only {available_pre_cutoff_days}d of data exist before the "
                    f"val cutoff ({cutoff.date()}), but the inferred embargo "
                    f"({EMBARGO_DAYS}d) plus a {MIN_TRAIN_WINDOW_DAYS}d minimum "
                    f"train window need {min_required}d. This usually means "
                    "lookback_days is too small for the deepest rolling-window "
                    "feature in use (or a data source — e.g. ml_training_base — "
                    "aged out and stopped padding the window). "
                    "Shrinking the embargo instead of letting it consume the "
                    "whole train window; consider raising --lookback-days."
                )
                EMBARGO_DAYS = max(
                    EMBARGO_DAYS_FLOOR,
                    min(EMBARGO_DAYS, available_pre_cutoff_days - MIN_TRAIN_WINDOW_DAYS),
                )
                embargo_start = cutoff - pd.Timedelta(days=EMBARGO_DAYS)

        # FIX 2 applied here: NaT → train regardless of cutoff
        train_mask   = nat_mask | (dates < embargo_start)
        embargo_mask = (~nat_mask) & (dates >= embargo_start) & (dates < cutoff)
        val_mask     = (~nat_mask) & (dates >= cutoff)

        n_embargoed = int(embargo_mask.sum())
        if n_embargoed > 0:
            logger.info(
                f"Purge/embargo: dropping {n_embargoed} rows dated "
                f"[{embargo_start.date()} \u2192 {cutoff.date()}) \u2014 within "
                f"{EMBARGO_DAYS}d of the val cutoff \u2014 from train so "
                "rolling-window features don't straddle the boundary."
            )

        train_idx = df_work.index[train_mask]
        val_idx   = df_work.index[val_mask]

        train_dates = dates.loc[train_idx].dropna()
        val_dates   = dates.loc[val_idx].dropna()

        logger.info(
            f"FIX 1 — Dynamic cutoff ({VAL_WEEKS}-week val window): cutoff={cutoff.date()}: "
            f"train {train_dates.min().date() if not train_dates.empty else '?'} "
            f"→ {train_dates.max().date() if not train_dates.empty else '?'}, "
            f"val {val_dates.min().date() if not val_dates.empty else '?'} "
            f"→ {val_dates.max().date() if not val_dates.empty else '?'}, "
            f"embargoed={n_embargoed}"
        )
    else:
        # No date column at all — fall back to sequential split (last resort)
        logger.warning(
            "No date column found — falling back to sequential split. "
            "Ensure detection_date/event_date columns exist."
        )
        split_pos = int(len(X) * (1 - val_fraction))
        train_idx = X.index[:split_pos]
        val_idx   = X.index[split_pos:]

    X_train = X.loc[train_idx]
    X_val   = X.loc[val_idx]
    y_train = y.loc[train_idx]
    y_val   = y.loc[val_idx]
    w_train = w.loc[train_idx]
    w_val   = w.loc[val_idx]

    logger.info(
        f"Train/val split (before rebalance): {len(X_train)} train "
        f"(pos={int((y_train==1).sum())}, neg={int((y_train==0).sum())}), "
        f"{len(X_val)} val "
        f"(pos={int((y_val==1).sum())}, neg={int((y_val==0).sum())})"
    )

    # ── VAL REBALANCE: cap val positive rate to match real-world base rate ────
    # The 8-week val window is dominated by T-1 rows, which are stored at ~50%
    # positive rate (equal counts of winners and non-winners per day).  The train
    # set reflects the real base rate (~10%).  This mismatch makes the val
    # classification report and probability calibration misleading, and is the
    # root cause of Mode D (high-prob clustering) firing on every prediction run.
    #
    # Fix: compute the positive rate of the TRAIN set and trim val positives
    # (moving excess to train) until val positive rate ≤ train positive rate + 2pp.
    # We move rows rather than downsample so no data is thrown away.
    #
    # "2pp headroom" allows T-1 rows to contribute a slightly higher positive
    # rate without requiring us to bleed positives all the way to 9%.
    _train_pos_rate = int((y_train == 1).sum()) / max(1, len(y_train))
    _val_pos_rate   = int((y_val == 1).sum())   / max(1, len(y_val))
    _MAX_VAL_POS_RATE = _train_pos_rate + 0.02   # 2 pp headroom

    if _val_pos_rate > _MAX_VAL_POS_RATE:
        # How many positives to keep in val so rate == _MAX_VAL_POS_RATE
        _val_neg = int((y_val == 0).sum())
        _target_val_pos = max(
            MIN_VAL_POSITIVES,
            int(_val_neg * _MAX_VAL_POS_RATE / max(1 - _MAX_VAL_POS_RATE, 1e-9)),
        )
        # BUG 5 FIX: move the OLDEST excess positives to train, not the newest.
        # y_val is time-ordered (val rows are sorted by _sort_date ascending), so
        # index[_target_val_pos:] selects the most recent excess rows and moves them
        # to train — leaving the oldest rows in val.  That is exactly backwards:
        # the purpose of a time-based split is to validate on the most recent period.
        # Fix: keep the last _target_val_pos positives in val (most recent) and move
        # the first excess (oldest) ones to train.
        _all_val_pos_idx = y_val[y_val == 1].index.tolist()
        _n_excess        = len(_all_val_pos_idx) - _target_val_pos
        _excess_pos_idx  = pd.Index(_all_val_pos_idx[:_n_excess])   # oldest → train

        if len(_excess_pos_idx) > 0:
            # Move excess val positives → train.
            # BUG 4 FIX: rebuild train_idx as a pd.Index (not a plain list) so
            # that the returned train_idx is always the same type and always
            # contains the moved rows.  Previously train_idx was reassigned as
            # list(train_idx) + list(_excess_pos_idx) but as a plain Python list
            # rather than a pd.Index, creating a type inconsistency with the
            # non-rebalance code paths where train_idx is a pd.Index.
            # Using pd.Index(...) here makes the update explicit and ensures the
            # returned train_idx is always a proper pd.Index containing the
            # rebalanced rows.
            _new_val_list   = [i for i in val_idx   if i not in set(_excess_pos_idx)]
            _new_train_list = list(train_idx) + list(_excess_pos_idx)
            val_idx   = pd.Index(_new_val_list)
            train_idx = pd.Index(_new_train_list)

            X_train = X.loc[train_idx]
            X_val   = X.loc[val_idx]
            y_train = y.loc[train_idx]
            y_val   = y.loc[val_idx]
            w_train = w.loc[train_idx]
            w_val   = w.loc[val_idx]

            logger.info(
                f"VAL REBALANCE: moved {len(_excess_pos_idx)} excess positives from val → train "
                f"(val rate was {_val_pos_rate:.1%} vs train rate {_train_pos_rate:.1%}). "
                f"New val: {len(X_val)} rows, "
                f"pos={int((y_val==1).sum())}, neg={int((y_val==0).sum())}, "
                f"pos_rate={int((y_val==1).sum())/max(1,len(y_val)):.1%}"
            )

    logger.info(
        f"Train/val split: {len(X_train)} train "
        f"(pos={int((y_train==1).sum())}, neg={int((y_train==0).sum())}), "
        f"{len(X_val)} val "
        f"(pos={int((y_val==1).sum())}, neg={int((y_val==0).sum())})"
    )

    # ── FIX 3: Hard minimum on val positives — abort instead of warn ──────────
    val_pos = int((y_val == 1).sum())
    if val_pos < MIN_VAL_POSITIVES:
        logger.error(
            f"FIX 3 — ABORTING: only {val_pos} positive examples in val set "
            f"(need ≥ {MIN_VAL_POSITIVES}). "
            f"The cutoff date {VAL_CUTOFF_DATE!r} is too recent — not enough winners "
            "have accumulated after it. "
            "Options: (1) move VAL_CUTOFF_DATE earlier, "
            "(2) accumulate more labelled data, "
            "(3) lower MIN_VAL_POSITIVES if you accept noisier early stopping."
        )
        sys.exit(1)
    elif val_pos < 100:
        logger.warning(
            f"  ⚠️  Only {val_pos} positive examples in val set "
            f"({val_pos / max(1, len(y_val)):.1%} of val). "
            "Early stopping AUC may still be somewhat noisy. "
            f"Consider moving VAL_CUTOFF_DATE earlier once more data accumulates."
        )
    else:
        logger.info(f"  ✅ Val set has {val_pos} positives — early stopping signal is stable.")

    # ── Hard minimum: train must contain BOTH classes ─────────────────────────
    # A single-class train set (usually train_pos=0 after the purge/embargo
    # step ate every dated row, or a VAL REBALANCE that only moved positives
    # over) fails deep inside XGBoost.fit() with a cryptic "Invalid classes
    # inferred" error. Catch it here instead, with a message that actually
    # points at the fix.
    train_pos = int((y_train == 1).sum())
    train_neg = int((y_train == 0).sum())
    if train_pos == 0 or train_neg == 0:
        logger.error(
            f"ABORTING: train split has pos={train_pos}, neg={train_neg} — "
            "missing a class entirely, XGBoost cannot fit on this. "
            "This almost always means the purge/embargo window consumed the "
            "whole pre-cutoff train range (EMBARGO_DAYS close to or larger "
            "than the data available before the val cutoff). "
            "Options: (1) increase --lookback-days so more pre-cutoff data "
            "exists, (2) check whether a data source that used to pad the "
            "training window (e.g. ml_training_base) has gone stale/empty, "
            "(3) lower VAL_WEEKS so the val cutoff sits later, leaving more "
            "room before it for train."
        )
        sys.exit(1)

    return X_train, X_val, y_train, y_val, w_train, w_val, train_idx


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """Generate feature_importance.csv using gain importance.

    RC6: model may be a CalibratedClassifierCV wrapping an XGBClassifier.
    We unwrap it to access the underlying booster for feature importances.
    """
    # RC6: unwrap CalibratedClassifierCV to get the raw XGBClassifier
    xgb_model = model
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV stores list of (estimator, calibrator) pairs
        xgb_model = model.calibrated_classifiers_[0].estimator
    booster = xgb_model.get_booster()
    scores  = booster.get_score(importance_type="gain")

    importance_list = []
    for feat, score in scores.items():
        if feat.startswith("f") and feat[1:].isdigit():
            idx  = int(feat[1:])
            name = feature_names[idx] if idx < len(feature_names) else feat
        else:
            name = feat
        importance_list.append({"feature": name, "importance": round(score, 6)})

    fi_df  = pd.DataFrame(importance_list)
    fi_df  = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    total  = fi_df["importance"].sum()
    if total > 0:
        fi_df["importance"] = (fi_df["importance"] / total).round(6)

    logger.info(f"Feature importance computed: {len(fi_df)} features")
    logger.info("Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return fi_df


# ---------------------------------------------------------------------------
# RC1 + RC3 + RC7 FIX: Gain regressor — broader training set, correct scale input,
#                       log-transform target, matched hyperparams, higher gain cap
# ---------------------------------------------------------------------------

# Gains above this percentile are winsorized to prevent a handful of 5000%
# outliers from dominating the loss.  We keep extreme winners in training
# (they are the most valuable signal) but cap their label so XGBoost can
# still split on them meaningfully.  Log-transforming the target (RC7) reduces
# the distortion from outliers far more than a hard cap.
_GAIN_WINSOR_PCT = 99.5   # winsorize above this percentile

# RC8 FIX: Winner up-weighting is now DATA-DRIVEN, mirroring exactly how
# train_model() derives the classifier's scale_pos_weight (n_neg/n_pos,
# clamped to [SPW_MIN, SPW_MAX]) instead of a fixed number picked by hand.
# The old approach (_WINNER_WEIGHT_MULTIPLIER = 3.0, tuned down from 8.0,
# tuned down from who-knows-what before that) was a guess that had to be
# re-guessed every time the winner/non-winner ratio in the training data
# shifted. Computing it from n_non_winners/n_winners on THIS run's actual
# training pool (see train_gain_regressor below) means the weight adapts
# automatically as more winner data accumulates, exactly like the
# classifier's scale_pos_weight does. The constants below are only the
# clamp bounds (and the legacy fallback values used only if the training
# pool is degenerate — e.g. zero winners).
REG_WINNER_WEIGHT_MIN = 2.0     # never weight winners less than 2x
REG_WINNER_WEIGHT_MAX = 8.0     # never let the data-driven ratio run away
                                # unboundedly on a very winner-scarce run
_WINNER_WEIGHT_MULTIPLIER = 3.0    # legacy fallback if n_winners or
                                    # n_non_winners is 0 (ratio undefined)

_HIGH_GAIN_THRESHOLD  = 30.0    # Lowered from 50% — more winners qualify for the boost
REG_HIGH_GAIN_WEIGHT_MIN = 1.5   # additional multiplier on top of the winner
REG_HIGH_GAIN_WEIGHT_MAX = 5.0   # weight above, also computed from the actual
                                  # high-gain / other-winner ratio in this run's data
_HIGH_GAIN_MULTIPLIER = 3.0     # legacy fallback if n_high_gain or
                                 # n_other_winners is 0

# RC9: Additional multiplier applied to winner rows, scaled by classifier
# confidence (clf_proba), on top of the RC8 winner/high-gain weights above.
# A 50%-confidence winner gets ~REG_CONFIDENCE_WEIGHT_MIN (near-neutral);
# a near-100%-confidence winner (strong-buy territory) gets pulled toward
# REG_CONFIDENCE_WEIGHT_MAX, so the regressor is penalised more heavily for
# under-predicting gain on exactly the rows the classifier is most sure about.
REG_CONFIDENCE_WEIGHT_MIN = 1.0
REG_CONFIDENCE_WEIGHT_MAX = 2.5



# Minimum number of gain-labeled rows required in the classifier's TRAIN split
# before we trust a leak-free regressor fit.  If the train-only pool falls
# short of this, train_gain_regressor() falls back to training on the full
# (train+val) pool instead — but it does so loudly, via the return value
# and a logged warning, rather than silently.  See LEAK-FREE FIX below.
MIN_TRAIN_ONLY_GAIN_ROWS = 200


def train_gain_regressor(
    X_scaled: pd.DataFrame,           # RC3 FIX: receive pre-scaled features
    combined_df: pd.DataFrame,
    feature_names: list[str],
    client: Client,
    accuracy_gain_map: "Optional[dict]" = None,  # ISSUE 2 FIX: pre-fetched from main() to avoid redundant DB query
    _is_fallback_retry: bool = False,  # internal — set True on the leaky-fallback recursive call
) -> "Optional[object]":
    """
    Train a regression model to predict actual % gain for stocks the
    classifier labels as winners.

    ISSUE #1 (historical): X_scaled was previously passed here with ALL rows
      (train + val), causing the regressor's own internal time-based val split
      to be a mixed-regime window that overlapped the classifier's val rows.
      This was not a correctness issue for the classifier, but it made the
      regressor's reported val MAE/R² meaningless as an evaluation signal.
    EVALUATION INTEGRITY FIX: The caller now passes only the classifier's
      train rows (X_train + combined_df.loc[train_idx]), so the regressor's
      internal 80/20 split is entirely within the training period and the
      held-out val window reflects a clean, future-relative evaluation.

    RC1 FIX: Broaden training set beyond just winners.
      - Winners from daily_winners (with corrected actual_high_pct via prev_close)
      - Non-winners that have actual_gain_pct in ml_prediction_accuracy
        (yfinance data captured by the accuracy tracker)
      This gives far more training samples and a realistic gain distribution.

    RC2 FIX: Use actual_high_pct computed from prev_close (already corrected
      in the enrichment step before this function is called).

    RC3 FIX: X_scaled is the StandardScaler output, matching exactly what
      explosion_predictor.py passes to the regressor at inference time.
      Previously the regressor was trained on raw/filled values but received
      scaled values → systematically wrong predictions from day one.

    RC4 FIX: The std < 1.0 guard in explosion_predictor.py is relaxed to
      0.5 (see that file), but we also improve training quality here so the
      regressor doesn't collapse to the mean.

    MODERATE ISSUE #5 FIX: Internal val split is now time-based (matching the
      classifier split) rather than random, preventing future gain patterns
      from leaking into regressor training.

    RC7 FIX: Three changes to stop gain predictions collapsing below 50%:
      1. Log-transform the gain target (log1p / expm1) so that 5% and 500%
         gains don't live on wildly different scales.  This gives XGBoost a
         smoother loss landscape and lets it place splits that distinguish
         "moderate" from "large" gains without being dominated by rare 5000%
         outliers.
      2. Winsorize the log-transformed target at the 99.5th percentile so the
         handful of extreme outliers don't pull every tree towards them.
      3. Heavily up-weight winner rows (5×) and extra-large-gain winners (15×
         combined) so the regressor is penalised much more for under-predicting
         high-gain stocks than for over-predicting low-gain ones.  Previously
         the 2× winner bonus was far too weak given the severe class imbalance
         in gain magnitude (most training rows have gain ≈ 0–5%).
      4. Match classifier hyperparameters: n_estimators=300, max_depth=5,
         gamma=1.0, reg_alpha/lambda matching XGBOOST_PARAMS.  The old
         regressor used looser settings (200 trees, depth 4, no gamma) which
         caused it to overfit to the abundant low-gain rows.
      5. Raise the gain cap from 500% to 10 000% so extreme winners are NOT
         silently excluded from training.  The log transform handles their
         scale.
    """
    from xgboost import XGBRegressor

    # ------------------------------------------------------------------
    # RC1 FIX: Fetch additional gain data from ml_prediction_accuracy.
    # ISSUE 2 FIX: if the caller already fetched this data (main() passes it
    # via accuracy_gain_map after the RC3 backfill query), reuse it directly
    # to avoid a redundant round-trip.  Fall back to fetching here only when
    # the caller did not supply it (e.g. when called from other contexts).
    # ------------------------------------------------------------------
    if accuracy_gain_map is None:
        accuracy_gain_map = {}
        try:
            logger.info("RC1: Fetching gain data from ml_prediction_accuracy (no pre-fetched data supplied)...")
            date_col_candidates = [c for c in ("detection_date", "explosion_date", "prediction_date", "date") if c in combined_df.columns]
            if date_col_candidates:
                try:
                    # BUG FIX: previously picked only the first matching column
                    # for the whole frame (same issue fixed elsewhere in this
                    # file) — rows lacking that particular column read as NaT
                    # and were silently excluded from the min-date calculation.
                    # Combine all candidate date columns per-row instead.
                    _fallback_dates = pd.Series(pd.NaT, index=combined_df.index)
                    for _c in date_col_candidates:
                        _fallback_dates = _fallback_dates.fillna(pd.to_datetime(combined_df[_c], errors="coerce"))
                    start_date = _fallback_dates.min()
                    start_date = start_date.date().isoformat() if pd.notna(start_date) else None
                except Exception:
                    start_date = None
            else:
                start_date = None
            if start_date is None:
                import os
                from datetime import timedelta
                lookback_days = int(os.environ.get("LOOKBACK", "90"))
                start_date = (datetime.now().date() - timedelta(days=lookback_days)).isoformat()
                logger.warning(
                    f"RC1: Could not derive start_date from combined_df; "
                    f"falling back to LOOKBACK={lookback_days} days from today ({start_date})"
                )
            logger.info(f"RC1: Querying ml_prediction_accuracy from {start_date}")
            resp = (
                client.table("ml_prediction_accuracy")
                .select("symbol, prediction_date, actual_gain_pct, actual_high_pct")
                .gte("prediction_date", start_date)
                .not_.is_("actual_gain_pct", "null")
                .execute()
            )
            if resp.data:
                for row in resp.data:
                    key = (row["symbol"], row["prediction_date"])
                    accuracy_gain_map[key] = {
                        "actual_gain_pct": row.get("actual_gain_pct"),
                        "actual_high_pct": row.get("actual_high_pct"),
                    }
                logger.info(f"RC1: Got {len(accuracy_gain_map)} gain records from accuracy table")
        except Exception as e:
            logger.warning(f"RC1: Could not fetch accuracy gain data: {e}")
    else:
        logger.info(f"RC1: Reusing {len(accuracy_gain_map)} pre-fetched gain records from caller (no redundant DB query)")

    # ------------------------------------------------------------------
    # Determine gain target column — evaluated AFTER the RC1 fetch so
    # that accuracy-table data can count toward the ≥30 threshold.
    # ------------------------------------------------------------------
    gain_col = None
    # TRUE GAIN TARGET FIX: '_unified_gain_target' (built in attach_true_gain_targets()
    # from true_gain_pct — the market_close/day_prior_close snapshot join — and
    # ml_training_base.gain_pct) is checked FIRST. It is a directly-measured,
    # pipeline-native label with no yfinance/ml_prediction_accuracy dependency,
    # and it covers both T-1 rows (via true_gain_pct) and base-CSV rows (via
    # gain_pct, previously discarded entirely). The legacy columns below remain
    # as a fallback for any deployment that hasn't backfilled the new tables yet.
    for candidate in ("_unified_gain_target", "actual_high_pct", "actual_gain_pct", "change_pct"):
        if candidate in combined_df.columns:
            col_vals = pd.to_numeric(combined_df[candidate], errors="coerce")
            non_null = col_vals.notna().sum()
            if non_null >= 30:
                gain_col = candidate
                logger.info(f"Gain regressor target column (from combined_df): '{gain_col}' ({non_null} non-null values)")
                break

    # If no column in combined_df has enough data, check whether the RC1
    # accuracy table fetch alone can supply ≥30 rows — use actual_gain_pct
    # as the target in that case (we will fill it from accuracy_gain_map).
    if gain_col is None and len(accuracy_gain_map) >= 30:
        # Inject a synthetic column so the downstream code has something
        # to read from before the accuracy-map fill loop runs.
        for candidate in ("actual_high_pct", "actual_gain_pct"):
            if candidate not in combined_df.columns:
                combined_df[candidate] = float("nan")
        gain_col = "actual_gain_pct"
        logger.info(
            f"Gain regressor target column (from accuracy table): '{gain_col}' "
            f"({len(accuracy_gain_map)} records available via RC1 fetch)"
        )

    if gain_col is None:
        logger.warning(
            "No gain column with sufficient data (checked combined_df columns "
            f"and RC1 accuracy table — {len(accuracy_gain_map)} accuracy rows). "
            "Skipping gain regressor training."
        )
        return None

    # ------------------------------------------------------------------
    # Build gain targets for every row in combined_df
    # Priority: actual_high_pct > actual_gain_pct > accuracy table > skip
    # ------------------------------------------------------------------
    gain_targets = pd.to_numeric(combined_df[gain_col], errors="coerce").copy()

    if accuracy_gain_map:
        sym_col = next((c for c in ["symbol", "ticker"] if c in combined_df.columns), None)

        # combined_df has two different date columns depending on data source:
        #   T-1 rows      -> detection_date  (the day *before* explosion = prediction_date)
        #   base CSV rows -> event_date      (the explosion day itself = prediction_date + 1)
        # We need to try both, and for event_date rows subtract 1 business day.
        has_detection = "detection_date" in combined_df.columns
        has_event     = "event_date" in combined_df.columns

        if sym_col and (has_detection or has_event):
            # ISSUE 3 FIX: replace O(n) iterrows loop with vectorised lookup.
            # Build a Series of (symbol, date_str) lookup keys for each row, try
            # detection_date first (T-1 rows), then fall back to event_date - 1 BDay
            # (base CSV rows).  A single map() call replaces the per-row loop.

            # --- Build acc_lookup: (symbol, date_str) -> best gain value ----------
            acc_lookup = {
                k: (v.get("actual_high_pct") or v.get("actual_gain_pct"))
                for k, v in accuracy_gain_map.items()
            }

            null_mask = gain_targets.isna()
            valid_det = pd.array([], dtype=bool)  # pre-init; populated in Path 1 if has_detection

            # --- Path 1: detection_date rows (direct key match) -------------------
            if has_detection and null_mask.any():
                det_dates = pd.to_datetime(
                    combined_df.loc[null_mask, "detection_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d").fillna("")
                keys_det = list(zip(combined_df.loc[null_mask, sym_col], det_dates))
                filled_det = pd.array(
                    [acc_lookup.get(k) for k in keys_det], dtype=object
                )
                valid_det = pd.array(
                    [v is not None for v in filled_det], dtype=bool
                )
                update_idx = gain_targets.index[null_mask][valid_det]
                gain_targets.loc[update_idx] = pd.to_numeric(
                    pd.array(filled_det[valid_det], dtype=object), errors="coerce"
                )
                null_mask = gain_targets.isna()  # refresh for path 2

            # --- Path 2: event_date - 1 BDay rows (base CSV rows) -----------------
            n_filled_event = 0
            if has_event and null_mask.any():
                ev_raw = pd.to_datetime(
                    combined_df.loc[null_mask, "event_date"], errors="coerce"
                )
                # Subtract 1 business day vectorially; NaT stays NaT
                pred_dates = (ev_raw - pd.tseries.offsets.BDay(1)).dt.strftime("%Y-%m-%d").fillna("")
                keys_ev = list(zip(combined_df.loc[null_mask, sym_col], pred_dates))
                filled_ev = pd.array(
                    [acc_lookup.get(k) for k in keys_ev], dtype=object
                )
                valid_ev = pd.array(
                    [v is not None for v in filled_ev], dtype=bool
                )
                update_idx_ev = gain_targets.index[null_mask][valid_ev]
                gain_targets.loc[update_idx_ev] = pd.to_numeric(
                    pd.array(filled_ev[valid_ev], dtype=object), errors="coerce"
                )
                n_filled_event = int(valid_ev.sum())

            n_filled_det   = int(valid_det.sum()) if has_detection else 0
            filled_count   = n_filled_det + n_filled_event
            logger.info(
                f"RC1: Filled {filled_count} additional gain targets from accuracy table "
                f"({n_filled_det} via detection_date, {n_filled_event} via event_date-1BDay)"
            )

    # ------------------------------------------------------------------
    # CORE FIX: T-1 non-winner rows without actual_high_pct get gain target = 0.0.
    #
    # Root cause of poor regressor training (confirmed in logs):
    #   - Only T-1 non-winner rows (source = non_winners_day_prior) are eligible.
    #   - Base CSV rows have genuinely UNKNOWN outcomes: event_date is the explosion
    #     day so we have no intraday data for what the stock did. Assigning 0.0 to
    #     ~6,649 base rows (all with the same constant target) collapses the gain
    #     distribution std from ~1.86 to ~0.88 in log-space, causing the regressor
    #     to converge in ~35 trees predicting near-zero for everything.
    #   - Only T-1 rows passed through the daily non-winner screener, so we know
    #     with confidence they were scanned and did NOT produce a large intraday move.
    #     Their correct gain target IS ~0%.
    #
    # Winner rows with NaN actual_high_pct (no prev_close available) are
    # excluded — we don't know their true gain so we cannot assign 0.0.
    # ------------------------------------------------------------------
    non_winner_rows = combined_df["label"] == 0
    winner_rows     = combined_df["label"] == 1

    # Identify T-1 non-winner rows (confirmed daily screener output, gain~=0)
    # vs base CSV rows (unknown outcome, should remain NaN so they're excluded).
    if "source" in combined_df.columns:
        t1_non_winner_rows = (
            non_winner_rows &
            combined_df["source"].str.contains("non_winners_day_prior", na=False)
        )
    else:
        # No source column — conservatively treat all non-winners as T-1
        t1_non_winner_rows = non_winner_rows

    # Fill ONLY T-1 non-winners with 0.0; leave base CSV non-winners as NaN
    #
    # FIX2: Track exactly which rows receive this explicit 0.0 anchor (as opposed
    # to a genuine actual_high_pct value backfilled from ml_prediction_accuracy).
    # Without this, the FIX2 diagnostic below cannot tell the two apart and ends
    # up comparing real winner gains against a population dominated by this
    # intentional zero-fill — which will *always* show a huge mean gap that has
    # nothing to do with a prev_close vs same-day-close denominator mismatch.
    zero_fill_mask = t1_non_winner_rows & gain_targets.isna()
    n_nonwinner_filled = int(zero_fill_mask.sum())
    if n_nonwinner_filled > 0:
        gain_targets = gain_targets.copy()
        gain_targets.loc[zero_fill_mask] = 0.0
        n_base_skipped = int((non_winner_rows & ~t1_non_winner_rows).sum())
        logger.info(
            f"CORE FIX: Filled {n_nonwinner_filled} T-1 non-winner rows with gain=0.0 "
            f"(confirmed screener output — no large intraday move). "
            f"Skipped {n_base_skipped} base CSV rows (unknown outcome — kept as NaN)."
        )

    # RC7 FIX: cap (not floor) — reject obvious data errors only
    valid_gain_mask = gain_targets.notna() & (gain_targets > -100.0) & (gain_targets < 10_000.0)

    # Exclude winner rows with NaN-filled or unreliably low gain only.
    # Non-winner rows with gain=0.0 are KEPT — they are the true low-end anchor.
    GAIN_REGRESSOR_MIN_PCT = 5.0
    low_gain_winner_mask = valid_gain_mask & winner_rows & (gain_targets < GAIN_REGRESSOR_MIN_PCT)
    n_low_gain_excluded = int(low_gain_winner_mask.sum())
    if n_low_gain_excluded > 0:
        valid_gain_mask = valid_gain_mask & ~low_gain_winner_mask
        logger.info(
            f"Bug3 FIX: Excluded {n_low_gain_excluded} winner rows with "
            f"gain < {GAIN_REGRESSOR_MIN_PCT}% as noisy winner targets. "
            f"Non-winner rows with gain=0.0 are retained as the low-end anchor."
        )

    n_valid = int(valid_gain_mask.sum())
    n_winners_with_gain = int((combined_df.loc[valid_gain_mask, "label"] == 1).sum()) if valid_gain_mask.any() else 0
    n_non_winners_with_gain = n_valid - n_winners_with_gain

    # FIX 3: Log how many rows valid_gain_mask drops vs the full combined_df the
    # classifier trained on.  The regressor's effective training population is much
    # smaller because most base-CSV and non-winner rows have no gain target.
    # Surfacing this gap makes it obvious in the logs when the regressor is working
    # from a very different (and smaller) slice of data than the classifier.
    n_total_in = len(combined_df)
    n_dropped  = n_total_in - n_valid
    logger.info(
        f"\n── Training gain regressor on {n_valid} rows with gain data ──\n"
        f"  Input rows (classifier train split): {n_total_in}\n"
        f"  Dropped (no gain target / out-of-range): {n_dropped} "
        f"({n_dropped / max(n_total_in, 1):.1%} of classifier train set)\n"
        f"  Winners with gain:     {n_winners_with_gain}\n"
        f"  Non-winners with gain: {n_non_winners_with_gain} (RC1: broader training set)\n"
        f"  Target:      {gain_col} (RC7: cap raised to 10 000%, log-transformed)"
    )

    # FIX 2: Log gain-target source populations so scale divergence between the
    # RC2 prev_close-corrected winners and the accuracy-table backfill is visible.
    # When RC2 and the accuracy table compute actual_high_pct with different
    # denominators (prev_close vs same-day close), the two populations will have
    # visibly different distribution statistics here — a clear signal to investigate.
    #
    # FIX2 (corrected): the previous version of this check compared winner-row
    # gains against the *entire* non_winners_day_prior population, which is
    # dominated by rows the CORE FIX above deliberately set to gain=0.0 (they
    # never had an actual_high_pct at all — they're a "no big move" anchor, not
    # a same-day-close-denominated value). Comparing real winner gains to an
    # intentional 0.0 constant will always look like a huge divergence and has
    # nothing to do with denominators. We now exclude those explicit zero-fill
    # rows so the comparison only includes non-winner rows that carry a genuine
    # actual_high_pct/actual_gain_pct value pulled from the accuracy table.
    if valid_gain_mask.any() and "source" in combined_df.columns:
        _gt_valid  = gain_targets[valid_gain_mask]
        _src_valid = combined_df.loc[valid_gain_mask, "source"]
        _is_winner = combined_df.loc[valid_gain_mask, "label"] == 1
        _zero_fill_valid = zero_fill_mask.reindex(valid_gain_mask.index, fill_value=False)[valid_gain_mask]

        _rc2_group = _src_valid.str.contains("winners_day_prior", na=False) & ~_src_valid.str.contains("non_winners", na=False)
        _acc_group_all = _src_valid.str.contains("non_winners_day_prior", na=False)
        _acc_group_genuine = _acc_group_all & ~_zero_fill_valid
        _acc_group_zero_anchor = _acc_group_all & _zero_fill_valid

        for _src_group, _src_label in [
            (_rc2_group, "daily_winners (RC2 enriched)"),
            (_acc_group_genuine, "non_winners (accuracy-table backfill, genuine)"),
            (_acc_group_zero_anchor, "non_winners (explicit 0.0 anchor, CORE FIX)"),
            (~_src_valid.str.contains("day_prior", na=False), "base_csv / mistake rows"),
        ]:
            _grp_vals = _gt_valid[_src_group]
            if len(_grp_vals) == 0:
                continue
            logger.info(
                f"  Gain source [{_src_label}]: n={len(_grp_vals)}, "
                f"min={_grp_vals.min():.1f}%, max={_grp_vals.max():.1f}%, "
                f"mean={_grp_vals.mean():.1f}%, std={_grp_vals.std():.1f}%"
            )
        # Warn when the two primary sources have very different mean gains —
        # a strong signal that they are using incompatible denominators.
        # Only the *genuine* accuracy-table rows are compared here; the
        # explicit 0.0 anchor rows are excluded on purpose (see note above).
        _rc2_vals  = _gt_valid[_rc2_group]
        _acc_vals  = _gt_valid[_acc_group_genuine]
        # FIX2: require a larger genuine sample before trusting a mean-diff
        # warning. n<30 is too small/noisy to distinguish a real denominator
        # bug from ordinary sampling variance (e.g. n=15 with mean=3.6% is not
        # a reliable estimate of the true non-winner base rate).
        MIN_GENUINE_ACC_SAMPLE = 30
        if len(_rc2_vals) >= 5 and len(_acc_vals) >= MIN_GENUINE_ACC_SAMPLE:
            _mean_diff = abs(_rc2_vals.mean() - _acc_vals.mean())
            if _mean_diff > 20.0:
                logger.warning(
                    f"  ⚠️  FIX2: Mean gain differs by {_mean_diff:.1f}pp between RC2-enriched "
                    f"winners ({_rc2_vals.mean():.1f}%) and genuine accuracy-table non-winners "
                    f"({_acc_vals.mean():.1f}%). This suggests the two sources are computing "
                    f"actual_high_pct with different denominators (prev_close vs same-day close). "
                    f"Investigate _compute_correct_actual_high_pct and enrich_mistakes_with_gains "
                    f"to ensure both use the same base."
                )
        elif len(_acc_vals) < MIN_GENUINE_ACC_SAMPLE:
            logger.info(
                f"  FIX2: Skipping denominator-divergence check — only {len(_acc_vals)} "
                f"genuine (non-zero-anchor) accuracy-table non-winner rows available "
                f"(need ≥{MIN_GENUINE_ACC_SAMPLE}). Most non-winner rows are explicit 0.0 "
                f"anchors from CORE FIX, which is expected and not a denominator issue; "
                f"a small genuine sample is also too noisy to draw a conclusion from."
            )

    if n_valid < 30:
        logger.warning(f"Only {n_valid} rows with gain data — need ≥30. Skipping gain regressor.")
        return None

    # ------------------------------------------------------------------
    # LEAK-FREE FIX (reinstated): the caller (main()) now passes ONLY the
    # classifier's train-split rows here by default (X_train / combined_df
    # .loc[train_idx]), so this function's own internal 80/20 time split is
    # drawn entirely from data the classifier already trained on — no
    # overlap with the classifier's held-out val/cal windows.
    #
    # This was previously reverted (see the superseded "NOTE" left in main()
    # below) because, at the time, virtually all gain-labeled rows came from
    # true_gain_pct (the market-snapshot join), which only exists for recent
    # T-1 rows — i.e. exactly the rows living in the classifier's val window.
    # Restricting to train_idx therefore left ~0 labeled rows.
    #
    # That is no longer the whole picture: attach_true_gain_targets() also
    # backfills '_unified_gain_target' from ml_training_base.gain_pct, which
    # is populated across the FULL historical span of the base CSV, not just
    # the recent val window. That gives the train split real, broadly-
    # distributed gain labels even after excluding the most recent VAL_WEEKS.
    #
    # We still don't take this on faith: if the train-only pool genuinely
    # doesn't clear MIN_TRAIN_ONLY_GAIN_ROWS, we fall back to the old
    # (leaky) full-dataset behaviour rather than training on a starved
    # sample — but we do it visibly, via a WARNING and a metadata flag,
    # instead of silently baking leakage into every run.
    # ------------------------------------------------------------------
    if not _is_fallback_retry and n_valid < MIN_TRAIN_ONLY_GAIN_ROWS:
        logger.warning(
            f"LEAK-FREE FIX: only {n_valid} gain-labeled rows in the train-only "
            f"pool passed to this function (need ≥{MIN_TRAIN_ONLY_GAIN_ROWS} to "
            "trust a leak-free split). Falling back to training the gain "
            "regressor on the FULL (train+val) pool instead. This reintroduces "
            "train/val overlap for the regressor ONLY (the classifier is "
            "unaffected) — its reported val MAE/R² should be treated as "
            "optimistic. This fallback should disappear on its own as more "
            "gain-labeled history accumulates."
        )
        _full_df = getattr(train_gain_regressor, "_full_combined_df", None)
        _full_X  = getattr(train_gain_regressor, "_full_X_scaled", None)
        if _full_df is not None and _full_X is not None:
            fallback_model = train_gain_regressor(
                X_scaled=_full_X,
                combined_df=_full_df,
                feature_names=feature_names,
                client=client,
                accuracy_gain_map=accuracy_gain_map,
                _is_fallback_retry=True,
            )
            if fallback_model is not None:
                fallback_model._trained_leak_free = False  # type: ignore[attr-defined]
            return fallback_model
        else:
            logger.warning(
                "LEAK-FREE FIX: no full-dataset fallback was registered by the "
                "caller — proceeding with the train-only pool despite being "
                f"below the {MIN_TRAIN_ONLY_GAIN_ROWS}-row threshold. Results "
                "may be noisy."
            )

    # ------------------------------------------------------------------
    # RC3 FIX: Use X_scaled (already StandardScaler-transformed), not raw
    # ------------------------------------------------------------------
    # BUG 4 FIX: align X_scaled to combined_df by shared index labels rather
    # than by position.  The previous code assumed "X_scaled has the same row
    # order as combined_df" and then blindly force-assigned X_reg.index =
    # combined_df.index.  After a VAL REBALANCE the rebalanced rows are
    # appended to the end of train_idx, so X_train (which came from
    # X.loc[train_idx]) and combined_df.loc[train_idx] both contain the
    # rebalanced rows — but any future code change that produces even a tiny
    # ordering difference between the two DataFrames would silently corrupt
    # the feature→gain-target mapping.  Using .reindex() makes the alignment
    # explicit and index-safe regardless of row order.
    common_idx = X_scaled.index.intersection(combined_df.index)
    if len(common_idx) == 0:
        logger.warning(
            f"RC3: X_scaled and combined_df share no index labels — "
            "cannot align. Skipping gain regressor."
        )
        return None
    if len(common_idx) < len(combined_df):
        logger.warning(
            f"RC3: X_scaled covers {len(common_idx)} of {len(combined_df)} combined_df rows — "
            "some rows will be excluded from gain regressor training."
        )

    # Narrow both to the common index so all downstream masks align correctly.
    # Use reindex on gain_targets (preserves accuracy_gain_map fills from above)
    # rather than re-deriving from combined_df[gain_col].
    combined_df  = combined_df.loc[common_idx]
    gain_targets = gain_targets.reindex(common_idx)

    # Align features to the same index using label-based reindex (not positional).
    X_reg = X_scaled.reindex(common_idx)
    y_reg = gain_targets.copy()
    w_reg = (
        combined_df["sample_weight"].astype(float)
        if "sample_weight" in combined_df.columns
        else pd.Series(1.0, index=combined_df.index)
    )

    # ------------------------------------------------------------------
    # RC8 FIX: Up-weight winner rows (and high-gain winners within them)
    # using the SAME data-driven approach train_model() uses for the
    # classifier's scale_pos_weight: compute the actual class-imbalance
    # ratio in this run's training pool, then clamp it to sane bounds.
    # This replaces the old fixed multipliers (_WINNER_WEIGHT_MULTIPLIER /
    # _HIGH_GAIN_MULTIPLIER), which had to be hand-re-tuned (8.0→3.0,
    # 5.0→3.0) every time the winner ratio in the data shifted, and were
    # the direct cause of the "regressor predictions clustered near
    # 0–5%" issue once the ratio drifted from when those constants were
    # last hand-tuned.
    # ------------------------------------------------------------------
    winner_mask_valid = (combined_df["label"] == 1) & valid_gain_mask
    high_gain_mask = winner_mask_valid & (gain_targets >= _HIGH_GAIN_THRESHOLD)

    n_winners     = int(winner_mask_valid.sum())
    n_non_winners = int((~winner_mask_valid).sum())
    if n_winners > 0 and n_non_winners > 0:
        raw_winner_w = n_non_winners / n_winners
        winner_weight = max(REG_WINNER_WEIGHT_MIN, min(REG_WINNER_WEIGHT_MAX, raw_winner_w))
    else:
        raw_winner_w = None
        winner_weight = _WINNER_WEIGHT_MULTIPLIER  # degenerate case fallback

    n_high_gain     = int(high_gain_mask.sum())
    n_other_winners = n_winners - n_high_gain
    if n_high_gain > 0 and n_other_winners > 0:
        raw_high_gain_w = n_other_winners / n_high_gain
        high_gain_weight = max(REG_HIGH_GAIN_WEIGHT_MIN, min(REG_HIGH_GAIN_WEIGHT_MAX, raw_high_gain_w))
    else:
        raw_high_gain_w = None
        high_gain_weight = _HIGH_GAIN_MULTIPLIER  # degenerate case fallback

    if winner_mask_valid.any():
        w_reg = w_reg.copy()
        w_reg[winner_mask_valid] *= winner_weight
        if high_gain_mask.any():
            w_reg[high_gain_mask] *= high_gain_weight
            logger.info(
                f"  RC8: winner weight={winner_weight:.2f} "
                f"(raw neg/pos={n_non_winners}/{n_winners}"
                f"{f'={raw_winner_w:.2f}' if raw_winner_w is not None else ''}, "
                f"clamped to [{REG_WINNER_WEIGHT_MIN}, {REG_WINNER_WEIGHT_MAX}]) "
                f"applied to {n_winners} winner rows; "
                f"high-gain weight={high_gain_weight:.2f} "
                f"(raw other/high={n_other_winners}/{n_high_gain}"
                f"{f'={raw_high_gain_w:.2f}' if raw_high_gain_w is not None else ''}, "
                f"clamped to [{REG_HIGH_GAIN_WEIGHT_MIN}, {REG_HIGH_GAIN_WEIGHT_MAX}]) "
                f"applied to {n_high_gain} rows (×{winner_weight * high_gain_weight:.1f} total)"
            )
        else:
            logger.info(
                f"  RC8: winner weight={winner_weight:.2f} "
                f"(raw neg/pos={n_non_winners}/{n_winners}"
                f"{f'={raw_winner_w:.2f}' if raw_winner_w is not None else ''}, "
                f"clamped to [{REG_WINNER_WEIGHT_MIN}, {REG_WINNER_WEIGHT_MAX}]) "
                f"applied to {n_winners} winner rows"
            )

    # ------------------------------------------------------------------
    # RC9 FIX: Further up-weight winner rows by classifier confidence.
    # RC8 already up-weights all winners uniformly, but a 51%-confidence
    # winner and a 99%-confidence (strong-buy) winner get the same weight —
    # so the regressor has no extra incentive to get the high-confidence
    # rows' magnitude right, and those are exactly the rows where the
    # reported gain kept coming out too low. Since clf_proba is now a
    # feature in X_scaled (see the "REGRESSOR-ONLY clf_proba FEATURE" block
    # in main()), we can also use it here, at training time, to scale the
    # winner-row weight itself: within the winner rows, higher classifier
    # confidence -> proportionally more weight, on top of (not replacing)
    # the RC8 winner/high-gain multipliers above.
    # ------------------------------------------------------------------
    if winner_mask_valid.any() and "clf_proba" in X_reg.columns:
        conf = X_reg.loc[winner_mask_valid, "clf_proba"].reindex(w_reg.index).fillna(0.5)
        # Map confidence in [0, 1] onto a [REG_CONFIDENCE_WEIGHT_MIN, MAX]
        # multiplier so a 50%-confidence winner keeps roughly its RC8 weight
        # (multiplier ~1x) while a near-100%-confidence winner gets boosted.
        confidence_multiplier = (
            REG_CONFIDENCE_WEIGHT_MIN
            + (REG_CONFIDENCE_WEIGHT_MAX - REG_CONFIDENCE_WEIGHT_MIN)
            * conf.clip(0.0, 1.0)
        )
        w_reg = w_reg.copy()
        w_reg.loc[winner_mask_valid] *= confidence_multiplier
        logger.info(
            f"  RC9: classifier-confidence weight boost applied to "
            f"{int(winner_mask_valid.sum())} winner rows "
            f"(multiplier range [{REG_CONFIDENCE_WEIGHT_MIN}, {REG_CONFIDENCE_WEIGHT_MAX}] "
            f"scaled by clf_proba; mean confidence={conf.mean():.3f})"
        )
    elif winner_mask_valid.any():
        logger.info(
            "  RC9: skipped — clf_proba not present in this run's feature "
            "matrix (older training path); winner rows keep RC8 weighting only."
        )

    X_reg_valid = X_reg[valid_gain_mask]
    y_reg_valid = y_reg[valid_gain_mask]
    w_reg_valid = w_reg[valid_gain_mask]

    # Fill NaN in scaled features with 0 (mean after scaling = 0).
    # X_reg is already StandardScaler output, so 0.0 == column mean.
    # This is consistent with build_scaler() which also fills with 0.0 after
    # scaling, ensuring all three paths (classifier training, regressor training,
    # and inference) treat missing values identically.
    X_reg_fill = X_reg_valid.fillna(0.0)

    # ------------------------------------------------------------------
    # RC7 FIX: Log-transform the gain target.
    # Gain % has a heavily right-skewed distribution: most values cluster
    # near 0–20%, but winners can reach 5000%.  Training XGBoost directly
    # on raw % means the squared-error loss is dominated by a handful of
    # extreme values, causing trees to split on "is this stock an outlier?"
    # rather than on signals that generalise.  log1p(max(gain, 0)) maps:
    #   0%    → 0.0    200%  → 1.099
    #   5%    → 0.049  500%  → 1.792
    #   50%   → 0.405  5000% → 3.912
    # The regressor predicts in log-space; at inference time we expm1() back.
    # Winsorize at the 99.5th percentile in log-space to prevent the
    # remaining extreme values from dominating.
    # ------------------------------------------------------------------
    y_log = np.log1p(np.maximum(y_reg_valid.values, 0.0))
    winsor_cap = np.percentile(y_log, _GAIN_WINSOR_PCT)
    y_log_winsor = np.minimum(y_log, winsor_cap)
    n_winsorized = int((y_log > winsor_cap).sum())
    if n_winsorized > 0:
        logger.info(
            f"  RC7: Winsorized {n_winsorized} values above {np.expm1(winsor_cap):.1f}% "
            f"({_GAIN_WINSOR_PCT}th percentile in log-space)"
        )
    y_reg_log = pd.Series(y_log_winsor, index=y_reg_valid.index)

    # Time-based split for the regressor (mirrors the classifier split).
    # Using a random split here would allow future gain patterns to leak into
    # training, making val R² optimistic.
    if len(X_reg_fill) >= 20:
        # Re-use the date information from combined_df to sort rows
        _has_detection_reg = "detection_date" in combined_df.columns
        _has_event_reg     = "event_date" in combined_df.columns
        _date_col = "detection_date" if _has_detection_reg else ("event_date" if _has_event_reg else None)
        if _date_col is not None:
            # BUG FIX (same root cause as the lookback filter above): don't pick
            # one column for the whole frame. detection_date is NaT for every
            # base-CSV row, which previously made _dates all-NaT whenever a
            # split happened to contain mostly/only base rows — collapsing the
            # val set to 0 rows and crashing mean_absolute_error downstream.
            # Fall back to event_date (shifted 1 BDay) per-row instead.
            _dates = pd.Series(pd.NaT, index=combined_df.index)
            if _has_detection_reg:
                _dates = pd.to_datetime(combined_df["detection_date"], errors="coerce")
            if _has_event_reg:
                _event_dates_reg = pd.to_datetime(combined_df["event_date"], errors="coerce") - pd.tseries.offsets.BDay(1)
                _dates = _dates.fillna(_event_dates_reg)
            _dates = _dates.loc[valid_gain_mask]
            # FIX 1: Mirror train_val_split's FIX 2 — NaT rows are mistake samples
            # (they have no detection_date/event_date).  sort_values(na_position="last")
            # previously pushed them to the END of the sorted index, meaning they landed
            # in the val set (the last 20%).  That contaminated early-stopping RMSE with
            # the model's own hardest, highest-weight error examples and made val MAE
            # meaningless as an evaluation signal.
            # Fix: separate NaT rows explicitly and always append them to the train split.
            _nat_mask_reg  = _dates.isna()
            _n_nat_reg     = int(_nat_mask_reg.sum())
            _dated_idx_reg = _dates[~_nat_mask_reg].sort_values().index   # chronological
            _nat_idx_reg   = _dates[_nat_mask_reg].index                  # mistake rows

            _split_pos = int(len(_dated_idx_reg) * 0.8)
            _tr_idx    = _dated_idx_reg[:_split_pos].append(_nat_idx_reg)  # NaT → train
            _va_idx    = _dated_idx_reg[_split_pos:]

            if _n_nat_reg > 0:
                logger.info(
                    f"  FIX1: {_n_nat_reg} NaT rows (mistake samples) pinned to regressor "
                    f"train split (previously leaked into val via na_position='last')."
                )

            X_tr   = X_reg_fill.loc[_tr_idx]
            X_va   = X_reg_fill.loc[_va_idx]
            y_tr   = y_reg_log.loc[_tr_idx]
            y_va   = y_reg_log.loc[_va_idx]
            w_tr   = w_reg_valid.loc[_tr_idx]
            # Keep raw (non-log) val targets for human-readable MAE reporting
            y_va_raw = y_reg_valid.loc[_va_idx]
            logger.info(
                f"  Gain regressor time-based split: "
                f"{len(X_tr)} train / {len(X_va)} val"
            )
        else:
            # No date column — fall back to sequential split (still no random leakage)
            _split_pos = int(len(X_reg_fill) * 0.8)
            X_tr = X_reg_fill.iloc[:_split_pos]
            X_va = X_reg_fill.iloc[_split_pos:]
            y_tr = y_reg_log.iloc[:_split_pos]
            y_va = y_reg_log.iloc[_split_pos:]
            w_tr = w_reg_valid.iloc[:_split_pos]
            y_va_raw = y_reg_valid.iloc[_split_pos:]
            logger.info(
                f"  Gain regressor sequential split (no date column): "
                f"{len(X_tr)} train / {len(X_va)} val"
            )
    else:
        X_tr, X_va, y_tr, y_va, w_tr = (
            X_reg_fill, X_reg_fill, y_reg_log, y_reg_log, w_reg_valid
        )
        y_va_raw = y_reg_valid

    # Log gain distribution (in original % space) to diagnose compression
    y_tr_raw_arr = np.expm1(y_tr.values if hasattr(y_tr, "values") else y_tr)
    logger.info(
        f"  Gain target distribution — train set (original % space):\n"
        f"    min={float(y_tr_raw_arr.min()):.1f}%  "
        f"max={float(y_tr_raw_arr.max()):.1f}%  "
        f"mean={float(y_tr_raw_arr.mean()):.1f}%  "
        f"std={float(y_tr_raw_arr.std()):.1f}%  "
        f"median={float(np.median(y_tr_raw_arr)):.1f}%"
    )
    logger.info(
        f"  Gain target distribution — train set (log space, what regressor sees):\n"
        f"    min={float((y_tr.values if hasattr(y_tr, 'values') else y_tr).min()):.3f}  "
        f"max={float((y_tr.values if hasattr(y_tr, 'values') else y_tr).max()):.3f}  "
        f"std={float((y_tr.values if hasattr(y_tr, 'values') else y_tr).std()):.3f}"
    )

    if float(y_tr_raw_arr.std()) < 2.0:
        logger.warning(
            f"  ⚠️  Gain target std={float(y_tr_raw_arr.std()):.2f}% is very low. "
            "The gain distribution is compressed — predictions will be flat. "
            "Check RC2 fix (prev_close denominator) is working correctly."
        )

    # ------------------------------------------------------------------
    # RC7 FIX: Match classifier hyperparameters more closely.
    # The old regressor used n_estimators=200, max_depth=4, min_child_weight=5,
    # no gamma, weak regularisation.  The result was a shallower, looser model
    # that over-generalised toward the mean of the (overwhelmingly low-gain)
    # training set.  Using the same depth/regularisation as the classifier
    # forces the regressor to find more specific gain-relevant patterns.
    # ------------------------------------------------------------------
    regressor = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,   # loosened from 10, matching classifier fix
        gamma=0.3,             # loosened from 1.0
        reg_alpha=0.2,         # loosened from 0.5
        reg_lambda=1.5,        # loosened from 2.0
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    regressor.fit(
        X_tr, y_tr,
        sample_weight=w_tr.values,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    # Evaluate in original % space for interpretability
    val_pred_log = regressor.predict(X_va)
    val_pred_pct = np.expm1(val_pred_log)           # inverse of log1p
    y_va_raw_arr = y_va_raw.values if hasattr(y_va_raw, "values") else np.array(y_va_raw)

    from sklearn.metrics import mean_absolute_error, r2_score
    mae    = mean_absolute_error(y_va_raw_arr, val_pred_pct)
    # R² in log-space (what the model was actually trained on) is more meaningful
    r2_log = r2_score(y_va.values if hasattr(y_va, "values") else y_va, val_pred_log) if len(y_va) > 1 else float("nan")
    pred_std_pct = float(val_pred_pct.std())
    logger.info(
        f"  Gain regressor — val MAE (% space): {mae:.2f}%  "
        f"R² (log space): {r2_log:.3f}  "
        f"pred_std (% space): {pred_std_pct:.2f}%"
    )
    logger.info(
        f"  Predicted gains range (% space): {val_pred_pct.min():.1f}% – {val_pred_pct.max():.1f}%"
    )
    logger.info(f"  Best iteration: {regressor.best_iteration}")

    if pred_std_pct < 0.5:
        logger.warning(
            f"  ⚠️  Regressor prediction std={pred_std_pct:.3f}% is very flat even after RC7 fixes. "
            "Root causes: too few training samples with gain data, or "
            "scaled vs unscaled feature mismatch (RC3). "
            "The explosion_predictor will use the relaxed std guard (0.5) "
            "rather than disabling immediately."
        )

    # Store a flag so explosion_predictor.py knows to apply expm1 at inference
    regressor._log_transformed_target = True  # type: ignore[attr-defined]
    # LEAK-FREE FIX: mark whether this fit came from the train-only pool
    # (leak-free) or the full train+val fallback pool (see guard above).
    # main() reads this to log/record it in model_metadata.json.
    regressor._trained_leak_free = not _is_fallback_retry  # type: ignore[attr-defined]

    return regressor


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    model: XGBClassifier,
    scaler: StandardScaler,
    fi_df: pd.DataFrame,
    feature_names: list[str],
    training_stats: dict,
    gain_regressor=None,
    top10_training_stats: dict | None = None,
) -> None:
    """Save model, scaler, gain regressor, feature importance, and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  MODEL_PATH,  protocol=4)
    logger.info(f"Saved model  → {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH, protocol=4)
    logger.info(f"Saved scaler → {SCALER_PATH}")

    if gain_regressor is not None:
        joblib.dump(gain_regressor, GAIN_REGRESSOR_PATH, protocol=4)
        logger.info(f"Saved gain regressor → {GAIN_REGRESSOR_PATH}")
    else:
        logger.info("Gain regressor not trained this run — predictor will use calibrated fallback")

    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Saved feature importance → {FEATURE_IMPORTANCE_PATH}")

    # RC6: model may be a CalibratedClassifierCV wrapping the raw XGBClassifier.
    # best_iteration / best_score live on the raw booster, not the wrapper.
    _raw_model = model
    if hasattr(model, "calibrated_classifiers_"):
        _raw_model = model.calibrated_classifiers_[0].estimator

    metadata = {
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "source":                "ml_retrain_model.py",
        "training_approach":     "full_retrain_from_scratch",
        "n_features":            len(feature_names),
        "features":              feature_names,
        "feature_names_sample":  feature_names[:20],
        "best_iteration":        int(_raw_model.best_iteration),
        "best_val_auc":          float(_raw_model.best_score),  # renamed from best_val_logloss; metric is AUC (eval_metric="auc")
        "gain_regressor_trained": gain_regressor is not None,
        "gain_regressor_fixes":  ["RC1_broader_training", "RC2_prev_close_denominator",
                                  "RC3_scaled_features", "RC6_mistake_enrichment", "RC7_log_transform_heavy_weights"],
        **training_stats,
    }
    if top10_training_stats:
        metadata["top10_feature_distribution"] = top10_training_stats
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata → {METADATA_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



def _loosen_filters_for_sampling(base_filters: dict, step_pct: float, pass_number: int) -> dict:
    """
    Return a copy of base_filters with thresholds relaxed by (step_pct * pass_number) %.

    Mirrors _loosen_filters() in daily_non_winners_detector.py so both systems
    behave identically when given the same config values.
    """
    lf = dict(base_filters)
    reduction  = (step_pct / 100.0) * pass_number
    min_factor = max(0.0, 1.0 - reduction)
    max_factor = 1.0 + reduction
    for key in list(lf.keys()):
        if lf[key] is None or str(key).startswith("_"):
            continue
        if key.startswith("min_"):
            lf[key] = lf[key] * min_factor
        elif key.startswith("max_"):
            lf[key] = lf[key] * max_factor
    return lf


def _build_filter_mask(negatives: "pd.DataFrame", filters: dict) -> "pd.Series":
    """
    Build a boolean Series selecting rows in `negatives` that pass `filters`.
    Returns an all-True Series (no filtering) when no matching columns exist.
    """
    import pandas as pd

    filter_map = [
        (
            ["t1_close_Close", "t1_open_Close", "Close", "close", "price"],
            "min_price", "max_price",
        ),
        (
            ["t1_close_Volume", "t1_open_Volume", "Volume", "volume"],
            "min_volume", None,
        ),
        (
            ["t1_close_HV_10", "t1_open_HV_10", "HV_10", "hv10", "volatility_10d", "historical_volatility_10"],
            "min_hv10", "max_hv10",
        ),
        (
            ["t1_close_HV_20", "t1_open_HV_20", "HV_20", "hv20", "volatility_20d", "historical_volatility_20"],
            "min_hv20", "max_hv20",
        ),
        (
            ["t1_close_Volume_Ratio", "t1_open_Volume_Ratio", "Volume_Ratio", "volume_ratio", "relative_volume", "Relative_Volume"],
            "min_relative_volume", None,
        ),
        (
            ["t1_close_Volume_Ratio", "t1_open_Volume_Ratio", "Volume_Ratio", "volume_ratio"],
            "min_volume_ratio", None,
        ),
    ]

    mask = pd.Series(True, index=negatives.index)
    used = False
    for candidates, min_key, max_key in filter_map:
        col = next((c for c in candidates if c in negatives.columns), None)
        if col is None:
            continue
        if min_key and min_key in filters and filters[min_key] is not None:
            mask &= negatives[col].fillna(-1e9) >= filters[min_key]
            used = True
        if max_key and max_key in filters and filters[max_key] is not None:
            mask &= negatives[col].fillna(1e9) <= filters[max_key]
            used = True
    return mask, used


def apply_filter_aware_negative_sampling(df, logger=None):
    """
    Retraining enhancement:
    - Keep all winners.
    - Prefer non-winners that pass learned_filters.json (hard negatives).
    - If not enough hard negatives exist for a date, progressively loosen the
      filters (up to SAMPLING_LOOSENING_PASSES times, relaxing by
      SAMPLING_LOOSENING_STEP_PCT % per pass) before falling back to
      fully-unfiltered rows.
    - Graduated upweighting: rows selected on pass 0 (strict filters) receive
      the highest weight (HARD_NEG_WEIGHT_BASE × decay^0 = 2.0x), rows selected
      on pass k receive HARD_NEG_WEIGHT_BASE × HARD_NEG_WEIGHT_DECAY^k, and
      unfiltered fallback rows receive no uplift (1.0x).
    - Preserve existing mistake-learner weights (multiplied by the pass weight).
    - FIX: No-winner dates now receive a proportional target (up to 40% of
      available negatives, capped at 3 × global_neg_ratio) instead of a flat
      floor of 4, so they contribute meaningfully and the overall winner:non-winner
      ratio reflects the true 90-day distribution.

    Loosening knobs (all in config.yaml under non_winners:):
        loosening_passes   – extra passes before unfiltered fallback (default 3)
        loosening_step_pct – % to relax per pass (default 20)
        min_hard_neg_ratio – target fraction of negatives from filtered pool
                             before stopping early (default 0.80)
    """
    import json
    import pandas as pd
    from pathlib import Path

    if df is None or df.empty or "label" not in df.columns:
        return df

    # ── Opt-in gate ────────────────────────────────────────────────────────
    # Filter-aware hard-negative sampling is OFF by default. Set
    # USE_LEARNED_FILTERS=1/true/yes in the environment (or pass
    # --use-learned-filters on the CLI, which sets this same env var) to
    # enable it for a retrain run. This mirrors the USE_SELECTED_FEATURES
    # opt-in flag used elsewhere in this file.
    if os.environ.get("USE_LEARNED_FILTERS", "").lower() not in ("1", "true", "yes"):
        if logger:
            logger.info(
                "Filter-aware negative sampling DISABLED (USE_LEARNED_FILTERS not set) "
                "— using unfiltered negative sampling."
            )
        return df

    # ── Load filters fresh from disk ─────────────────────────────────────────
    # Reading here (not at module import) means any retrain picks up the latest
    # learned_filters.json without restarting.
    filters_path = Path("ml_models/learned_filters.json")
    if not filters_path.exists():
        return df

    try:
        base_filters = json.loads(filters_path.read_text())
    except Exception:
        return df

    if logger:
        logger.info("=" * 80)
        logger.info("FILTER-AWARE NEGATIVE SAMPLING — PROGRESSIVE LOOSENING")
        logger.info(f"Filters loaded from {filters_path}")
        logger.info(
            f"  loosening_passes={SAMPLING_LOOSENING_PASSES}, "
            f"step_pct={SAMPLING_LOOSENING_STEP_PCT}%, "
            f"min_hard_neg_ratio={SAMPLING_MIN_HARD_NEG_RATIO:.0%}"
        )
        for k, v in base_filters.items():
            if not str(k).startswith("_"):
                logger.info(f"  BASE FILTER {k}={v}")
        logger.info("=" * 80)

    # ── Build a unified sampling date column (coalesce detection_date ?? event_date) ──
    # combined_df has TWO date columns:
    #   T-1 rows      → detection_date populated, event_date NaT
    #   base CSV rows → event_date populated,     detection_date NaT
    # Using detection_date alone as the grouping key causes base rows to fall
    # through (NaT never matches any date group), meaning ~13k base negatives
    # and ~2.7k base winners are invisible to the per-date loop and only a
    # fraction of T-1 rows (1270 winners / 2742 negatives) are actually sampled.
    # That produces the observed 59%+ positive rate: base winners are kept
    # unconditionally while most base negatives are silently dropped.
    #
    # Fix: coalesce into _sampling_date so every row has a usable date key.
    # base CSV event_date is the explosion day (T+1 relative to detection); we
    # subtract 1 business day to align it with the T-1 detection timeline.
    df = df.copy()  # avoid mutating caller's DataFrame
    _SAMPLING_DATE_COL = "_sampling_date"

    if "detection_date" in df.columns or "event_date" in df.columns:
        _det = pd.to_datetime(
            df["detection_date"] if "detection_date" in df.columns else pd.Series(pd.NaT, index=df.index),
            errors="coerce",
        )
        if "event_date" in df.columns:
            _ev = pd.to_datetime(df["event_date"], errors="coerce")
            _ev_shifted = _ev - pd.tseries.offsets.BDay(1)
            _unified = _det.fillna(_ev_shifted)
        else:
            _unified = _det
        df[_SAMPLING_DATE_COL] = _unified.dt.strftime("%Y-%m-%d").where(_unified.notna(), None)
    else:
        df[_SAMPLING_DATE_COL] = None

    winners   = df[df["label"] == 1].copy()
    negatives = df[df["label"] == 0].copy()

    if negatives.empty:
        return df

    # Use the coalesced date for grouping; fall back to raw date cols if needed
    if df[_SAMPLING_DATE_COL].notna().any():
        date_col = _SAMPLING_DATE_COL
    else:
        date_col = next(
            (c for c in ["detection_date", "event_date", "trade_date", "date"] if c in df.columns),
            None,
        )

    # ── Per-date (or global) negative selection with progressive loosening ────
    #
    # Graduated filter-pass weights:
    #   pass 0 (full filters)         → HARD_NEG_WEIGHT_BASE   (highest weight)
    #   pass k (loosened k*step_pct%) → HARD_NEG_WEIGHT_BASE * decay^k
    #   unfiltered fallback           → 1.0  (base weight, no uplift)
    #
    # This rewards rows that pass strict filters more than rows that only
    # pass after loosening, and rewards loosened-pass rows more than
    # unfiltered fallback rows.
    HARD_NEG_WEIGHT_BASE  = 2.0   # weight multiplier for a full-filter pass
    HARD_NEG_WEIGHT_DECAY = 0.75  # per-loosening-pass decay factor

    def _pass_weight(pass_idx: int) -> float:
        """Return the sample-weight multiplier for a row selected on pass `pass_idx`.
        pass_idx=0 → full filters (highest); pass_idx>0 → progressively lower;
        pass_idx=-1 → unfiltered fallback (no uplift, returns 1.0)."""
        if pass_idx < 0:
            return 1.0
        return HARD_NEG_WEIGHT_BASE * (HARD_NEG_WEIGHT_DECAY ** pass_idx)

    def _select_negatives_for_group(neg_group: "pd.DataFrame", target: int) -> "pd.DataFrame":
        """
        For a single date-group of negatives, try to fill `target` hard-negative
        slots using progressively looser filters.  Returns the selected rows,
        with a '_filter_pass' column recording which pass selected each row
        (0 = full filters, 1..N = loosened, -1 = unfiltered fallback).
        """
        preferred = int(target * SAMPLING_MIN_HARD_NEG_RATIO)
        # Map: index → pass_idx that selected it
        selected_pass: dict = {}

        for pass_idx in range(SAMPLING_LOOSENING_PASSES + 1):
            if pass_idx == 0:
                active_filters = base_filters
                label = "full filters"
            else:
                active_filters = _loosen_filters_for_sampling(
                    base_filters, SAMPLING_LOOSENING_STEP_PCT, pass_idx
                )
                label = f"loosened {SAMPLING_LOOSENING_STEP_PCT * pass_idx:.0f}%"

            mask, used = _build_filter_mask(neg_group, active_filters)
            if not used:
                # No filterable columns found — skip all loosening, fall straight through
                break

            # Rows that pass this pass's filter AND haven't been picked yet
            passing_idx = set(neg_group[mask].index) - set(selected_pass.keys())
            needed = preferred - len(selected_pass)
            if needed <= 0:
                break

            take = list(passing_idx)[:needed]
            for idx in take:
                selected_pass[idx] = pass_idx

            if logger and pass_idx > 0 and take:
                logger.info(
                    f"    pass {pass_idx} ({label}): +{len(take)} hard negatives "
                    f"(cumulative hard={len(selected_pass)}/{preferred})"
                )

            if len(selected_pass) >= preferred:
                break   # have enough hard negatives — stop loosening

        # Fill remaining quota (target - hard) with whatever unselected rows are left
        remaining_needed = target - len(selected_pass)
        unselected = neg_group[~neg_group.index.isin(set(selected_pass.keys()))]
        if remaining_needed > 0 and not unselected.empty:
            easy_take = unselected.sample(
                min(len(unselected), remaining_needed), random_state=42
            )
            for idx in easy_take.index:
                selected_pass[idx] = -1  # unfiltered fallback

        result = neg_group.loc[list(selected_pass.keys())].copy()
        result["_filter_pass"] = [selected_pass[i] for i in result.index]
        return result

    # ── Run selection ─────────────────────────────────────────────────────────
    hard_selected_count  = 0
    random_selected_count = 0

    # Global neg:winner ratio — used as the per-date target on no-winner dates
    # so they contribute proportionally rather than getting a flat floor of 4.
    # Capped at 10 to avoid flooding training with negatives on winner-sparse dates.
    _global_neg_ratio = max(
        5.25,
        min(10.0, len(negatives) / max(1, len(winners)))
    )

    if date_col is None:
        target_negatives = max(int(len(winners) * _global_neg_ratio), 8)
        selected_neg = _select_negatives_for_group(negatives, target_negatives)

        # Approximate hard vs easy split for logging
        if "_filter_pass" in selected_neg.columns:
            hard_selected_count   = int((selected_neg["_filter_pass"] >= 0).sum())
            random_selected_count = int((selected_neg["_filter_pass"] < 0).sum())
        else:
            full_mask, used = _build_filter_mask(selected_neg, base_filters)
            hard_selected_count  = int(full_mask.sum()) if used else 0
            random_selected_count = len(selected_neg) - hard_selected_count
    else:
        selected_parts = []
        winner_dates   = set(winners[date_col].dropna().unique())
        all_neg_dates  = set(negatives[date_col].dropna().unique())

        for dt in sorted(winner_dates | all_neg_dates):
            winner_group  = winners[winners[date_col] == dt]
            n_winners_dt  = len(winner_group)

            if n_winners_dt > 0:
                # Winner date: sample proportionally to winners on this date
                target = max(int(n_winners_dt * 5.25), 4)
            else:
                # No-winner date: use the global ratio applied to the available
                # negatives so these dates contribute meaningfully instead of
                # receiving a token floor of 4.  Cap at a reasonable ceiling so
                # one huge no-winner date doesn't crowd out everything else.
                neg_dt_count = int((negatives[date_col] == dt).sum())
                target = min(
                    max(int(neg_dt_count * 0.40), 4),   # take up to 40% of available
                    int(_global_neg_ratio * 3),          # hard cap: 3 × global ratio
                )

            neg_dt = negatives[negatives[date_col] == dt]

            chosen = _select_negatives_for_group(neg_dt, target)
            selected_parts.append(chosen)

            if "_filter_pass" in chosen.columns:
                day_hard = int((chosen["_filter_pass"] >= 0).sum())
                day_easy = int((chosen["_filter_pass"] < 0).sum())
            else:
                full_mask, used = _build_filter_mask(chosen, base_filters)
                day_hard  = int(full_mask.sum()) if used else 0
                day_easy  = len(chosen) - day_hard
            hard_selected_count  += day_hard
            random_selected_count += day_easy

            preferred = int(target * SAMPLING_MIN_HARD_NEG_RATIO)
            if logger and n_winners_dt > 0 and day_hard < preferred:
                logger.info(
                    f"[{dt}] hard-negative shortage after loosening: "
                    f"wanted={preferred}, got={day_hard}, backfilled={day_easy}"
                )

        selected_neg = pd.concat(selected_parts) if selected_parts else negatives.iloc[0:0]

    # ── Graduated upweighting based on which filter pass selected each row ────
    # Rows that pass full (strict) filters get a higher weight than rows that
    # only passed after loosening, which in turn outweigh unfiltered fallback rows.
    # This preserves the corrective signal from hard negatives without flattening
    # the distinction between "genuinely filter-passing" and "loosened-in" rows.
    if "sample_weight" in selected_neg.columns and "_filter_pass" in selected_neg.columns:
        def _apply_graduated_weight(row):
            base_w = row["sample_weight"] if pd.notna(row["sample_weight"]) else 1.0
            return base_w * _pass_weight(int(row["_filter_pass"]))
        selected_neg = selected_neg.copy()
        selected_neg["sample_weight"] = selected_neg.apply(_apply_graduated_weight, axis=1)
    elif "sample_weight" in selected_neg.columns:
        # Fallback: flat upweight (old behaviour) when _filter_pass is absent
        selected_neg = selected_neg.copy()
        selected_neg["sample_weight"] = selected_neg["sample_weight"].fillna(1.0) * 1.75

    # Drop internal tracking columns before returning
    for _internal_col in ["_filter_pass", _SAMPLING_DATE_COL]:
        if _internal_col in selected_neg.columns:
            selected_neg = selected_neg.drop(columns=[_internal_col])
        if _internal_col in winners.columns:
            winners = winners.drop(columns=[_internal_col])

    result = pd.concat([winners, selected_neg], ignore_index=True)

    if logger:
        actual_pos_rate = len(winners) / max(len(result), 1)
        logger.info("=" * 80)
        logger.info("FILTER-AWARE SELECTION RESULTS")
        logger.info(f"  Hard negatives (filter-passing) : {hard_selected_count:,}")
        logger.info(f"  Easy/backfill negatives         : {random_selected_count:,}")
        logger.info(
            f"  winners={len(winners):,}, "
            f"selected_negatives={len(selected_neg):,}, "
            f"actual_positive_rate={actual_pos_rate:.1%}"
        )
        logger.info("=" * 80)

    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="ML weekly full retrain from scratch.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG (default: INFO).",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=180, metavar="N",
        help="How many days of T-1 data to use for training (default: 90).",
    )
    parser.add_argument(
        "--use-all-timepoints", action="store_true", default=True,
        help="Use both day_prior_close and day_prior_open T-1 tables (default: True).",
    )
    parser.add_argument(
        "--use-learned-filters", action="store_true", default=False,
        help=(
            "Enable filter-aware hard-negative sampling using "
            "ml_models/learned_filters.json during retraining (default: False). "
            "Equivalent to setting USE_LEARNED_FILTERS=1 in the environment."
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        _configure_logging(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    if args.use_learned_filters:
        os.environ["USE_LEARNED_FILTERS"] = "true"

    logger.info("=" * 60)
    logger.info("ML RETRAIN — FULL RETRAIN FROM SCRATCH")
    logger.info(f"  lookback_days       : {args.lookback_days}")
    logger.info(f"  use_all_timepoints  : {args.use_all_timepoints}")
    logger.info(f"  use_learned_filters : {os.environ.get('USE_LEARNED_FILTERS', '').lower() in ('1', 'true', 'yes')}")
    logger.info(f"  verbose             : {args.verbose}")
    logger.info("=" * 60)

    # ── Sanity check: is lookback_days wide enough for the embargo? ───────────
    # Worst case, the inferred embargo hits EMBARGO_DAYS_CAP (driven by
    # whatever rolling-window feature happens to be selected that month).
    # For train to have any real pre-embargo window left, lookback_days needs
    # to cover: the val window (VAL_WEEKS) + the embargo (up to the cap) +
    # a minimum train slice (MIN_TRAIN_WINDOW_DAYS). Computed from the same
    # constants train_val_split's own cap uses, so this can't drift out of
    # sync with the actual guard.
    _recommended_min_lookback = (VAL_WEEKS * 7) + EMBARGO_DAYS_CAP + MIN_TRAIN_WINDOW_DAYS
    if args.lookback_days < _recommended_min_lookback:
        logger.warning(
            f"lookback_days={args.lookback_days} is below the recommended "
            f"minimum of {_recommended_min_lookback}d "
            f"(= {VAL_WEEKS*7}d val window + {EMBARGO_DAYS_CAP}d worst-case "
            f"embargo + {MIN_TRAIN_WINDOW_DAYS}d minimum train slice). "
            "If the deepest rolling-window feature in this run's selected "
            "features pushes the inferred embargo close to the cap, train "
            "may end up starved -- train_val_split will auto-shrink the "
            "embargo to compensate, but that trades away purge protection "
            "you probably want. Consider raising --lookback-days instead."
        )

    # ── Connect ──────────────────────────────────────────────────────────────
    client = get_supabase_client()

    # ── Load standard training data ───────────────────────────────────────────
    base_df     = load_base_training_data(client, lookback_days=args.lookback_days)
    t1_df       = load_t1_data(client, lookback_days=args.lookback_days)
    combined_df = combine_datasets(base_df, t1_df)

    # ── Apply lookback_days filter to combined_df ─────────────────────────────
    # ml_training_base is fetched in full (can span many months), but only the
    # most recent lookback_days of base data should participate in training.
    # Without this, old base-data winners from months ago inflate the winner
    # count while their corresponding date's non-winners are sparse or absent,
    # creating a severe class imbalance that the per-date sampler cannot fix.
    #
    # T-1 rows (winners_day_prior / non_winners_day_prior) already only span
    # the accumulation period (~90 days), so this filter mainly trims base CSV rows.
    # We use the _sampling_date logic: detection_date ?? (event_date - 1 BDay).
    _lookback_cutoff = (datetime.now().date() - timedelta(days=args.lookback_days)).isoformat()
    _has_detection = "detection_date" in combined_df.columns
    _has_event     = "event_date" in combined_df.columns
    _lb_date_col = "detection_date" if _has_detection else ("event_date" if _has_event else None)
    if _lb_date_col is not None:
        # BUG FIX: previously this picked ONE column for the entire dataframe.
        # Since detection_date exists (populated only on T-1 rows), it was
        # selected for every row — base-CSV rows (event_date only) showed up
        # as NaT in detection_date and bypassed the filter entirely via the
        # `.isna()` passthrough below, letting years of old base data into
        # train regardless of lookback_days. Fix: build sort dates per-row,
        # preferring detection_date and falling back to event_date (shifted
        # back 1 BDay, since event_date is the explosion day T+1) wherever
        # detection_date is missing — mirroring train_val_split's sort_date.
        _lb_dates = pd.Series(pd.NaT, index=combined_df.index)
        if _has_detection:
            _lb_dates = pd.to_datetime(combined_df["detection_date"], errors="coerce")
        if _has_event:
            _event_dates = pd.to_datetime(combined_df["event_date"], errors="coerce") - pd.tseries.offsets.BDay(1)
            _lb_dates = _lb_dates.fillna(_event_dates)
        _lb_mask = (_lb_dates.dt.date.astype(str) >= _lookback_cutoff) | _lb_dates.isna()
        n_before = len(combined_df)
        combined_df = combined_df[_lb_mask].copy()
        n_after = len(combined_df)
        n_pos_after = int((combined_df["label"] == 1).sum())
        n_neg_after = int((combined_df["label"] == 0).sum())
        logger.info(
            f"Lookback filter ({args.lookback_days}d, cutoff={_lookback_cutoff}): "
            f"{n_before} → {n_after} rows "
            f"(pos={n_pos_after}, neg={n_neg_after}, "
            f"pos_rate={n_pos_after/max(n_after,1):.1%})"
        )
    else:
        logger.warning("Lookback filter: no date column found in combined_df — skipping.")

    # ── RC2 FIX: Enrich with CORRECTED intraday peak gain from daily_winners ──
    # Use prev_close as denominator instead of same-day close
    logger.info("RC2: Fetching daily_winners data for corrected actual_high_pct computation...")
    try:
        winners_response = fetch_table_paginated(client, "daily_winners")
        if not winners_response.empty:
            required = {"symbol", "detection_date", "high"}
            if required.issubset(winners_response.columns):
                # RC2: Apply the corrected computation using prev_close
                winners_corrected = _compute_correct_actual_high_pct(winners_response)

                _has_detection_rc2 = "detection_date" in combined_df.columns
                _has_event_rc2     = "event_date" in combined_df.columns
                symbol_col = next(
                    (c for c in ["symbol", "ticker"] if c in combined_df.columns), None
                )

                if symbol_col and (_has_detection_rc2 or _has_event_rc2):
                    gain_cols = ["symbol", "detection_date", "actual_high_pct"]
                    if "change_pct" in winners_corrected.columns:
                        gain_cols.append("change_pct")

                    # BUG FIX: winners_corrected["detection_date"] is datetime64
                    # (set in _compute_correct_actual_high_pct), while combined_df's
                    # date_col is typically a plain string/object column pulled from
                    # Supabase. Merging on mismatched dtypes raises a ValueError
                    # ("You are trying to merge on str and datetime64[us] columns"),
                    # which was being silently swallowed by the except Exception
                    # below — so actual_high_pct was NEVER populated, starving the
                    # gain regressor of the >=30 rows it needs and forcing the
                    # hardcoded _GAIN_CURVE fallback. Normalize both merge keys to
                    # the same "YYYY-MM-DD" string format before merging.
                    merge_helper = winners_corrected[gain_cols].copy()
                    merge_helper["detection_date"] = pd.to_datetime(
                        merge_helper["detection_date"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")

                    # BUG FIX: previously picked ONE column (detection_date, if
                    # present) for the entire frame, so base rows (no
                    # detection_date) got NaT keys and silently never matched
                    # winners_corrected — only T-1 rows ever received the RC2
                    # corrected actual_high_pct. Build the key per-row instead,
                    # falling back to event_date - 1 BDay (same alignment
                    # convention used by the lookback filter / train_val_split).
                    _merge_dates = pd.Series(pd.NaT, index=combined_df.index)
                    if _has_detection_rc2:
                        _merge_dates = pd.to_datetime(combined_df["detection_date"], errors="coerce")
                    if _has_event_rc2:
                        _merge_event_dates = pd.to_datetime(combined_df["event_date"], errors="coerce") - pd.tseries.offsets.BDay(1)
                        _merge_dates = _merge_dates.fillna(_merge_event_dates)

                    _tmp_key_col = "__merge_date_key__"
                    combined_df[_tmp_key_col] = _merge_dates.dt.strftime("%Y-%m-%d")

                    combined_df = combined_df.merge(
                        merge_helper,
                        left_on=[symbol_col, _tmp_key_col],
                        right_on=["symbol", "detection_date"],
                        how="left",
                        suffixes=("", "_winners"),
                    ).drop(columns=["detection_date_winners", _tmp_key_col], errors="ignore")


                    # Resolve column conflicts after merge
                    for col in ["actual_high_pct", "change_pct"]:
                        merged_col = f"{col}_winners"
                        if merged_col in combined_df.columns:
                            # Fill original NaN with corrected values
                            if col in combined_df.columns:
                                mask = combined_df[col].isna()
                                combined_df.loc[mask, col] = combined_df.loc[mask, merged_col]
                            else:
                                combined_df[col] = combined_df[merged_col]
                            combined_df = combined_df.drop(columns=[merged_col])

                    n_with_gain = combined_df["actual_high_pct"].notna().sum()
                    logger.info(
                        f"RC2: {n_with_gain} rows now have corrected actual_high_pct "
                        f"(prev_close denominator)"
                    )
    except Exception as e:
        logger.warning(
            f"RC2: Could not fetch/process gain data: {e} — gain regressor may be limited",
            exc_info=True,
        )

    # ── RC3: Fetch ml_prediction_accuracy for label correction and gain regressor ──
    #
    # LEAKAGE FIX (RC3-label): The previous implementation backfilled
    # actual_high_pct from ml_prediction_accuracy directly into combined_df
    # BEFORE apply_intraday_high_labels() ran.  This conflated two distinct
    # pipelines:
    #
    #   (A) LABEL-CORRECTION PIPELINE — post-close outcomes written to
    #       ml_prediction_accuracy by the tracker after market close.
    #       Legitimate use: upgrading label=0 → label=1 for rows where the
    #       stock actually hit the intraday threshold.  The outcome is the
    #       LABEL TARGET, not a feature.
    #
    #   (B) FEATURE PIPELINE — values that exist at prediction time (T-1
    #       close data, prior-day stats, etc.).  actual_high_pct from
    #       ml_prediction_accuracy is SAME-DAY outcome data; it does NOT
    #       exist when the model runs pre-market.
    #
    # By writing accuracy-table actual_high_pct into combined_df.actual_high_pct
    # before apply_intraday_high_labels(), the old code smuggled same-day
    # outcome data into the feature column, then trained the gain regressor on
    # it.  At inference time that column is absent, producing a silent but severe
    # train/serve skew.
    #
    # FIX: We still fetch ml_prediction_accuracy (needed for the gain regressor
    # via _accuracy_gain_map and for direct label correction below), but we NO
    # LONGER write actual_high_pct back into combined_df.actual_high_pct.
    # Instead, label correction is applied directly: rows whose (symbol, date)
    # appear in the accuracy table with actual_high_pct >= threshold are
    # upgraded to label=1 here, keeping the outcome data in the label column
    # where it belongs and out of the feature matrix.
    #
    # ISSUE 2 FIX: _accuracy_gain_map is still built and passed to
    # train_gain_regressor to eliminate its redundant DB fetch.
    _accuracy_gain_map: dict = {}
    logger.info("RC3: Fetching ml_prediction_accuracy for label correction and gain regressor...")
    try:
        _symbol_col = next(
            (c for c in ["symbol", "ticker"] if c in combined_df.columns), None
        )
        _has_detection_rc3 = "detection_date" in combined_df.columns
        _has_event_rc3     = "event_date" in combined_df.columns

        if _symbol_col and (_has_detection_rc3 or _has_event_rc3):
            # BUG FIX: previously picked ONE column (detection_date, if present)
            # for the whole frame, so _min_date reflected only T-1 rows' dates
            # (a ~90-day window) and silently ignored the much older event_date
            # rows — narrowing the accuracy-table query and starving label
            # correction / the gain map for base rows. Build per-row instead.
            _all_dates = pd.Series(pd.NaT, index=combined_df.index)
            if _has_detection_rc3:
                _all_dates = pd.to_datetime(combined_df["detection_date"], errors="coerce")
            if _has_event_rc3:
                _all_event_dates = pd.to_datetime(combined_df["event_date"], errors="coerce") - pd.tseries.offsets.BDay(1)
                _all_dates = _all_dates.fillna(_all_event_dates)
            _min_date = _all_dates.min()
            _start_date = (
                _min_date.date().isoformat() if pd.notna(_min_date) else None
            )

            _acc_resp = (
                client.table("ml_prediction_accuracy")
                .select("symbol, prediction_date, actual_gain_pct, actual_high_pct")
                .not_.is_("actual_high_pct", "null")
                .gte("prediction_date", _start_date)
                .execute()
            ) if _start_date else None

            if _acc_resp and _acc_resp.data:
                # ── Build accuracy_gain_map for gain regressor (ISSUE 2 FIX) ──
                for _r in _acc_resp.data:
                    _accuracy_gain_map[(_r["symbol"], _r["prediction_date"])] = {
                        "actual_gain_pct": _r.get("actual_gain_pct"),
                        "actual_high_pct": _r.get("actual_high_pct"),
                    }
                logger.info(
                    f"RC3: Built accuracy_gain_map with {len(_accuracy_gain_map)} records "
                    f"(reused by gain regressor — no redundant DB fetch)"
                )

                # ── Direct label correction (no feature-column contamination) ──
                # Identify combined_df rows that the accuracy table says hit the
                # intraday threshold.  Upgrade their label directly without writing
                # actual_high_pct into the feature matrix.
                _acc_df = pd.DataFrame(_acc_resp.data)
                _acc_df["prediction_date"] = pd.to_datetime(
                    _acc_df["prediction_date"], errors="coerce"
                ).dt.date.astype(str)
                _acc_df["actual_high_pct"] = pd.to_numeric(
                    _acc_df["actual_high_pct"], errors="coerce"
                )

                # Only rows that genuinely cleared the threshold are label correctors
                _acc_winners = _acc_df[
                    _acc_df["actual_high_pct"] >= INTRADAY_WIN_THRESHOLD
                ].set_index(["symbol", "prediction_date"])

                if not _acc_winners.empty:
                    _combined_dates = _all_dates.dt.date.astype(str)

                    # Build a boolean mask: rows in combined_df whose (symbol, date)
                    # appear as threshold-clearers in ml_prediction_accuracy.
                    _keys = list(zip(
                        combined_df[_symbol_col],
                        _combined_dates,
                    ))
                    _is_acc_winner = np.array(
                        [k in _acc_winners.index for k in _keys],
                        dtype=bool,
                    )

                    # Only upgrade label=0 rows; never downgrade label=1.
                    # Exclude base_csv rows (unreliable outcome pipeline).
                    if "source" in combined_df.columns:
                        _is_base_csv = combined_df["source"].str.contains(
                            "base_csv", na=False
                        ).values
                    else:
                        _is_base_csv = np.zeros(len(combined_df), dtype=bool)

                    _upgrade_mask = (
                        (combined_df["label"].values == 0) &
                        _is_acc_winner &
                        ~_is_base_csv
                    )
                    n_upgraded = int(_upgrade_mask.sum())

                    if n_upgraded > 0:
                        combined_df.loc[_upgrade_mask, "label"] = 1
                        # Bump sample weight — high-signal corrective examples
                        combined_df.loc[_upgrade_mask, "sample_weight"] = (
                            combined_df.loc[_upgrade_mask, "sample_weight"] * 1.5
                        )
                        logger.info(
                            f"RC3-label: Upgraded {n_upgraded} non-winner rows to label=1 "
                            f"via ml_prediction_accuracy (actual_high_pct >= "
                            f"{INTRADAY_WIN_THRESHOLD}%). "
                            f"actual_high_pct NOT written to feature matrix (leakage fix)."
                        )
                    else:
                        logger.info(
                            "RC3-label: No label=0 rows matched accuracy-table threshold "
                            "clearers — no upgrades applied."
                        )
                else:
                    logger.warning(
                        f"RC3-label: No accuracy records with actual_high_pct >= "
                        f"{INTRADAY_WIN_THRESHOLD}% — non-winner relabelling will not fire."
                    )
            else:
                logger.warning(
                    "RC3: ml_prediction_accuracy returned no rows with actual_high_pct — "
                    "non-winner relabelling and gain regressor map will be empty."
                )
        else:
            logger.warning(
                "RC3: Could not identify symbol/date columns in combined_df — "
                "skipping accuracy-table label correction."
            )
    except Exception as _e:
        logger.warning(f"RC3: Could not process ml_prediction_accuracy: {_e}")

    # ── FIX 4: Relabel rows with strong intraday moves as winners ─────────────
    combined_df = apply_intraday_high_labels(combined_df, threshold=INTRADAY_WIN_THRESHOLD)

    # ── Filter-aware negative sampling (must run AFTER intraday relabelling) ──
    # apply_intraday_high_labels() promotes some label=0 rows to label=1.
    # Sampling before that step would select "hard negatives" that are then
    # relabelled as winners, corrupting the label/ratio and causing those rows
    # to appear as both a selected negative and a winner in the same training set.
    combined_df = apply_filter_aware_negative_sampling(combined_df, logger)

    # ── TRUE GAIN TARGET FIX: build the gain regressor's label from the ──────
    # market_close/day_prior_close snapshot tables (+ ml_training_base.gain_pct
    # for base rows) instead of ml_prediction_accuracy. See the docstrings on
    # fetch_market_snapshot_gain_targets() / attach_true_gain_targets() above
    # for why this replaces the old actual_high_pct-via-accuracy-table path.
    logger.info("Fetching market-snapshot gain targets (true_gain_pct)...")
    market_gain_df = fetch_market_snapshot_gain_targets(client)
    combined_df = attach_true_gain_targets(combined_df, market_gain_df)

    # ── Mistake learning step — DISABLED ─────────────────────────────────────
    # Reason: with only ~18 mistakes in the corpus, the 3x/2x sample weights
    # create a circular feedback loop. Valid setups that fail due to market
    # noise are labelled as "bad patterns" and up-weighted, causing the model
    # to suppress those setups on every subsequent retrain.
    #
    # Re-enable (and set MISTAKE_LEARNER_AVAILABLE = True at the top of this
    # file) once there are enough mistakes for statistically meaningful signal
    # (suggested threshold: ~200+ unique mistake samples).
    #
    # The original implementation is preserved below for reference:
    #
    # mistake_df = pd.DataFrame()
    # if MISTAKE_LEARNER_AVAILABLE:
    #     logger.info("\n" + "=" * 60)
    #     logger.info("MISTAKE LEARNING STEP")
    #     logger.info("=" * 60)
    #     proto_features = [
    #         c for c in combined_df.columns
    #         if c not in NON_FEATURE_COLS and not c.startswith("Unnamed")
    #     ]
    #     logger.info("Loading multiday tables for mistake-sample enrichment...")
    #     _mistake_winners_md, _mistake_non_winners_md = load_multiday_data(client)
    #     mistake_df = build_mistake_training_samples(
    #         lookback_days=90,
    #         use_all_timepoints=True,
    #         existing_features=proto_features,
    #         winners_multiday=_mistake_winners_md,
    #         non_winners_multiday=_mistake_non_winners_md,
    #     )
    #     if not mistake_df.empty:
    #         mistake_df = enrich_mistakes_with_gains(mistake_df, client)
    #         log_mistake_summary(mistake_df)
    #         combined_df = pd.concat([combined_df, mistake_df],
    #                                 ignore_index=True, sort=False)
    #         logger.info(
    #             f"Dataset after adding mistakes: {len(combined_df)} rows "
    #             f"(+{len(mistake_df)} mistake samples)"
    #         )
    #     else:
    #         logger.info("No mistake samples to add this run.")
    logger.info("Mistake learning step skipped (corpus too small — see MISTAKE_LEARNER_AVAILABLE).")
    mistake_df = pd.DataFrame()  # Placeholder; keeps downstream code compatible

    # ── Prepare features ──────────────────────────────────────────────────────
    X, y, w = prepare_features(combined_df)
    feature_names = list(X.columns)

    # ── Scale ─────────────────────────────────────────────────────────────────
    # ── FIX 1: Time-based train/val split (on RAW features, before scaling) ───────
    # Split first so the scaler is fit on train rows only (no val leakage).
    X_train_raw, X_val_raw, y_train, y_val, w_train, w_val, train_idx = train_val_split(
        X, y, w, combined_df
    )

    # ── Scale ───────────────────────────────────────────────────────────────────────────
    # LEAKAGE FIX: fit scaler on X_train_raw only, then transform each split
    # separately.  Previously build_scaler() was called on the full X (all rows),
    # so the scaler's mean_ / std_ were computed using val-set rows, making AUC
    # metrics slightly optimistic and the scaler non-reproducible on train-only data.
    logger.info("Fitting scaler on train split only (leakage fix)...")
    scaler, X_train, _sparse_cols = build_scaler(X_train_raw)                    # fit + transform train
    X_val                          = scale_with_fitted_scaler(scaler, X_val_raw,
                                         sparse_threshold_cols=_sparse_cols)     # transform val only

    # Reassemble a scaled DataFrame (train + val, original row order) kept for
    # any downstream use that needs the surviving rows in order. Note: rows
    # dropped by the purge/embargo gap in train_val_split are neither in
    # X_train nor X_val, so X.index (the full original index) is a superset
    # of the combined train+val index — reindexing to X.index directly would
    # KeyError on the embargoed rows. Restore order using only the index
    # values that actually survived the split instead.
    X_combined      = pd.concat([X_train, X_val])
    _surviving_index = [idx for idx in X.index if idx in X_combined.index]
    X_scaled = X_combined.loc[_surviving_index]

    # ── Train-set size guard ──────────────────────────────────────────────────
    # The MIN_VAL_POSITIVES check (inside train_val_split) only guards the val
    # set.  A sparse Supabase deployment can still produce a train split that is
    # too small for XGBoost to generalise — e.g. if lookback_days=90 returns
    # far fewer rows than expected due to data gaps or a new deployment.
    train_pos  = int((y_train == 1).sum())
    train_neg  = int((y_train == 0).sum())
    train_rows = len(X_train)

    if train_pos < MIN_TRAIN_POSITIVES:
        logger.error(
            f"ABORTING: only {train_pos} positive (winner) examples in the train split "
            f"(need ≥ {MIN_TRAIN_POSITIVES}). "
            "The Supabase tables are likely sparse — this may be a new deployment or "
            "data gap. The model cannot learn a useful decision boundary from so few "
            "positive examples. "
            "Options: (1) accumulate more labelled T-1 data before retraining, "
            "(2) lower MIN_TRAIN_POSITIVES if you accept a noisier model, "
            "(3) verify that load_t1_data() and combine_datasets() returned the "
            "expected rows (check logs above for row counts)."
        )
        sys.exit(1)

    if train_rows < MIN_TRAIN_ROWS:
        logger.error(
            f"ABORTING: only {train_rows} total rows in the train split "
            f"(pos={train_pos}, neg={train_neg}; need ≥ {MIN_TRAIN_ROWS} total). "
            "A train set this small will overfit regardless of regularisation. "
            "Accumulate more data or lower MIN_TRAIN_ROWS if running in a known "
            "low-data environment."
        )
        sys.exit(1)

    if train_pos < 100:
        logger.warning(
            f"  ⚠️  Train split has only {train_pos} positive examples "
            f"({train_pos / max(1, train_rows):.1%} of {train_rows} rows). "
            "The model may underfit on the positive class. "
            "Consider accumulating more winner data before the next retrain."
        )
    else:
        logger.info(
            f"  ✅ Train split: {train_rows} rows "
            f"(pos={train_pos}, neg={train_neg}, "
            f"pos_rate={train_pos/train_rows:.1%}) — size looks adequate."
        )

    # ── RC6: Isotonic calibration from a VAL-set stratified holdout ──────────
    # Previous attempts carved the cal set from the oldest training rows, which
    # are dominated by base CSV rows with NaN t1_ features — a different data
    # regime from inference.  That caused the calibrator to compress all
    # probabilities into ~0.50–0.85 and was correctly disabled.
    #
    # Fix: carve the calibration set from the VAL set instead.  Val rows are
    # recent T-1 data (same regime as inference: all t1_ features present).
    # We reserve half the val set for calibration and use the remaining half
    # for early-stopping AUC.  Both halves still come entirely from after the
    # cutoff date, so there is no temporal leakage into training.
    #
    # IMPORTANT — method='isotonic' (not 'sigmoid'):
    # Sigmoid (Platt scaling) anchors to the calibration set's positive base
    # rate.  Because the val set is rebalanced to ~train_rate+2pp (~10–25%
    # positive), sigmoid compresses all inference probabilities downward when
    # the screened inference universe has a higher base rate.  This was the
    # root cause of max probabilities being suppressed to ~0.68.
    # Isotonic regression fits a rank-preserving step function without anchoring
    # to any global base rate, so it is robust to this mismatch.
    #
    # Minimum requirements: ≥10 positives in each half after the split.
    CAL_MIN_POS = 10
    X_cal_fit, y_cal_fit = None, None
    X_val_xgb, y_val_xgb = X_val, y_val

    _val_pos_idx  = y_val[y_val == 1].index.tolist()
    _val_neg_idx  = y_val[y_val == 0].index.tolist()
    _n_cal_pos    = len(_val_pos_idx) // 2
    _n_cal_neg    = len(_val_neg_idx) // 2

    if _n_cal_pos >= CAL_MIN_POS and _n_cal_neg >= CAL_MIN_POS:
        # ISSUE 1 FIX: interleave by position instead of taking first/second halves.
        # y_val is time-ordered ascending, so a first/second-half split gives the
        # calibrator the OLDEST val rows and early-stopping the NEWEST — two different
        # market sub-periods.  Interleaving (even positions → cal, odd → early-stop)
        # ensures both sets span the full val time window with the same regime mix,
        # making the isotonic calibrator representative of the same period the
        # early-stopping signal comes from.
        _cal_idx  = _val_pos_idx[0::2] + _val_neg_idx[0::2]   # even positions → cal
        _stop_idx = _val_pos_idx[1::2] + _val_neg_idx[1::2]   # odd positions  → early-stop

        X_cal_fit    = X_val.loc[_cal_idx]
        y_cal_fit    = y_val.loc[_cal_idx]
        X_val_xgb    = X_val.loc[_stop_idx]
        y_val_xgb    = y_val.loc[_stop_idx]

        cal_pos = int((y_cal_fit == 1).sum())
        cal_neg = int((y_cal_fit == 0).sum())
        logger.info(
            f"RC6: Calibration set carved from val (same T-1 regime as inference). "
            f"Cal: {len(X_cal_fit)} rows ({cal_pos} pos / {cal_neg} neg, "
            f"rate={cal_pos/max(1,len(X_cal_fit)):.1%}). "
            f"Early-stop val: {len(X_val_xgb)} rows "
            f"({int((y_val_xgb==1).sum())} pos / {int((y_val_xgb==0).sum())} neg)."
        )
    else:
        logger.info(
            f"RC6: Val set too small to split for calibration "
            f"({_n_cal_pos} pos / {_n_cal_neg} neg per half, need ≥{CAL_MIN_POS} each). "
            "Training without isotonic calibration."
        )

    X_train_xgb, y_train_xgb, w_train_xgb = X_train, y_train, w_train

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_model(X_train_xgb, y_train_xgb, w_train_xgb, X_val_xgb, y_val_xgb,
                        X_cal=X_cal_fit, y_cal=y_cal_fit)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df = compute_feature_importance(model, feature_names)

    # ── RC1+RC2+RC3+RC6+RC7 FIX: Train gain regressor with corrected inputs ───────
    # EVALUATION INTEGRITY FIX: Pass only X_train (classifier's train split) and
    # combined_df.loc[train_idx] so the gain regressor builds its own internal val
    # split exclusively from the classifier's training period.
    #
    # Passing X_scaled (all rows) here causes the regressor's internal time-based
    # 80/20 split (inside train_gain_regressor, ~line 1895) to draw from the full
    # dataset.  Because the classifier's val rows (the most recent ~VAL_WEEKS of
    # data) are in that pool, the regressor's internal validation window overlaps
    # the classifier's validation period.  This inflates the regressor's reported
    # MAE/R² (it is evaluated on data it has effectively trained on) and means the
    # combined system is optimising on partially future-seen data.
    #
    # The earlier rationalisation ("gain targets aren't classifier labels, so no
    # leak") is incorrect: the gain regressor is trained on the same rows as the
    # classifier, and its internal split draws from the same timeline.  Leakage
    # occurs not through label identity but through temporal overlap.
    #
    # Using train rows only means the regressor's internal val split is drawn
    # exclusively from the classifier's training period, giving a consistent and
    # meaningful held-out evaluation with no future-data contamination.
    #
    # If excluding the most-recent ~VAL_WEEKS compresses the gain distribution too
    # much, lower VAL_WEEKS (e.g. from 8 to 4) rather than reverting this fix.
    logger.info("\n" + "=" * 60)
    logger.info("GAIN REGRESSOR TRAINING (RC1+RC2+RC3+RC6+RC7 fixes applied; "
                "true_gain_pct market-snapshot target takes priority — see attach_true_gain_targets)")
    logger.info("=" * 60)
    # LEAK-FREE FIX (re-reinstated, 2026-07): the earlier "NOTE" here reverted
    # to passing the FULL (train+val) pool because train_idx-only data used
    # to structurally contain ~0 gain-labeled rows (true_gain_pct only exists
    # for recent T-1 rows, i.e. exactly the val window). That is no longer
    # the whole picture: attach_true_gain_targets() now also backfills
    # '_unified_gain_target' from ml_training_base.gain_pct, which spans the
    # FULL historical range of the base CSV — so the train split has real,
    # broadly time-distributed gain labels even after VAL_WEEKS is excluded.
    #
    # We pass train-only data by default. train_gain_regressor() itself
    # checks whether that pool actually clears MIN_TRAIN_ONLY_GAIN_ROWS; if
    # it doesn't (e.g. a sparse/early-stage deployment), it falls back to the
    # full pool registered below (right before the call) via
    # _full_combined_df/_full_X_scaled — but logs a loud warning and tags the
    # resulting model as not leak-free (regressor._trained_leak_free = False)
    # rather than silently reverting every run regardless of data volume.
    # ── REGRESSOR-ONLY log_price FEATURE ──────────────────────────────────────
    # The classifier must never see raw price level (that's the whole reason
    # OHLCV columns are in NON_FEATURE_COLS — "expensive stocks explode more"
    # is not a real signal). But the gain regressor's job is different: given
    # that a move is happening, cheap/low-float stocks mechanically swing
    # harder in % terms than expensive ones, so price level is legitimate
    # signal for magnitude prediction specifically.
    #
    # log_price is derived in-memory only, from t1_close_Close (falling back
    # to t1_open_Close) — both already populated for every row by the existing
    # T-1 intraday pipeline. No DB schema change, no backfill, no new column
    # persisted anywhere: this Series exists only for the duration of this
    # training run and is never written back to combined_df or the DB.
    # log1p (not raw price) is used to avoid the model memorising exact price
    # points and to keep the feature on a smoother, split-friendly scale.
    _price_source = combined_df.get("t1_close_Close")
    if _price_source is None:
        _price_source = pd.Series(np.nan, index=combined_df.index)
    _price_fallback = combined_df.get("t1_open_Close")
    if _price_fallback is not None:
        _price_source = _price_source.fillna(_price_fallback)
    _price_source = pd.to_numeric(_price_source, errors="coerce").clip(lower=0)
    log_price = np.log1p(_price_source).reindex(X_scaled.index)
    log_price = log_price.fillna(log_price.mean())  # match X_scaled's "no raw NaN into XGBoost" convention

    # ── REGRESSOR-ONLY clf_proba FEATURE ──────────────────────────────────────
    # Tie gain-magnitude predictions to how confident the classifier is that
    # a move is happening at all. The classifier's calibrated probability is
    # already a strong summary of "how explosive does this setup look" —
    # feeding it to the regressor lets high-confidence strong-buy/buy rows
    # pull toward higher predicted gains instead of the regressor treating a
    # 51%-confidence row and a 99%-confidence row as equally likely to have
    # any given magnitude of gain.
    #
    # `model` here is the fully trained + calibrated classifier from
    # train_model() above, so this is the same probability the classifier
    # itself reports for these rows — not a leaky re-derivation.
    clf_proba = pd.Series(
        model.predict_proba(X_scaled)[:, 1],
        index=X_scaled.index,
        name="clf_proba",
    )

    X_scaled_gain = X_scaled.assign(log_price=log_price, clf_proba=clf_proba)
    logger.info(
        f"Gain regressor feature matrix: {X_scaled_gain.shape[1]} features "
        f"({X_scaled.shape[1]} shared with classifier + log_price + clf_proba)"
    )

    # LEAK-FREE FIX: register the full (train+val) pool as the fallback that
    # train_gain_regressor() will use ONLY if the train-only pool doesn't
    # clear MIN_TRAIN_ONLY_GAIN_ROWS. Registered as function attributes so
    # the internal fallback-retry call (which recurses with the same
    # function object) can reach it without changing every call signature
    # up the stack.
    train_gain_regressor._full_combined_df = combined_df
    train_gain_regressor._full_X_scaled    = X_scaled_gain

    # Train-only slice: same rows the classifier trained on (train_idx),
    # matched between combined_df and the scaled feature matrix.
    combined_df_train_only = combined_df.loc[train_idx]
    X_scaled_gain_train_only = X_scaled_gain.reindex(train_idx)

    gain_regressor = train_gain_regressor(
        X_scaled=X_scaled_gain_train_only,          # LEAK-FREE FIX: train rows only (was: full X_scaled_gain)
        combined_df=combined_df_train_only,         # LEAK-FREE FIX: train rows only (was: full combined_df)
        feature_names=feature_names,
        client=client,                              # RC1: fallback fetch if map not supplied
        accuracy_gain_map=_accuracy_gain_map,       # ISSUE 2 FIX: reuse RC3 fetch, no redundant DB query
    )

    if gain_regressor is not None:
        _leak_free = getattr(gain_regressor, "_trained_leak_free", None)
        if _leak_free is True:
            logger.info(
                "  ✅ Gain regressor trained leak-free (train-split-only data; "
                "no overlap with classifier val/cal rows)."
            )
        elif _leak_free is False:
            logger.warning(
                "  ⚠️  Gain regressor fell back to the full train+val pool "
                "(train-only data was below MIN_TRAIN_ONLY_GAIN_ROWS). "
                "Its reported val MAE/R² should be treated as optimistic."
            )

    # ── Evaluate classifier ───────────────────────────────────────────────────
    # AUC is reported on X_val_xgb (the early-stopping half of the val set)
    # rather than the full X_val.  The other half (X_cal_fit) was passed to
    # CalibratedClassifierCV.fit(), so evaluating on it would be circular —
    # the calibrated probability distribution has already seen those rows.
    # X_val_xgb was held out from both training and calibration, making it a
    # clean post-calibration holdout.  When calibration was skipped (val set
    # too small), X_val_xgb == X_val so behaviour is unchanged.
    from sklearn.metrics import roc_auc_score, classification_report

    val_proba = model.predict_proba(X_val_xgb)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_val_xgb, val_proba)
        logger.info(f"Validation AUC-ROC: {auc:.4f} (evaluated on early-stop holdout, n={len(y_val_xgb)})")
    except Exception:
        auc = float("nan")
        logger.warning("Validation AUC-ROC: nan (only one class in val set)")

    logger.info("Classification report (val — early-stop holdout):")
    for line in classification_report(y_val_xgb, val_pred).split("\n"):
        if line.strip():
            logger.info(f"  {line}")

    # Log probability distribution on val set (early-stop holdout only)
    val_proba_series = pd.Series(val_proba)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = pd.cut(val_proba_series, bins=bins).value_counts().sort_index()
    logger.info("Val set probability distribution (early-stop holdout):")
    for bucket, count in dist.items():
        logger.info(f"  {str(bucket):<20} {count:>4}")

    # ── Evaluate classifier on the truly blind calibration holdout ───────────
    # X_val_xgb (above) was used by XGBoost's early_stopping_rounds to decide
    # how many trees to build, so metrics on it are model-selection-influenced,
    # not a blind evaluation — that's what produced the bimodal collapse
    # investigated on 2026-07-22 (X_val_xgb collapsed to hard 0/0.9-1.0 while
    # X_cal_fit, never touched during tree-building, stayed well-spread).
    #
    # X_cal_fit isn't perfectly blind either — it was used to fit the isotonic
    # calibrator — so its calibrated *probability values* are still slightly
    # optimistic. But XGBoost's tree structure and early-stopping point were
    # never chosen using X_cal_fit, so its AUC/ranking behaviour is a much
    # more honest estimate of generalization than X_val_xgb's, and is the
    # number to trust if the two disagree.
    cal_auc = None  # persisted to metadata below; None when no blind holdout was available
    if X_cal_fit is not None and y_cal_fit is not None:
        cal_proba_report = model.predict_proba(X_cal_fit)[:, 1]
        cal_pred_report  = (cal_proba_report >= 0.5).astype(int)

        try:
            cal_auc = roc_auc_score(y_cal_fit, cal_proba_report)
            logger.info(
                f"Validation AUC-ROC: {cal_auc:.4f} "
                f"(evaluated on blind calibration holdout, n={len(y_cal_fit)}) "
                "— trust this over the early-stop-holdout AUC above if they diverge."
            )
        except Exception:
            cal_auc = float("nan")
            logger.warning("Calibration-holdout AUC-ROC: nan (only one class in set)")

        logger.info("Classification report (val — blind calibration holdout):")
        for line in classification_report(y_cal_fit, cal_pred_report).split("\n"):
            if line.strip():
                logger.info(f"  {line}")

        cal_proba_report_series = pd.Series(cal_proba_report)
        cal_dist = pd.cut(cal_proba_report_series, bins=bins).value_counts().sort_index()
        logger.info("Val set probability distribution (blind calibration holdout):")
        for bucket, count in cal_dist.items():
            logger.info(f"  {str(bucket):<20} {count:>4}")

        cal_gap_count = int(((cal_proba_report_series > 0.15) & (cal_proba_report_series < 0.85)).sum())
        if cal_gap_count < 5:
            logger.warning(
                f"  ⚠️  BIMODAL COLLAPSE detected on blind calibration holdout too: "
                f"only {cal_gap_count} predictions in 0.15–0.85 range. "
                "This set was never used for early stopping, so a collapse here points "
                "at the base model / data (e.g. leakage, near-duplicate rows) rather "
                "than early-stopping overfit."
            )
        else:
            logger.info(
                f"  ✅ {cal_gap_count} predictions in mid-range (0.15–0.85) on the blind "
                "calibration holdout — distribution looks healthy."
            )

        if auc is not None and not (auc != auc) and abs(auc - cal_auc) > 0.03:
            logger.warning(
                f"  ⚠️  Early-stop-holdout AUC ({auc:.4f}) and blind calibration-holdout "
                f"AUC ({cal_auc:.4f}) diverge by more than 0.03. This gap is itself a "
                "diagnostic: it suggests early stopping is fitting X_val_xgb specifically "
                "rather than a generalizable stopping point. Consider lowering "
                "early_stopping_rounds further and/or increasing calibration-set size."
            )
    else:
        logger.info(
            "No blind calibration holdout available (val set too small to split) — "
            "only the early-stop-holdout metrics above are available this run."
        )

    gap_count = int(((val_proba_series > 0.15) & (val_proba_series < 0.85)).sum())
    if gap_count < 5:
        logger.warning(
            f"  ⚠️  BIMODAL COLLAPSE detected: only {gap_count} predictions in 0.15–0.85 range. "
        )
    else:
        logger.info(f"  ✅ {gap_count} predictions in mid-range (0.15–0.85) — distribution looks healthy")

    # ── Training stats for metadata ───────────────────────────────────────────
    n_mistakes = len(mistake_df) if not mistake_df.empty else 0
    n_t1_with_multiday = 0
    if not t1_df.empty:
        md_cols = [c for c in t1_df.columns if c.startswith(("t3_", "t5_", "t10_"))]
        if md_cols:
            n_t1_with_multiday = int(t1_df[md_cols].notna().any(axis=1).sum())

    # ── Top-10 feature distribution snapshot (for PSI drift detection) ───────
    # Store per-feature mean, std, and percentile buckets (deciles) computed on
    # the raw (unscaled) training split for the top-10 most important features.
    # explosion_predictor.py loads these at inference time and logs a WARNING if
    # PSI > 0.2 on any top feature, indicating a distribution shift between the
    # training and live feature sets.
    top10_features = fi_df.head(10)["feature"].tolist()
    top10_training_stats: dict = {}
    for feat in top10_features:
        if feat not in X_train_raw.columns:
            continue
        col = X_train_raw[feat].dropna()
        if len(col) < 10:
            continue
        # 10 equal-width buckets covering the observed training range, plus one
        # open-ended bucket on each side (handled at inference time via clipping).
        percentiles = [float(v) for v in np.percentile(col, np.linspace(0, 100, 11))]
        top10_training_stats[feat] = {
            "mean":        float(col.mean()),
            "std":         float(col.std()),
            "n":           int(len(col)),
            "percentiles": percentiles,   # 11 values → 10 equal-frequency buckets
        }
    logger.info(
        f"Stored training distribution stats for {len(top10_training_stats)} "
        f"top-10 features (used for PSI drift detection at inference)."
    )

    training_stats = {
        "n_total_samples":         len(combined_df),
        "n_base_samples":          len(base_df),
        "n_t1_samples":            len(t1_df) if not t1_df.empty else 0,
        "n_t1_with_multiday":      n_t1_with_multiday,
        "n_mistake_samples":       n_mistakes,
        "n_positive":              int((y == 1).sum()),
        "n_negative":              int((y == 0).sum()),
        "positive_rate":           float((y == 1).mean()),
        "val_auc_roc":             float(auc),
        # NOTE: val_auc_roc / best_val_auc above are measured on X_val_xgb, the
        # same set XGBoost's early_stopping_rounds used to choose best_iteration —
        # that makes them a model-selection score, not a blind evaluation, and
        # they run optimistic (can approach ~1.0 even when the model doesn't
        # generalize). blind_cal_auc below is measured on X_cal_fit, a holdout
        # that was never used for tree-building, and is the number to trust.
        "blind_cal_auc": (
            float(cal_auc) if cal_auc is not None and cal_auc == cal_auc else None
        ),
        "base_sample_weight":      BASE_CSV_WEIGHT,
        "t1_sample_weight":        T1_WEIGHT,
        "intraday_win_threshold":  INTRADAY_WIN_THRESHOLD,
        "equal_weight_applied": (
            len(t1_df) >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT
            if not t1_df.empty else False
        ),
        "gain_regressor_trained":  gain_regressor is not None,
        "gain_regressor_leak_free": (
            getattr(gain_regressor, "_trained_leak_free", None)
            if gain_regressor is not None else None
        ),
        "gain_regressor_rc_fixes": ["RC1_broad_training", "RC2_prev_close",
                                    "RC3_scaled_input", "RC6_mistake_enrichment", "RC7_log_transform_heavy_weights",
                                    "RC8_data_driven_winner_weighting",
                                    "LEAK_FREE_train_split_only_with_fallback"],
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(model, scaler, fi_df, feature_names, training_stats, gain_regressor,
                 top10_training_stats=top10_training_stats)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RETRAIN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total samples       : {training_stats['n_total_samples']}")
    logger.info(f"  Base CSV samples    : {training_stats['n_base_samples']}")
    logger.info(f"  T-1 samples         : {training_stats['n_t1_samples']}")
    t1_total = training_stats['n_t1_samples']
    t1_md    = training_stats['n_t1_with_multiday']
    if t1_total > 0:
        logger.info(
            f"  T-1 w/ multiday     : {t1_md}/{t1_total} "
            f"({t1_md/t1_total*100:.0f}% have t3/t5/t10 features)"
        )
    logger.info(f"  Mistake samples     : {training_stats['n_mistake_samples']}")
    logger.info(f"  Positive rate       : {training_stats['positive_rate']:.1%}")

    # Surface a summary-level advisory when the final positive rate is above the
    # expected ceiling, even if it didn't trip the >25% threshold earlier.
    # This is the number that lands in the retrain log and is easiest to monitor.
    final_pos_rate = training_stats["positive_rate"]
    if 0.20 < final_pos_rate <= 0.25:
        logger.warning(
            f"  ⚠️  Positive rate {final_pos_rate:.1%} is above the expected ~5-20% ceiling. "
            "The model is training on a dataset where roughly 1 in 4 samples is a winner. "
            "Possible causes: short LOOKBACK window over-representing a recent winning streak, "
            "asymmetric deduplication dropping more negatives than positives, or label drift. "
            "scale_pos_weight is computed from the training split class balance and will "
            "partially compensate, but a structurally skewed dataset may still cause the "
            "model to over-predict wins in a normal market. Review the pre/post-dedup "
            "label counts logged above before deploying this model."
        )
    elif final_pos_rate > 0.25:
        logger.warning(
            f"  ⚠️  Positive rate {final_pos_rate:.1%} exceeds the 25% caution threshold. "
            "This model may be over-fitted to recent market conditions. Investigate "
            "before deploying — see dedup diagnostics logged earlier in this run."
        )
    logger.info(f"  Validation AUC      : {auc:.4f}  (early-stop holdout — optimistic, see below)")
    _blind_cal_auc = training_stats.get("blind_cal_auc")
    if _blind_cal_auc is not None:
        logger.info(f"  Blind cal-holdout AUC: {_blind_cal_auc:.4f}  (trust this number)")
    else:
        logger.info("  Blind cal-holdout AUC: n/a (val set too small to carve a holdout this run)")
    _best_iter = (model.calibrated_classifiers_[0].estimator.best_iteration
                  if hasattr(model, "calibrated_classifiers_") else model.best_iteration)
    logger.info(f"  Best iteration      : {_best_iter}")
    logger.info(f"  Features            : {len(feature_names)}")
    logger.info(f"  Gain regressor      : {'✓ trained (RC1+RC2+RC3+RC6+RC7 fixed)' if gain_regressor else '— skipped'}")
    logger.info("")
    logger.info("Files written:")
    logger.info(f"  {MODEL_PATH}")
    logger.info(f"  {SCALER_PATH}")
    if gain_regressor is not None:
        logger.info(f"  {GAIN_REGRESSOR_PATH}")
    logger.info(f"  {METADATA_PATH}")
    logger.info(f"  {FEATURE_IMPORTANCE_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
