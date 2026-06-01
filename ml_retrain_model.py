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
     (SPW_MAX raised from 3.0 → 5.0 to better handle the ~8.8x production imbalance.)

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
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
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

# Mistake learner — high-signal training samples from past prediction errors
try:
    from ml_mistake_learner import build_mistake_training_samples, log_mistake_summary
    MISTAKE_LEARNER_AVAILABLE = True
except ImportError:
    MISTAKE_LEARNER_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ml_mistake_learner.py not found — mistake-learning step will be skipped."
    )

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

BASE_CSV_WEIGHT         = 1.5
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
SPW_MAX = 10.0   # Raised from 5.0 — actual imbalance is ~8.7x (11.5% positive rate).
                 # Capping at 5.0 was under-weighting positives and collapsing the
                 # probability spread toward HOLD.  10.0 allows the model to use the
                 # natural class imbalance ratio, producing higher max probabilities
                 # for the strongest signals and restoring BUY/STRONG BUY signals.

XGBOOST_PARAMS = {
    "n_estimators":       300,
    "max_depth":          3,       # reduced from 6 → less overfitting
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   10,      # raised from 3 → requires more samples per leaf
    "gamma":              2.0,     # raised from 0.1 → higher minimum gain to split
    "reg_alpha":          0.5,     # raised from 0.1 → more L1 regularisation
    "reg_lambda":         2.0,     # raised from 1.0 → more L2 regularisation
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
    "early_stopping_rounds": 100,  # Raised from 50 — prevents stopping at 34 trees.
}

# Columns excluded from the feature matrix X.
NON_FEATURE_COLS = {
    "id", "created_at", "updated_at", "date", "symbol", "ticker",
    "label", "source", "sample_weight", "detection_date", "explosion_date",
    "change_pct", "rank", "notes", "mistake_type", "actual_gain_pct",
    "actual_high_pct", "_sort_date",
    # Label-leaking columns: present in training tables but unavailable at prediction time
    "gain_pct", "volume_spike",
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
}

T1_MARKER_PREFIXES = ("t1_", "open_", "close_")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
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


def fetch_table_paginated(client: Client, table: str, page_size: int = 1000) -> pd.DataFrame:
    """Fetch all rows from a Supabase table using pagination."""
    rows   = []
    offset = 0
    while True:
        resp = (
            client.table(table)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        logger.info(f"  {table}: fetched {len(rows)} rows so far...")
        if len(batch) < page_size:
            break
        offset += page_size
    df = pd.DataFrame(rows)
    logger.info(f"  {table}: total {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_base_training_data(client: Client) -> pd.DataFrame:
    """Load original CSV data from ml_training_base."""
    logger.info(f"Loading base training data from '{TABLE_BASE}'...")
    df = fetch_table_paginated(client, TABLE_BASE)
    if df.empty:
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
    if 'actual_high_pct' in base_df.columns:
        would_upgrade = (
            (base_df['label'] == 0) & 
            (pd.to_numeric(base_df['actual_high_pct'], errors='coerce') >= 15.0)
        ).sum()
        logger.info(f"  Rows intraday_high_labels would upgrade: {would_upgrade}")


def load_multiday_data(client: Client) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            df = fetch_table_paginated(client, table)
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


def load_t1_data(client: Client) -> pd.DataFrame:
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
    winners_multiday, non_winners_multiday = load_multiday_data(client)

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
        df = fetch_table_paginated(client, table)
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

    if not accuracy_rows:
        logger.info("RC6: No accuracy data found for mistake symbols — skipping enrichment")
        return mistake_df

    acc_df = pd.DataFrame(accuracy_rows).rename(columns={"prediction_date": "detection_date"})
    acc_df = acc_df.dropna(subset=["symbol", "detection_date"])

    result = mistake_df.copy()
    # Merge in actual_gain_pct and actual_high_pct where missing
    merged = result.merge(
        acc_df[["symbol", "detection_date", "actual_gain_pct", "actual_high_pct"]],
        on=["symbol", "detection_date"],
        how="left",
        suffixes=("", "_acc"),
    )

    # Fill missing values from the accuracy table
    if "actual_gain_pct" not in merged.columns:
        merged["actual_gain_pct"] = np.nan
    if "actual_high_pct" not in merged.columns:
        merged["actual_high_pct"] = np.nan

    # Prefer existing values; only fill where NaN
    for col in ["actual_gain_pct", "actual_high_pct"]:
        acc_col = f"{col}_acc"
        if acc_col in merged.columns:
            was_nan = merged[col].isna()
            merged.loc[was_nan, col] = merged.loc[was_nan, acc_col]
            merged = merged.drop(columns=[acc_col])

    enriched_count = merged["actual_high_pct"].notna().sum()
    logger.info(
        f"RC6: Enriched {enriched_count}/{len(merged)} mistake rows with gain data"
    )
    return merged


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
    if t1_df.empty:
        logger.info("Combining: base data only (no T-1 data yet)")
        return base_df.copy()

    t1_count = len(t1_df)
    if t1_count >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT:
        logger.info(
            f"T-1 data ({t1_count} rows) >= threshold ({MIN_T1_ROWS_FOR_EQUAL_WEIGHT}). "
            "Using equal sample weights (1.0 / 1.0)."
        )
        base_df = base_df.copy()
        base_df["sample_weight"] = 1.0
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

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    nan_pct = X.isna().mean().mean() * 100
    logger.info(f"Overall NaN rate: {nan_pct:.1f}% (expected for cross-lag rows)")

    return X, y, w


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def build_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """
    Fit scaler on train-split rows only. Returns scaler + scaled X_train.

    LEAKAGE FIX: The scaler is now fit exclusively on X_train so that
    validation-set rows never influence the scaler's mean_ / std_ parameters.
    Call scale_with_fitted_scaler(scaler, X_val) to transform the val set
    (or any other split) using the same, already-fitted scaler.

    NaN RESTORATION FIX (t1_ features): Sparse columns (coverage < SPARSE_THRESHOLD)
    have NaN restored AFTER scaling so XGBoost can use its native missing-value
    branch logic.  Previously fillna(col_mean) → scale → fillna(0.0) made these
    columns appear as the constant 0.0 for 85% of rows, hiding them from gain-based
    feature importance entirely.  StandardScaler still receives NaN-free input
    (required), but XGBoost receives NaN for genuinely absent values, matching the
    inference path in _scale_features() in explosion_predictor.py.
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

    return scaler, X_scaled


def scale_with_fitted_scaler(
    scaler: StandardScaler,
    X: pd.DataFrame,
    sparse_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Transform X using an already-fitted scaler (e.g. to scale the val set or
    to reassemble a full scaled DataFrame for the gain regressor).

    NaN RESTORATION FIX: mirrors build_scaler — sparse columns (those whose
    scaler mean_ was computed from < sparse_threshold coverage) have NaN restored
    after scaling so XGBoost receives genuinely missing values rather than 0.0.
    The threshold is inferred from scaler.n_samples_seen_ and the non-zero
    count stored in scaler.mean_ (columns that were all-NaN get mean_=0).
    Simpler: we restore NaN wherever the INPUT X had NaN, for all columns whose
    overall NaN rate in X exceeds (1 - sparse_threshold).  This is consistent
    with build_scaler which uses X_train coverage to decide.
    """
    col_means = pd.Series(scaler.mean_, index=X.columns)
    X_filled  = X.fillna(col_means)

    X_scaled_vals = scaler.transform(X_filled)
    X_scaled      = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
    X_scaled      = X_scaled.fillna(0.0)

    # ── Restore NaN for sparse columns (mirrors build_scaler logic) ────────────
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
    (isotonic regression, cv='prefit') before being returned. This
    corrects the extreme probability clustering caused by AUC training
    + heavy scale_pos_weight, pulling predictions away from 0/1 and
    restoring meaningful separation across the SIGNAL_THRESHOLDS.
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

    # RC6: Post-training probability calibration
    # AUC training + scale_pos_weight together push probabilities toward
    # extremes, causing 60%+ of post-screener stocks to cluster at STRONG BUY.
    # Fitting a sigmoid calibrator on a clean held-out calibration set corrects
    # this without affecting AUC / rank order (isotonic regression is rank-preserving).
    if X_cal is not None and y_cal is not None:
        n_cal_pos = int((y_cal == 1).sum())
        n_cal_neg = int((y_cal == 0).sum())
        if n_cal_pos >= 10 and n_cal_neg >= 10:
            logger.info(
                f"RC6: Fitting isotonic probability calibrator on "
                f"{len(y_cal)} calibration samples "
                f"({n_cal_pos} pos / {n_cal_neg} neg)."
            )
            calibrated_model = CalibratedClassifierCV(
                model, method="sigmoid", cv="prefit"
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
            return calibrated_model
        else:
            logger.warning(
                f"RC6: Calibration set too small or imbalanced "
                f"({n_cal_pos} pos / {n_cal_neg} neg) — "
                "skipping sigmoid calibration. Returning raw model."
            )
    else:
        logger.info(
            "RC6: No calibration set provided — returning raw (uncalibrated) model. "
            "Pass X_cal/y_cal to train_model() to enable sigmoid calibration."
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

    To change the val window size, adjust VAL_WEEKS in the configuration block.
    """
    df_work = df_with_dates.copy()

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

        # FIX 2 applied here: NaT → train regardless of cutoff
        train_mask = nat_mask | (dates < cutoff)
        val_mask   = (~nat_mask) & (dates >= cutoff)

        train_idx = df_work.index[train_mask]
        val_idx   = df_work.index[val_mask]

        train_dates = dates.loc[train_idx].dropna()
        val_dates   = dates.loc[val_idx].dropna()

        logger.info(
            f"FIX 1 — Dynamic cutoff ({VAL_WEEKS}-week val window): cutoff={cutoff.date()}: "
            f"train {train_dates.min().date() if not train_dates.empty else '?'} "
            f"→ {train_dates.max().date() if not train_dates.empty else '?'}, "
            f"val {val_dates.min().date() if not val_dates.empty else '?'} "
            f"→ {val_dates.max().date() if not val_dates.empty else '?'}"
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
        _excess_pos_idx = y_val[y_val == 1].index[_target_val_pos:]  # move these to train

        if len(_excess_pos_idx) > 0:
            # Move excess val positives → train
            val_idx   = [i for i in val_idx   if i not in set(_excess_pos_idx)]
            train_idx = list(train_idx) + list(_excess_pos_idx)

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

# Weight multiplier applied to winner rows in regressor training.
# The training set is overwhelmingly non-winners (label=0, gain≈0-5%).
# Without boosting winner weights the regressor learns "predict ~5%" for
# everything and never reaches 50%+ territory.
_WINNER_WEIGHT_MULTIPLIER = 3.0    # FIX #9: Reduced from 8.0.
                                    # With ~15% positive rate, 8× was equivalent to
                                    # training on a ~59% winner dataset (8×15% / (8×15%+85%)),
                                    # causing the regressor to underfit non-winner rows.
                                    # This explains the suspiciously high gain floor (~22%)
                                    # seen in prediction logs — even the worst-ranked stock
                                    # received a high predicted gain because non-winner rows
                                    # were barely seen during training.
                                    # 3× gives ~35% effective winner rate, a more balanced
                                    # training signal while still teaching the regressor that
                                    # high-gain regimes exist.

_HIGH_GAIN_THRESHOLD  = 30.0    # Lowered from 50% — more winners qualify for the boost
_HIGH_GAIN_MULTIPLIER = 3.0     # FIX #9: Reduced from 5.0 (was 40x total, now 9x total).
                                 # 40× weight made high-gain outliers dominate the loss
                                 # surface and pushed all predictions toward the high end.


def train_gain_regressor(
    X_scaled: pd.DataFrame,           # RC3 FIX: receive pre-scaled features
    combined_df: pd.DataFrame,
    feature_names: list[str],
    client: Client,
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
    # RC1 FIX: Fetch additional gain data from ml_prediction_accuracy FIRST.
    # This must happen before the gain-column check because the accuracy
    # table is often the primary source of gain labels (the base CSV rows
    # have no gain data at all).  Previously this fetch ran after the check,
    # so the function returned early before RC1 could supply any data.
    # ------------------------------------------------------------------
    accuracy_gain_map: dict = {}
    try:
        logger.info("RC1: Fetching gain data from ml_prediction_accuracy for non-winner rows...")
        # FIX #7: Derive start_date from the earliest date in combined_df rather
        # than datetime.now() - 120 days.  Using wall-clock time caused two problems:
        #   (a) Inconsistency: the classifier uses a LOOKBACK env var (e.g. 90 days)
        #       but the gain regressor was silently fetching a wider 120-day window.
        #   (b) Future-data leakage: in backfill runs datetime.now() is today, so
        #       the query could return accuracy rows that post-date the training data.
        # Using the earliest date in combined_df anchors the window to the training
        # period, keeping both models aligned and backfill-safe.
        date_col = next(
            (c for c in ("detection_date", "explosion_date", "prediction_date", "date")
             if c in combined_df.columns),
            None,
        )
        if date_col is not None:
            try:
                start_date = pd.to_datetime(combined_df[date_col], errors="coerce").min()
                start_date = start_date.date().isoformat() if pd.notna(start_date) else None
            except Exception:
                start_date = None
        else:
            start_date = None
        if start_date is None:
            # Fallback: should not normally occur; use LOOKBACK env var if set
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

    # ------------------------------------------------------------------
    # Determine gain target column — evaluated AFTER the RC1 fetch so
    # that accuracy-table data can count toward the ≥30 threshold.
    # ------------------------------------------------------------------
    gain_col = None
    for candidate in ("actual_high_pct", "actual_gain_pct", "change_pct"):
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
            filled_count = 0
            for idx, row in combined_df.iterrows():
                if pd.notna(gain_targets[idx]):
                    continue  # already have a value

                # Support both column naming conventions:
                # T-1 rows use "symbol" + "detection_date"
                # base CSV rows use "ticker" + "event_date"
                sym = row.get("symbol") or row.get("ticker")
                if not sym or str(sym) == "nan":
                    continue
                acc_data = None

                # Try detection_date first (T-1 rows -- direct match to prediction_date)
                if has_detection:
                    d = str(row.get("detection_date", ""))[:10]
                    if d and d != "nan":
                        acc_data = accuracy_gain_map.get((sym, d))

                # Fall back to event_date - 1 business day (base CSV rows)
                if acc_data is None and has_event:
                    ev = row.get("event_date")
                    if ev and str(ev) not in ("nan", "NaT", "None"):
                        try:
                            pred_date = (
                                pd.Timestamp(ev) - pd.tseries.offsets.BDay(1)
                            ).strftime("%Y-%m-%d")
                            acc_data = accuracy_gain_map.get((sym, pred_date))
                        except Exception:
                            pass

                if acc_data:
                    val = acc_data.get("actual_high_pct") or acc_data.get("actual_gain_pct")
                    if val is not None:
                        gain_targets[idx] = float(val)
                        filled_count += 1
            logger.info(f"RC1: Filled {filled_count} additional gain targets from accuracy table")

    # ------------------------------------------------------------------
    # RC7 FIX: Raise gain cap from 500% → 10 000%.
    # The old 500% cap silently excluded the best-performing stocks from
    # training (the logs showed a max of 5329.6%).  The log transform below
    # handles the scale of extreme values; we only need to drop obvious data
    # errors (negative gains > -100% are physically impossible; gains in the
    # millions are likely bad data).
    # ------------------------------------------------------------------
    valid_gain_mask = gain_targets.notna() & (gain_targets > -100.0) & (gain_targets < 10_000.0)

    # ── FIX (Bug 3): Exclude low-gain winner noise rows from regressor training ──
    # When prev_close_db is unavailable, same-day open is used as a proxy
    # (RC2 fallback), inflating gains for gap-up stocks and producing
    # systematically wrong labels for many of the best winners.
    # Additionally, winner rows with actual_high_pct < 10% are mostly noise
    # (minor intraday moves or data errors) that pull predictions toward the
    # mean.  Non-winner rows are kept regardless of gain magnitude because
    # they are correctly labelled (gain ≈ 0-5%) and anchor the low-gain regime.
    GAIN_REGRESSOR_MIN_PCT = 5.0    # Lowered from 10% — excluding winners below 10%
                                    # was throwing away too many training examples and
                                    # compressed the upper range of predictions.
    winner_rows = combined_df["label"] == 1
    low_gain_winner_mask = valid_gain_mask & winner_rows & (gain_targets < GAIN_REGRESSOR_MIN_PCT)
    n_low_gain_excluded = int(low_gain_winner_mask.sum())
    if n_low_gain_excluded > 0:
        valid_gain_mask = valid_gain_mask & ~low_gain_winner_mask
        logger.info(
            f"Bug3 FIX: Excluded {n_low_gain_excluded} winner rows with "
            f"gain < {GAIN_REGRESSOR_MIN_PCT}% as noisy training targets. "
            f"Non-winner rows are kept regardless of gain magnitude."
        )

    n_valid = int(valid_gain_mask.sum())
    n_winners_with_gain = int((combined_df.loc[valid_gain_mask, "label"] == 1).sum()) if valid_gain_mask.any() else 0
    n_non_winners_with_gain = n_valid - n_winners_with_gain

    logger.info(
        f"\n── Training gain regressor on {n_valid} rows with gain data ──\n"
        f"  Winners:     {n_winners_with_gain}\n"
        f"  Non-winners: {n_non_winners_with_gain} (RC1: broader training set)\n"
        f"  Target:      {gain_col} (RC7: cap raised to 10 000%, log-transformed)"
    )

    if n_valid < 30:
        logger.warning(f"Only {n_valid} rows with gain data — need ≥30. Skipping gain regressor.")
        return None

    # ------------------------------------------------------------------
    # RC3 FIX: Use X_scaled (already StandardScaler-transformed), not raw
    # ------------------------------------------------------------------
    # X_scaled has the same row order as combined_df
    if len(X_scaled) != len(combined_df):
        logger.warning(
            f"RC3: X_scaled length ({len(X_scaled)}) != combined_df ({len(combined_df)}) — "
            "cannot align. Skipping gain regressor."
        )
        return None

    # Align index so we can use valid_gain_mask safely
    X_reg = X_scaled.copy()
    X_reg.index = combined_df.index
    y_reg = gain_targets.copy()
    w_reg = (
        combined_df["sample_weight"].astype(float)
        if "sample_weight" in combined_df.columns
        else pd.Series(1.0, index=combined_df.index)
    )

    # ------------------------------------------------------------------
    # RC7 FIX: Heavily up-weight winner rows, especially large-gain ones.
    # The training set has far more non-winner rows (gain ≈ 0–5%) than
    # winner rows (gain can be 50–5000%).  A 2× winner bonus is lost in
    # the noise — we need a much stronger signal for the regressor to
    # learn that high-gain stocks are a distinct regime.
    # ------------------------------------------------------------------
    winner_mask_valid = (combined_df["label"] == 1) & valid_gain_mask
    high_gain_mask = winner_mask_valid & (gain_targets >= _HIGH_GAIN_THRESHOLD)

    if winner_mask_valid.any():
        w_reg = w_reg.copy()
        w_reg[winner_mask_valid] *= _WINNER_WEIGHT_MULTIPLIER
        if high_gain_mask.any():
            w_reg[high_gain_mask] *= _HIGH_GAIN_MULTIPLIER
            logger.info(
                f"  RC7: up-weighted {winner_mask_valid.sum()} winner rows ×{_WINNER_WEIGHT_MULTIPLIER}, "
                f"{high_gain_mask.sum()} high-gain (≥{_HIGH_GAIN_THRESHOLD}%) rows ×{_WINNER_WEIGHT_MULTIPLIER * _HIGH_GAIN_MULTIPLIER:.0f} total"
            )
        else:
            logger.info(f"  RC7: up-weighted {winner_mask_valid.sum()} winner rows ×{_WINNER_WEIGHT_MULTIPLIER}")

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
        _date_col = next(
            (c for c in ["detection_date", "event_date"] if c in combined_df.columns),
            None,
        )
        if _date_col is not None:
            _dates = pd.to_datetime(
                combined_df.loc[valid_gain_mask, _date_col], errors="coerce"
            )
            _sorted_idx = _dates.sort_values(na_position="last").index
            _split_pos  = int(len(_sorted_idx) * 0.8)
            _tr_idx = _sorted_idx[:_split_pos]
            _va_idx = _sorted_idx[_split_pos:]
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
        min_child_weight=10,
        gamma=1.0,
        reg_alpha=0.5,
        reg_lambda=2.0,
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
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata → {METADATA_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



def apply_filter_aware_negative_sampling(df, logger=None):
    """
    Retraining enhancement:
    - Keep all winners.
    - Prefer non-winners that pass learned_filters.json.
    - Backfill with same-date random negatives.
    - Upweight filter-passing negatives.
    - Preserve existing mistake-learner weights.
    """
    import json
    from pathlib import Path
    import pandas as pd

    if df is None or df.empty or "label" not in df.columns:
        return df

    filters_path = Path("ml_models/learned_filters.json")
    if not filters_path.exists():
        return df

    try:
        filters = json.loads(filters_path.read_text())
    except Exception:
        return df

    winners = df[df["label"] == 1].copy()
    negatives = df[df["label"] == 0].copy()

    if negatives.empty:
        return df

    mask = pd.Series(True, index=negatives.index)

    filter_map = [
        ("close", "min_price", "max_price"),
        ("price", "min_price", "max_price"),
        ("volume", "min_volume", None),
        ("hv10", "min_hv10", None),
        ("hv20", "min_hv20", None),
        ("relative_volume", "min_relative_volume", None),
        ("volume_ratio", "min_volume_ratio", None),
    ]

    used_filter=False
    for col, min_key, max_key in filter_map:
        if col not in negatives.columns:
            continue

        if min_key and min_key in filters:
            mask &= negatives[col].fillna(-1e9) >= filters[min_key]
            used_filter=True

        if max_key and max_key in filters:
            mask &= negatives[col].fillna(1e9) <= filters[max_key]
            used_filter=True

    if not used_filter:
        return df

    hard_neg = negatives[mask].copy()
    easy_neg = negatives[~mask].copy()

    date_col = next(
        (c for c in ["detection_date", "event_date", "trade_date", "date"]
         if c in df.columns),
        None
    )

    if date_col is None:
        selected_neg = hard_neg.copy()
    else:
        selected_parts = []

        for dt, winner_group in winners.groupby(date_col):
            target_negatives = max(len(winner_group) * 4, 8)

            hard_dt = hard_neg[hard_neg[date_col] == dt]
            easy_dt = easy_neg[easy_neg[date_col] == dt]

            preferred_target = int(target_negatives * 0.80)

            chosen_hard = (
                hard_dt.sample(min(len(hard_dt), preferred_target), random_state=42)
                if len(hard_dt) else hard_dt
            )

            remaining = target_negatives - len(chosen_hard)

            chosen_easy = (
                easy_dt.sample(min(len(easy_dt), remaining), random_state=42)
                if remaining > 0 and len(easy_dt) else easy_dt.iloc[0:0]
            )

            selected_parts.append(chosen_hard)
            selected_parts.append(chosen_easy)

        selected_neg = (
            pd.concat(selected_parts, ignore_index=True)
            if selected_parts else hard_neg
        )

    if "sample_weight" in selected_neg.columns:
        selected_neg.loc[:, "sample_weight"] = (
            selected_neg["sample_weight"].fillna(1.0) * 1.75
        )

    result = pd.concat([winners, selected_neg], ignore_index=True)

    if logger:
        logger.info(
            f"Filter-aware retraining active: "
            f"winners={len(winners)}, "
            f"hard_negatives_available={len(hard_neg)}, "
            f"selected_negatives={len(selected_neg)}"
        )

    return result



def main() -> int:
    logger.info("=" * 60)
    logger.info("ML RETRAIN — FULL RETRAIN FROM SCRATCH")
    logger.info("=" * 60)

    # ── Connect ──────────────────────────────────────────────────────────────
    client = get_supabase_client()

    # ── Load standard training data ───────────────────────────────────────────
    base_df     = load_base_training_data(client)
    t1_df       = load_t1_data(client)
    combined_df = combine_datasets(base_df, t1_df)
    combined_df = apply_filter_aware_negative_sampling(combined_df, logger)

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

                symbol_col = next(
                    (c for c in ["symbol", "ticker"] if c in combined_df.columns), None
                )
                date_col = next(
                    (c for c in ["detection_date_x", "detection_date", "event_date"]
                     if c in combined_df.columns), None
                )

                if symbol_col and date_col:
                    gain_cols = ["symbol", "detection_date", "actual_high_pct"]
                    if "change_pct" in winners_corrected.columns:
                        gain_cols.append("change_pct")

                    combined_df = combined_df.merge(
                        winners_corrected[gain_cols],
                        left_on=[symbol_col, date_col],
                        right_on=["symbol", "detection_date"],
                        how="left",
                        suffixes=("", "_winners"),
                    ).drop(columns=["detection_date_winners"], errors="ignore")

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
        logger.warning(f"RC2: Could not fetch/process gain data: {e} — gain regressor may be limited")

    # ── FIX 4: Relabel rows with strong intraday moves as winners ─────────────
    combined_df = apply_intraday_high_labels(combined_df, threshold=INTRADAY_WIN_THRESHOLD)

    # ── Load mistake samples and append AFTER combine_datasets ───────────────
    mistake_df = pd.DataFrame()
    if MISTAKE_LEARNER_AVAILABLE:
        logger.info("\n" + "=" * 60)
        logger.info("MISTAKE LEARNING STEP")
        logger.info("=" * 60)

        proto_features = [
            c for c in combined_df.columns
            if c not in NON_FEATURE_COLS and not c.startswith("Unnamed")
        ]

        # Load multiday tables so mistake rows get t3_/t5_/t10_ features,
        # matching the enrichment applied to all regular T-1 rows in load_t1_data().
        # Without this, mistake samples land in combined_df with every multiday
        # feature as NaN while still carrying 3×/2× sample weights — giving the
        # model a strong but half-blind corrective signal.
        logger.info("Loading multiday tables for mistake-sample enrichment...")
        _mistake_winners_md, _mistake_non_winners_md = load_multiday_data(client)

        mistake_df = build_mistake_training_samples(
            lookback_days=90,
            use_all_timepoints=True,
            existing_features=proto_features,
            winners_multiday=_mistake_winners_md,
            non_winners_multiday=_mistake_non_winners_md,
        )

        if not mistake_df.empty:
            # RC6 FIX: Enrich mistake rows with actual gain data BEFORE appending
            # so they contribute to gain regressor training
            mistake_df = enrich_mistakes_with_gains(mistake_df, client)

            log_mistake_summary(mistake_df)
            combined_df = pd.concat([combined_df, mistake_df],
                                    ignore_index=True, sort=False)
            logger.info(
                f"Dataset after adding mistakes: {len(combined_df)} rows "
                f"(+{len(mistake_df)} mistake samples)"
            )
        else:
            logger.info("No mistake samples to add this run.")
    else:
        logger.warning("ml_mistake_learner not available — skipping mistake-learning step.")

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
    scaler, X_train = build_scaler(X_train_raw)                    # fit + transform train
    X_val           = scale_with_fitted_scaler(scaler, X_val_raw)  # transform val only

    # Reassemble a full scaled DataFrame (train + val, original row order) kept
    # for any downstream use that genuinely needs all rows.
    X_scaled = pd.concat([X_train, X_val]).loc[X.index]  # restore original row order

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
    # Minimum requirements: ≥10 positives in each half after the split.
    CAL_MIN_POS = 10
    X_cal_fit, y_cal_fit = None, None
    X_val_xgb, y_val_xgb = X_val, y_val

    _val_pos_idx  = y_val[y_val == 1].index.tolist()
    _val_neg_idx  = y_val[y_val == 0].index.tolist()
    _n_cal_pos    = len(_val_pos_idx) // 2
    _n_cal_neg    = len(_val_neg_idx) // 2

    if _n_cal_pos >= CAL_MIN_POS and _n_cal_neg >= CAL_MIN_POS:
        # Stratified split of val: first half → calibration, second half → early-stop
        _cal_idx  = _val_pos_idx[:_n_cal_pos]  + _val_neg_idx[:_n_cal_neg]
        _stop_idx = _val_pos_idx[_n_cal_pos:]  + _val_neg_idx[_n_cal_neg:]

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
    logger.info("GAIN REGRESSOR TRAINING (RC1+RC2+RC3+RC6+RC7 fixes applied)")
    logger.info("=" * 60)
    gain_regressor = train_gain_regressor(
        X_scaled=X_train,                           # train rows only — no val-period leakage
        combined_df=combined_df.loc[train_idx],     # matching train-split rows
        feature_names=feature_names,
        client=client,                              # RC1: fetch additional gain data
    )

    # ── Evaluate classifier ───────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, classification_report

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_val, val_proba)
        logger.info(f"Validation AUC-ROC: {auc:.4f}")
    except Exception:
        auc = float("nan")
        logger.warning("Validation AUC-ROC: nan (only one class in val set)")

    logger.info("Classification report (val):")
    for line in classification_report(y_val, val_pred).split("\n"):
        if line.strip():
            logger.info(f"  {line}")

    # Log probability distribution on val set
    val_proba_series = pd.Series(val_proba)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = pd.cut(val_proba_series, bins=bins).value_counts().sort_index()
    logger.info("Val set probability distribution:")
    for bucket, count in dist.items():
        logger.info(f"  {str(bucket):<20} {count:>4}")

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
        "base_sample_weight":      BASE_CSV_WEIGHT,
        "t1_sample_weight":        T1_WEIGHT,
        "intraday_win_threshold":  INTRADAY_WIN_THRESHOLD,
        "equal_weight_applied": (
            len(t1_df) >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT
            if not t1_df.empty else False
        ),
        "gain_regressor_trained":  gain_regressor is not None,
        "gain_regressor_rc_fixes": ["RC1_broad_training", "RC2_prev_close",
                                    "RC3_scaled_input", "RC6_mistake_enrichment", "RC7_log_transform_heavy_weights"],
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(model, scaler, fi_df, feature_names, training_stats, gain_regressor)

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
    logger.info(f"  Validation AUC      : {auc:.4f}")
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
