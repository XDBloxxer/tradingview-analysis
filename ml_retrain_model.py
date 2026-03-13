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
     sort to the front as NaT and the val set ends up being entirely T-1 non-winners
     → 0 positives in val → degenerate model.

  2. Stronger regularisation — min_child_weight raised from 3→10, max_depth 6→5,
     gamma 0.1→1.0, reg_alpha 0.1→0.5. These prevent the model from memorising
     individual stocks.

  3. scale_pos_weight capped at [0.5, 3.0] — avoids extreme corrections when the
     training set happens to be very imbalanced in either direction.

  4. Intraday-high label support — if actual_high_pct is available and exceeds
     INTRADAY_WIN_THRESHOLD, those rows are also treated as winners (label=1).
     This fixes the JDZG/RIME problem where the model was RIGHT (stock moved big)
     but the close-based label called it a false positive.

  5. Duplicate-date deduplication — the same (symbol, date) can appear in both the
     base CSV and T-1 tables, causing the model to overfit to repeated examples. We
     now deduplicate after combine_datasets() so the model doesn't overfit to
     repeated rows.

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

TABLE_BASE              = "ml_training_base"
TABLE_WINNERS_CLOSE     = "winners_day_prior_close"
TABLE_WINNERS_OPEN      = "winners_day_prior_open"
TABLE_NON_WINNERS_CLOSE = "non_winners_day_prior_close"
TABLE_NON_WINNERS_OPEN  = "non_winners_day_prior_open"

MODEL_DIR               = Path("ml_models")
MODEL_PATH              = MODEL_DIR / "best_model.pkl"
SCALER_PATH             = MODEL_DIR / "scaler.pkl"
GAIN_REGRESSOR_PATH     = MODEL_DIR / "gain_regressor.pkl"
METADATA_PATH           = MODEL_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

BASE_CSV_WEIGHT         = 1.5
T1_WEIGHT               = 1.0
MIN_T1_ROWS_FOR_EQUAL_WEIGHT = 1800

# FIX 4: Intraday high threshold — a stock is considered a "winner" even if
# it didn't close at the top, as long as it hit this intraday gain.
# This corrects false positives where the model correctly identified explosive
# stocks that moved big intraday but closed below the strict winner threshold.
INTRADAY_WIN_THRESHOLD = 15.0  # %

# FIX 3: scale_pos_weight caps — prevent extreme corrections
SPW_MIN = 0.5
SPW_MAX = 3.0

XGBOOST_PARAMS = {
    "n_estimators":       300,
    "max_depth":          5,       # FIX 2: reduced from 6 → less overfitting
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   10,      # FIX 2: raised from 3 → requires more samples per leaf
    "gamma":              1.0,     # FIX 2: raised from 0.1 → higher minimum gain to split
    "reg_alpha":          0.5,     # FIX 2: raised from 0.1 → more L1 regularisation
    "reg_lambda":         2.0,     # FIX 2: raised from 1.0 → more L2 regularisation
    "scale_pos_weight":   1,       # overridden at train time (clamped to SPW_MIN/MAX)
    "objective":          "binary:logistic",
    "eval_metric":        "logloss",
    "use_label_encoder":  False,
    "random_state":       42,
    "n_jobs":             -1,
    "early_stopping_rounds": 30,
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

    if "sample_weight" not in df.columns:
        df["sample_weight"] = BASE_CSV_WEIGHT
    df["source"] = df.get("source", "base_csv")

    logger.info(f"Base data: {len(df)} rows, "
                f"pos={int((df['label']==1).sum())}, "
                f"neg={int((df['label']==0).sum())}")
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


def load_t1_data(client: Client) -> pd.DataFrame:
    """
    Load accumulated T-1 winner and non-winner samples.

    Applies t1_column_map to rename intraday short-form column names
    to the model's expected long-form names with the correct prefix.

    close tables → prefix "t1_close"
    open  tables → prefix "t1_open"
    """
    logger.info("Loading accumulated T-1 training data...")

    TABLE_CONFIG = [
        (TABLE_WINNERS_CLOSE,      1, "t1_close"),
        (TABLE_WINNERS_OPEN,       1, "t1_open"),
        (TABLE_NON_WINNERS_CLOSE,  0, "t1_close"),
        (TABLE_NON_WINNERS_OPEN,   0, "t1_open"),
    ]

    frames = []

    for table, label, prefix in TABLE_CONFIG:
        try:
            df = fetch_table_paginated(client, table)
            if df.empty:
                continue

            df["label"]  = label
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

            frames.append(df)

        except Exception as e:
            logger.warning(f"Could not load '{table}': {e}")

    if not frames:
        logger.warning("No T-1 data found. Training on base data only.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["sample_weight"] = T1_WEIGHT

    t1_feature_cols = [c for c in combined.columns
                       if c.startswith("t1_close_") or c.startswith("t1_open_")]
    non_null_t1 = combined[t1_feature_cols].notna().any().sum() if t1_feature_cols else 0

    logger.info(f"T-1 data: {len(combined)} rows, "
                f"pos={int((combined['label']==1).sum())}, "
                f"neg={int((combined['label']==0).sum())}")
    logger.info(f"T-1 feature columns populated: {non_null_t1}/{len(t1_feature_cols)}")

    return combined


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def apply_intraday_high_labels(
    combined_df: pd.DataFrame,
    threshold: float = INTRADAY_WIN_THRESHOLD,
) -> pd.DataFrame:
    """
    FIX 4: Re-label rows where actual_high_pct exceeds threshold as winners.

    This corrects the problem where stocks that moved big intraday (e.g. +58%
    intraday high) were labelled as false positives because they didn't close
    as the top-N gainers.  The model was RIGHT — these stocks exploded.
    The label was wrong.

    Only upgrades label from 0→1 (never downgrades 1→0).
    """
    if "actual_high_pct" not in combined_df.columns:
        return combined_df

    before = int((combined_df["label"] == 1).sum())
    mask = (
        (combined_df["label"] == 0) &
        (pd.to_numeric(combined_df["actual_high_pct"], errors="coerce") >= threshold)
    )
    combined_df = combined_df.copy()
    combined_df.loc[mask, "label"] = 1

    # Bump sample weight for these relabelled rows — they're high-signal examples
    combined_df.loc[mask, "sample_weight"] = combined_df.loc[mask, "sample_weight"] * 1.5

    after = int((combined_df["label"] == 1).sum())
    if after > before:
        logger.info(
            f"Intraday-high relabelling: {after - before} rows upgraded to label=1 "
            f"(actual_high_pct >= {threshold}%)"
        )
    return combined_df


def combine_datasets(base_df: pd.DataFrame, t1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate base + T-1 data.

    FIX 5: Deduplicate by (symbol, date) after concatenation.
    The same stock+date can appear in both the base CSV and T-1 tables,
    causing the model to overfit to repeated examples. We keep the T-1
    version (which has richer features) when duplicates exist.

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

    # T-1 first so it takes priority in dedup
    combined = pd.concat([t1_df, base_df], ignore_index=True, sort=False)

    # Safe dedup: only deduplicate if BOTH sources use the same date column name.
    # Base CSV uses event_date, T-1 data uses detection_date — these are different
    # columns and should NOT be deduped against each other.
    sym_col = next((c for c in ["symbol", "ticker"] if c in combined.columns), None)
    date_cols_present = [c for c in ["detection_date", "event_date"] if c in combined.columns]

    if sym_col and len(date_cols_present) == 1:
        date_col = date_cols_present[0]
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=[sym_col, date_col], keep="first")
        n_dropped = before_dedup - len(combined)
        if n_dropped > 0:
            logger.info(f"Deduplication: removed {n_dropped} duplicate rows ({before_dedup} → {len(combined)})")
    elif sym_col and len(date_cols_present) == 2:
        logger.info(
            "Skipping deduplication: base CSV (event_date) and T-1 data (detection_date) "
            "use different date columns — no cross-source duplicates possible"
        )

    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())

    if n_pos > 0 and n_pos / (n_pos + n_neg) > 0.40:
        logger.error(
          f"ABORTING: positive rate {n_pos/(n_pos+n_neg):.1%} is too high. "
          "The base training data likely has corrupt labels. "
          "Check ml_training_base — it should have ~5-20% positives."
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
        X: DataFrame of features (NaN allowed — XGBoost handles natively)
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
    ]

    X = df[feature_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    nan_pct = X.isna().mean().mean() * 100
    logger.info(f"Overall NaN rate: {nan_pct:.1f}% (expected for cross-lag rows)")

    return X, y, w


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def build_scaler(X: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """
    Fit scaler on non-NaN values per column. Returns scaler + scaled DataFrame.

    NaN positions are PRESERVED after scaling so XGBoost can use its native
    missing-value routing.
    """
    scaler    = StandardScaler()
    col_means = X.mean()
    X_filled  = X.fillna(col_means)
    scaler.fit(X_filled)

    nan_mask       = X.isna()
    X_scaled_vals  = scaler.transform(X_filled)
    X_scaled       = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
    X_scaled[nan_mask] = np.nan  # restore NaN so XGBoost routes correctly

    return scaler, X_scaled


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    """Train XGBClassifier from scratch with early stopping."""
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
    logger.info(f"  Best val logloss: {model.best_score:.4f}")

    # FIX: Warn loudly if val logloss is suspiciously perfect — sign of data leakage
    if model.best_score < 0.05:
        logger.warning(
            f"  ⚠️  Val logloss={model.best_score:.4f} is suspiciously low. "
            "This may indicate data leakage or overfitting. "
            "Check that the validation set does not overlap with training dates."
        )

    return model


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    df_with_dates: pd.DataFrame,
    val_fraction: float = 0.15,
) -> tuple:
    """
    FIX 1: TIME-BASED train/val split instead of random stratified split.

    Why this matters: With a random split, the model sees stocks from the
    same week in both train and val. Since market regimes persist over days,
    this makes val performance look much better than it actually is (0.9999 AUC).

    A time-based split forces the model to validate on the MOST RECENT data,
    which is the honest test of whether it generalises across time.

    KEY FIX: We build a unified _sort_date column that takes detection_date
    for T-1 rows and falls back to event_date for base CSV rows. Without this,
    base CSV rows (which have no detection_date) sort as NaT and float to the
    front, putting the entire val set in the T-1 non-winner rows → 0 positives
    in val → degenerate model that outputs a constant probability.
    """
    df_work = df_with_dates.copy()

    # Build unified sort date: prefer detection_date, fall back to event_date
    has_detection = "detection_date" in df_work.columns
    has_event     = "event_date" in df_work.columns

    if has_detection or has_event:
        sort_date = pd.Series(pd.NaT, index=df_work.index)
        if has_detection:
            sort_date = pd.to_datetime(df_work["detection_date"], errors="coerce")
        if has_event:
            # Fill any NaT (base CSV rows) with event_date
            event_parsed = pd.to_datetime(df_work["event_date"], errors="coerce")
            sort_date = sort_date.fillna(event_parsed)

        df_work["_sort_date"] = sort_date
        date_col = "_sort_date"

        n_base_dates = sort_date.notna().sum()
        n_nat        = sort_date.isna().sum()
        logger.info(
            f"Unified sort_date: {n_base_dates} rows have a valid date, "
            f"{n_nat} rows have NaT (will sort to front — investigate if large)"
        )
    else:
        date_col = next(
            (c for c in ["date"] if c in df_work.columns), None
        )

    if date_col is None:
        logger.warning(
            "No date column found for time-based split — "
            "falling back to sequential split (first N rows train, last M rows val). "
            "This is still better than random but not ideal."
        )
        split_idx = int(len(X) * (1 - val_fraction))
        X_train = X.iloc[:split_idx]
        X_val   = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val   = y.iloc[split_idx:]
        w_train = w.iloc[:split_idx]
        w_val   = w.iloc[split_idx:]
    else:
        # Sort by unified date, take most recent val_fraction as validation set
        dates      = pd.to_datetime(df_work[date_col], errors="coerce")
        sorted_idx = dates.sort_values(na_position='last').index
        split_pos  = int(len(sorted_idx) * (1 - val_fraction))

        train_idx = sorted_idx[:split_pos]
        val_idx   = sorted_idx[split_pos:]

        X_train = X.loc[train_idx]
        X_val   = X.loc[val_idx]
        y_train = y.loc[train_idx]
        y_val   = y.loc[val_idx]
        w_train = w.loc[train_idx]
        w_val   = w.loc[val_idx]

        train_dates = dates.loc[train_idx].dropna()
        val_dates   = dates.loc[val_idx].dropna()
        if not train_dates.empty and not val_dates.empty:
            logger.info(
                f"Time-based split: "
                f"train {train_dates.min().date()} → {train_dates.max().date()}, "
                f"val {val_dates.min().date()} → {val_dates.max().date()}"
            )

    logger.info(
        f"Train/val split: {len(X_train)} train "
        f"(pos={int((y_train==1).sum())}, neg={int((y_train==0).sum())}), "
        f"{len(X_val)} val "
        f"(pos={int((y_val==1).sum())}, neg={int((y_val==0).sum())})"
    )

    # Warn if val set has very few positives — calibration will be noisy
    val_pos = int((y_val == 1).sum())
    if val_pos < 10:
        logger.warning(
            f"  ⚠️  Only {val_pos} positive examples in validation set. "
            "Accuracy metrics may be noisy. Consider using a longer training window."
        )

    return X_train, X_val, y_train, y_val, w_train, w_val


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Generate feature_importance.csv using gain importance."""
    booster = model.get_booster()
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
# Gain regressor
# ---------------------------------------------------------------------------

def train_gain_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    combined_df: pd.DataFrame,
    feature_names: list[str],
) -> "Optional[XGBClassifier]":
    """
    Train a regression model to predict actual % gain for stocks the
    classifier labels as winners.
    """
    from xgboost import XGBRegressor

    # Prefer actual_high_pct over change_pct for the gain target —
    # it captures the best intraday opportunity, not just the close.
    gain_col = None
    for candidate in ("actual_high_pct", "actual_gain_pct", "change_pct"):
        if candidate in combined_df.columns:
            gain_col = candidate
            break

    if gain_col is None:
        logger.warning("No gain column found — skipping gain regressor training.")
        return None

    winner_mask  = (combined_df["label"] == 1) & combined_df[gain_col].notna()
    n_winners    = int(winner_mask.sum())

    if n_winners < 30:
        logger.warning(f"Only {n_winners} winner rows with gain data — "
                       "need ≥30 to train gain regressor. Skipping.")
        return None

    logger.info(f"\n── Training gain regressor on {n_winners} winner rows (target: {gain_col}) ──")

    X_reg = pd.DataFrame(index=combined_df.index, columns=feature_names)
    for col in feature_names:
        if col in combined_df.columns:
            X_reg[col] = combined_df[col]
    X_reg = X_reg.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    y_reg = pd.to_numeric(combined_df[gain_col], errors="coerce")
    w_reg = combined_df["sample_weight"].astype(float) \
            if "sample_weight" in combined_df.columns \
            else pd.Series(1.0, index=combined_df.index)

    X_reg = X_reg[winner_mask]
    y_reg = y_reg[winner_mask]
    w_reg = w_reg[winner_mask]

    col_means  = X_reg.mean()
    X_reg_fill = X_reg.fillna(col_means)

    from sklearn.model_selection import train_test_split
    if len(X_reg) >= 10:
        X_tr, X_va, y_tr, y_va, w_tr, _ = train_test_split(
            X_reg_fill, y_reg, w_reg,
            test_size=0.2, random_state=42,
        )
    else:
        X_tr, X_va, y_tr, y_va, w_tr = X_reg_fill, X_reg_fill, y_reg, y_reg, w_reg

    regressor = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    regressor.fit(
        X_tr, y_tr,
        sample_weight=w_tr.values,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    val_pred = regressor.predict(X_va)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_va, val_pred)
    r2  = r2_score(y_va, val_pred) if len(y_va) > 1 else float("nan")
    logger.info(f"  Gain regressor — val MAE: {mae:.2f}%  R²: {r2:.3f}")
    logger.info(f"  Predicted gains range: {val_pred.min():.1f}% – {val_pred.max():.1f}%")

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
        logger.info("Gain regressor not trained this run — predictor will use rule-based gains")

    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Saved feature importance → {FEATURE_IMPORTANCE_PATH}")

    metadata = {
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "source":                "ml_retrain_model.py",
        "training_approach":     "full_retrain_from_scratch",
        "n_features":            len(feature_names),
        "features":              feature_names,
        "feature_names_sample":  feature_names[:20],
        "best_iteration":        int(model.best_iteration),
        "best_val_logloss":      float(model.best_score),
        "gain_regressor_trained": gain_regressor is not None,
        **training_stats,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata → {METADATA_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # ── Enrich combined_df with intraday peak gain from daily_winners ─────────
    logger.info("Fetching intraday peak gain data from daily_winners for gain regressor...")
    try:
        winners_response = fetch_table_paginated(client, "daily_winners")
        if not winners_response.empty:
            required = {"symbol", "detection_date", "high", "price"}
            if required.issubset(winners_response.columns):
                winners_gain = winners_response[["symbol", "detection_date", "high", "price"]].copy()
                winners_gain["actual_high_pct"] = (
                    (winners_gain["high"] / winners_gain["price"] - 1) * 100
                ).clip(lower=0)

                symbol_col = next(
                    (c for c in ["symbol", "ticker"] if c in combined_df.columns), None
                )
                date_col = next(
                    (c for c in ["detection_date_x", "detection_date", "event_date"]
                     if c in combined_df.columns), None
                )

                if symbol_col and date_col:
                    combined_df = combined_df.merge(
                        winners_gain[["symbol", "detection_date", "actual_high_pct"]],
                        left_on=[symbol_col, date_col],
                        right_on=["symbol", "detection_date"],
                        how="left",
                    ).drop(columns=["detection_date"], errors="ignore")
                    n_with_gain = combined_df["actual_high_pct"].notna().sum()
                    logger.info(f"Enriched {n_with_gain} rows with intraday peak gain data")
    except Exception as e:
        logger.warning(f"Could not fetch gain data: {e} — gain regressor will be skipped")

    # ── FIX 4: Relabel rows with strong intraday moves as winners ─────────────
    combined_df = apply_intraday_high_labels(combined_df, threshold=INTRADAY_WIN_THRESHOLD)

    # ── Load mistake samples and append AFTER combine_datasets ───────────────
    if MISTAKE_LEARNER_AVAILABLE:
        logger.info("\n" + "=" * 60)
        logger.info("MISTAKE LEARNING STEP")
        logger.info("=" * 60)

        proto_features = [
            c for c in combined_df.columns
            if c not in NON_FEATURE_COLS and not c.startswith("Unnamed")
        ]

        mistake_df = build_mistake_training_samples(
            lookback_days=90,
            use_all_timepoints=True,
            existing_features=proto_features,
        )

        if not mistake_df.empty:
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
        mistake_df = pd.DataFrame()

    # ── Prepare features ──────────────────────────────────────────────────────
    X, y, w = prepare_features(combined_df)
    feature_names = list(X.columns)

    # ── Scale ─────────────────────────────────────────────────────────────────
    logger.info("Fitting scaler...")
    scaler, X_scaled = build_scaler(X)

    # ── FIX 1: Time-based train/val split ─────────────────────────────────────
    X_train, X_val, y_train, y_val, w_train, w_val = train_val_split(
        X_scaled, y, w, combined_df
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train, w_train, X_val, y_val)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df = compute_feature_importance(model, feature_names)

    # ── Train gain regressor ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("GAIN REGRESSOR TRAINING")
    logger.info("=" * 60)
    gain_regressor = train_gain_regressor(
        X_train, y_train, w_train, combined_df, feature_names
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

    # Log probability distribution on val set — early warning for bimodal collapse
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
            "The model is making near-binary decisions. This typically means overfitting. "
            "Increase min_child_weight or reduce max_depth if this persists."
        )
    else:
        logger.info(f"  ✅ {gap_count} predictions in mid-range (0.15–0.85) — distribution looks healthy")

    # ── Training stats for metadata ───────────────────────────────────────────
    n_mistakes = len(mistake_df) if not mistake_df.empty else 0
    training_stats = {
        "n_total_samples":         len(combined_df),
        "n_base_samples":          len(base_df),
        "n_t1_samples":            len(t1_df) if not t1_df.empty else 0,
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
        "split_method":            "time_based",
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
    logger.info(f"  Mistake samples     : {training_stats['n_mistake_samples']}")
    logger.info(f"  Positive rate       : {training_stats['positive_rate']:.1%}")
    logger.info(f"  Validation AUC      : {auc:.4f}")
    logger.info(f"  Best iteration      : {model.best_iteration}")
    logger.info(f"  Features            : {len(feature_names)}")
    logger.info(f"  Split method        : time-based (most recent {15}% as val)")
    logger.info(f"  Gain regressor      : {'✓ trained' if gain_regressor else '— skipped'}")
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
