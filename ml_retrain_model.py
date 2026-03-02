#!/usr/bin/env python3
"""
ml_retrain_model.py  —  Weekly Full Retrain From Scratch

FIXES IN THIS VERSION (2026-03-02 v2):

FIX 1 — scale_pos_weight capped too low (was [0.5, 3.0]):
  New cap: [1.0, 30.0].

FIX 2 — Regularisation slightly relaxed for t3_-only prediction:
  min_child_weight=5, gamma=0.3.

FIX 3 — Gain regressor date merge was broken:
  The previous code tried to merge combined_df (which has event_date from base
  CSV) against daily_winners (which has detection_date). These date columns
  name different things. The merge used left_on=[symbol_col, date_col] where
  date_col was usually 'event_date', so it never matched detection_date in
  winners_gain → near-zero match rate → <30 winners → regressor never trained.

  Fix: Enrich ONLY the T-1 rows (which have detection_date) separately from
  the base CSV rows (which have event_date). Use the correct date column for
  each source. Then recombine before passing to train_gain_regressor().

FIX 4 — Gain regressor index mismatch:
  train_gain_regressor() received X_train (a time-split subset of combined_df)
  but built winner_mask on the full combined_df. The indices don't align, so
  X_reg[winner_mask] silently selected the wrong rows or crashed.

  Fix: train_gain_regressor() now takes the FULL X_scaled DataFrame and
  combined_df together, builds winner_mask on combined_df, then aligns X_scaled
  using combined_df's index before subsetting.

FIX 5 — Gain regressor scale inconsistency:
  The gain regressor was trained on raw (unscaled) features by filling NaN with
  col_means of the raw X_reg. But at prediction time, predict_with_targets()
  calls regressor.predict(X_scaled) — the StandardScaler-transformed features.
  Training on raw features and predicting on scaled ones produces garbage.

  Fix: train_gain_regressor() now receives X_scaled_full (the complete scaled
  feature matrix) and trains on that directly. No separate NaN-filling needed
  because scaling already used the scaler mean as fill value.
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

try:
    from t1_column_map import rename_t1_columns
    T1_MAP_AVAILABLE = True
except ImportError:
    T1_MAP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "t1_column_map.py not found — T-1 features will not be renamed."
    )

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

INTRADAY_WIN_THRESHOLD = 15.0  # %

SPW_MIN = 1.0
SPW_MAX = 30.0

XGBOOST_PARAMS = {
    "n_estimators":       300,
    "max_depth":          5,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,
    "gamma":              0.3,
    "reg_alpha":          0.3,
    "reg_lambda":         1.5,
    "scale_pos_weight":   1,
    "objective":          "binary:logistic",
    "eval_metric":        "logloss",
    "use_label_encoder":  False,
    "random_state":       42,
    "n_jobs":             -1,
    "early_stopping_rounds": 30,
}

NON_FEATURE_COLS = {
    "id", "created_at", "updated_at", "date", "symbol", "ticker",
    "label", "source", "sample_weight", "detection_date", "explosion_date",
    "change_pct", "rank", "notes", "mistake_type", "actual_gain_pct",
    "actual_high_pct", "_sort_date",
    "gain_pct", "volume_spike",
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


def load_t1_data(client: Client) -> pd.DataFrame:
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
    if "actual_high_pct" not in combined_df.columns:
        return combined_df

    before = int((combined_df["label"] == 1).sum())
    mask = (
        (combined_df["label"] == 0) &
        (pd.to_numeric(combined_df["actual_high_pct"], errors="coerce") >= threshold)
    )
    combined_df = combined_df.copy()
    combined_df.loc[mask, "label"] = 1
    combined_df.loc[mask, "sample_weight"] = combined_df.loc[mask, "sample_weight"] * 1.5

    after = int((combined_df["label"] == 1).sum())
    if after > before:
        logger.info(
            f"Intraday-high relabelling: {after - before} rows upgraded to label=1 "
            f"(actual_high_pct >= {threshold}%)"
        )
    return combined_df


def combine_datasets(base_df: pd.DataFrame, t1_df: pd.DataFrame) -> pd.DataFrame:
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

    combined = pd.concat([t1_df, base_df], ignore_index=True, sort=False)

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
            "use different date columns"
        )

    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())
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

    raw_ratio = n_neg / n_pos if n_pos > 0 else float("inf")
    clamped   = max(SPW_MIN, min(SPW_MAX, raw_ratio))
    logger.info(
        f"Class balance: {n_pos} pos / {n_neg} neg → "
        f"natural ratio {raw_ratio:.1f} → "
        f"scale_pos_weight will be clamped to [{SPW_MIN}, {SPW_MAX}] = {clamped:.1f}"
    )

    return combined


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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
    scaler    = StandardScaler()
    col_means = X.mean()
    X_filled  = X.fillna(col_means)
    scaler.fit(X_filled)

    nan_mask       = X.isna()
    X_scaled_vals  = scaler.transform(X_filled)
    X_scaled       = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
    X_scaled[nan_mask] = np.nan

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
    params = XGBOOST_PARAMS.copy()

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0 and n_neg > 0:
        raw_spw = n_neg / n_pos
        clamped_spw = max(SPW_MIN, min(SPW_MAX, raw_spw))
        params["scale_pos_weight"] = round(clamped_spw, 3)
        logger.info(
            f"  scale_pos_weight: raw={raw_spw:.3f} → clamped to {clamped_spw:.3f} "
            f"(limits: [{SPW_MIN}, {SPW_MAX}])"
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

    if model.best_score < 0.05:
        logger.warning(
            f"  ⚠️  Val logloss={model.best_score:.4f} is suspiciously low. "
            "This may indicate data leakage or overfitting."
        )

    return model


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    df_with_dates: pd.DataFrame,
    val_fraction: float = 0.15,
) -> tuple:
    df_work = df_with_dates.copy()

    has_detection = "detection_date" in df_work.columns
    has_event     = "event_date" in df_work.columns

    if has_detection or has_event:
        sort_date = pd.Series(pd.NaT, index=df_work.index)
        if has_detection:
            sort_date = pd.to_datetime(df_work["detection_date"], errors="coerce")
        if has_event:
            event_parsed = pd.to_datetime(df_work["event_date"], errors="coerce")
            sort_date = sort_date.fillna(event_parsed)

        df_work["_sort_date"] = sort_date
        date_col = "_sort_date"

        n_base_dates = sort_date.notna().sum()
        n_nat        = sort_date.isna().sum()
        logger.info(
            f"Unified sort_date: {n_base_dates} rows have a valid date, "
            f"{n_nat} rows have NaT"
        )
    else:
        date_col = next(
            (c for c in ["date"] if c in df_work.columns), None
        )

    if date_col is None:
        logger.warning(
            "No date column found for time-based split — "
            "falling back to sequential split."
        )
        split_idx = int(len(X) * (1 - val_fraction))
        X_train = X.iloc[:split_idx]
        X_val   = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val   = y.iloc[split_idx:]
        w_train = w.iloc[:split_idx]
        w_val   = w.iloc[split_idx:]
    else:
        dates      = pd.to_datetime(df_work[date_col], errors="coerce")
        sorted_idx = dates.sort_values().index
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

    val_pos = int((y_val == 1).sum())
    if val_pos < 10:
        logger.warning(
            f"  ⚠️  Only {val_pos} positive examples in validation set. "
            "Accuracy metrics may be noisy."
        )

    return X_train, X_val, y_train, y_val, w_train, w_val


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
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
# FIX 3+4+5: Gain regressor — fixed date enrichment, index alignment, scaling
# ---------------------------------------------------------------------------

def enrich_t1_data_with_gain(t1_df: pd.DataFrame, client: Client) -> pd.DataFrame:
    """
    FIX 3: Enrich T-1 rows with actual_high_pct using detection_date (correct).

    T-1 rows have detection_date (the day the winner was detected).
    daily_winners also uses detection_date.
    This merge is correct; the old code used event_date which is a base CSV column.

    Only runs on T-1 rows (those with detection_date). Base CSV rows are enriched
    separately by main() if they happen to have matching event_dates.
    """
    if t1_df.empty or "detection_date" not in t1_df.columns:
        return t1_df

    logger.info("Fetching intraday peak gain data for T-1 winners...")
    try:
        winners_response = fetch_table_paginated(client, "daily_winners")
        if winners_response.empty:
            logger.warning("  daily_winners table is empty — gain regressor may not train")
            return t1_df

        required = {"symbol", "detection_date", "high", "price"}
        if not required.issubset(winners_response.columns):
            logger.warning(f"  daily_winners missing columns: {required - set(winners_response.columns)}")
            return t1_df

        winners_gain = winners_response[["symbol", "detection_date", "high", "price"]].copy()
        winners_gain["actual_high_pct"] = (
            (winners_gain["high"].astype(float) / winners_gain["price"].astype(float) - 1) * 100
        ).clip(lower=0)

        # Only keep the gain cols to avoid duplicate columns after merge
        winners_gain = winners_gain[["symbol", "detection_date", "actual_high_pct"]]

        sym_col = next((c for c in ["symbol", "ticker"] if c in t1_df.columns), None)
        if sym_col is None:
            logger.warning("  No symbol column in T-1 data — skipping gain enrichment")
            return t1_df

        before = len(t1_df)
        t1_df = t1_df.merge(
            winners_gain,
            left_on=[sym_col, "detection_date"],
            right_on=["symbol", "detection_date"],
            how="left",
            suffixes=("", "_gain"),
        )

        # Clean up duplicate symbol column if merge added one
        if "symbol_gain" in t1_df.columns:
            t1_df = t1_df.drop(columns=["symbol_gain"])

        n_with_gain = t1_df["actual_high_pct"].notna().sum()
        logger.info(
            f"  ✓ Enriched {n_with_gain}/{before} T-1 rows with intraday peak gain data"
        )

        if n_with_gain < 30:
            logger.warning(
                f"  Only {n_with_gain} winner rows have gain data. "
                "Gain regressor needs ≥30 to train. "
                "Accumulate more T-1 data by running the detection pipeline."
            )
    except Exception as e:
        logger.warning(f"  Could not fetch gain data for T-1: {e} — gain regressor may be skipped")

    return t1_df


def train_gain_regressor(
    X_scaled_full: pd.DataFrame,
    y_full: pd.Series,
    w_full: pd.Series,
    combined_df: pd.DataFrame,
    feature_names: list[str],
):
    """
    FIX 4+5: Train gain regressor on SCALED features aligned by index.

    Args:
        X_scaled_full:  Complete scaled feature matrix (same index as combined_df).
                        Using the SCALED version ensures regressor and classifier
                        operate on the same feature space at prediction time.
        y_full:         Labels for all rows (aligned with X_scaled_full).
        w_full:         Sample weights (aligned with X_scaled_full).
        combined_df:    Full combined DataFrame with actual_high_pct column.
        feature_names:  List of feature names (columns of X_scaled_full).
    """
    from xgboost import XGBRegressor

    gain_col = None
    for candidate in ("actual_high_pct", "actual_gain_pct", "change_pct"):
        if candidate in combined_df.columns:
            gain_col = candidate
            break

    if gain_col is None:
        logger.warning("No gain column found — skipping gain regressor training.")
        return None

    # FIX 4: Use combined_df.index to align X_scaled_full correctly.
    # X_scaled_full shares the same index as combined_df after prepare_features + build_scaler.
    winner_mask = (
        (combined_df["label"] == 1) &
        pd.to_numeric(combined_df[gain_col], errors="coerce").notna()
    )

    n_winners = int(winner_mask.sum())

    if n_winners < 30:
        logger.warning(
            f"Only {n_winners} winner rows with gain data — "
            "need ≥30 to train gain regressor. Skipping."
        )
        return None

    logger.info(f"\n── Training gain regressor on {n_winners} winner rows (target: {gain_col}) ──")

    # FIX 4+5: Align X_scaled_full to combined_df's index before masking.
    # Both should already share the same integer RangeIndex after reset_index.
    # If indices differ, align explicitly.
    if not X_scaled_full.index.equals(combined_df.index):
        logger.warning(
            "X_scaled_full and combined_df indices differ — reindexing X_scaled_full. "
            "This shouldn't happen; check that prepare_features + build_scaler preserve index."
        )
        X_scaled_full = X_scaled_full.reindex(combined_df.index)

    X_reg  = X_scaled_full.loc[winner_mask]   # FIX 5: already scaled, no raw data
    y_reg  = pd.to_numeric(combined_df.loc[winner_mask, gain_col], errors="coerce")
    w_reg  = w_full.loc[winner_mask] if hasattr(w_full, "loc") else w_full[winner_mask]

    # FIX 5: Scaled features may still have NaN (from the scaler's NaN-preservation).
    # Fill with 0 (scaled mean) since scaler already centered around 0.
    X_reg_filled = X_reg.fillna(0.0)

    from sklearn.model_selection import train_test_split
    if len(X_reg_filled) >= 10:
        X_tr, X_va, y_tr, y_va, w_tr, _ = train_test_split(
            X_reg_filled, y_reg, w_reg,
            test_size=0.2, random_state=42,
        )
    else:
        X_tr, X_va, y_tr, y_va, w_tr = (
            X_reg_filled, X_reg_filled, y_reg, y_reg, w_reg
        )

    logger.info(f"  Gain regressor train: {len(X_tr)} rows, val: {len(X_va)} rows")
    logger.info(f"  Gain range: {y_reg.min():.1f}% – {y_reg.max():.1f}%  mean={y_reg.mean():.1f}%")

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
        sample_weight=w_tr.values if hasattr(w_tr, "values") else w_tr,
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

    client = get_supabase_client()

    base_df = load_base_training_data(client)
    t1_df   = load_t1_data(client)

    # FIX 3: Enrich T-1 rows with gain data BEFORE combine_datasets.
    # This uses detection_date (correct), not event_date (wrong).
    if not t1_df.empty:
        t1_df = enrich_t1_data_with_gain(t1_df, client)

    combined_df = combine_datasets(base_df, t1_df)

    combined_df = apply_intraday_high_labels(combined_df, threshold=INTRADAY_WIN_THRESHOLD)

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
        mistake_df_len = len(mistake_df) if not mistake_df.empty else 0
    else:
        logger.warning("ml_mistake_learner not available — skipping mistake-learning step.")
        mistake_df_len = 0

    X, y, w = prepare_features(combined_df)
    feature_names = list(X.columns)

    logger.info("Fitting scaler...")
    scaler, X_scaled = build_scaler(X)

    # FIX 4+5: Reset index on X_scaled and combined_df together so they stay aligned.
    # prepare_features + build_scaler should preserve index, but reset to be safe.
    combined_df = combined_df.reset_index(drop=True)
    X_scaled    = X_scaled.reset_index(drop=True)
    y           = y.reset_index(drop=True)
    w           = w.reset_index(drop=True)

    X_train, X_val, y_train, y_val, w_train, w_val = train_val_split(
        X_scaled, y, w, combined_df
    )

    model = train_model(X_train, y_train, w_train, X_val, y_val)

    fi_df = compute_feature_importance(model, feature_names)

    logger.info("\n" + "=" * 60)
    logger.info("GAIN REGRESSOR TRAINING")
    logger.info("=" * 60)
    # FIX 4+5: Pass X_scaled_full (complete, scaled) so regressor uses same scale as classifier.
    gain_regressor = train_gain_regressor(
        X_scaled_full=X_scaled,
        y_full=y,
        w_full=w,
        combined_df=combined_df,
        feature_names=feature_names,
    )

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

    val_proba_series = pd.Series(val_proba)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = pd.cut(val_proba_series, bins=bins).value_counts().sort_index()
    logger.info("Val set probability distribution:")
    for bucket, count in dist.items():
        logger.info(f"  {str(bucket):<20} {count:>4}")

    gap_count = int(((val_proba_series > 0.15) & (val_proba_series < 0.85)).sum())
    if gap_count < 5:
        logger.warning(
            f"  ⚠️  BIMODAL COLLAPSE detected: only {gap_count} predictions in 0.15–0.85 range."
        )
    else:
        logger.info(f"  ✅ {gap_count} predictions in mid-range (0.15–0.85) — distribution looks healthy")

    training_stats = {
        "n_total_samples":         len(combined_df),
        "n_base_samples":          len(base_df),
        "n_t1_samples":            len(t1_df) if not t1_df.empty else 0,
        "n_mistake_samples":       mistake_df_len,
        "n_positive":              int((y == 1).sum()),
        "n_negative":              int((y == 0).sum()),
        "positive_rate":           float((y == 1).mean()),
        "val_auc_roc":             float(auc),
        "base_sample_weight":      BASE_CSV_WEIGHT,
        "t1_sample_weight":        T1_WEIGHT,
        "intraday_win_threshold":  INTRADAY_WIN_THRESHOLD,
        "spw_min":                 SPW_MIN,
        "spw_max":                 SPW_MAX,
        "equal_weight_applied": (
            len(t1_df) >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT
            if not t1_df.empty else False
        ),
        "gain_regressor_trained":  gain_regressor is not None,
        "split_method":            "time_based",
    }

    save_outputs(model, scaler, fi_df, feature_names, training_stats, gain_regressor)

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
    logger.info(f"  SPW cap             : [{SPW_MIN}, {SPW_MAX}]")
    logger.info(f"  Split method        : time-based (most recent {15}% as val)")
    logger.info(f"  Gain regressor      : {'✓ trained' if gain_regressor else '— skipped (need ≥30 winner rows with gain data)'}")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--use-all-timepoints", action="store_true", default=True)
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.exit(main())
