#!/usr/bin/env python3
"""
ml_retrain_model.py  —  Weekly Full Retrain From Scratch

FIXES IN THIS VERSION (2026-03-02):

FIX 1 — scale_pos_weight capped too low (was [0.5, 3.0]):
  The previous cap of 3.0 was designed for datasets with ~75% negative samples,
  not the actual class balance in this system. Daily winners represent roughly
  0.5–1% of all US stocks on any given day — a natural imbalance ratio of
  100:1 to 200:1. When we also include non-winner samples from the training set
  (which are intentionally sampled at a modest ratio), the training imbalance
  is typically 5:1 to 20:1 depending on accumulation so far.

  A cap of 3.0 means the model was being trained as if positives were only 3×
  underrepresented — it learned to predict AVOID for anything borderline because
  the cost of a false positive was 3× the cost of a false negative, when it
  should be 5–20×. This makes the model extremely conservative.

  New cap: [1.0, 30.0]. The lower bound of 1.0 prevents the rare case where
  we have more winners than non-winners in a small accumulated training set.
  The upper bound of 30.0 is a safety rail against degenerate datasets.

FIX 2 — Regularisation slightly relaxed for t3_-only prediction:
  The previous strong regularisation (min_child_weight=10, gamma=1.0) was
  appropriate for a dataset where T-1 features carry most of the signal.
  When the model must rely primarily on t3_/t5_/t10_ features (which are
  less discriminative), too-strong regularisation causes all predictions to
  collapse toward the base rate. Relaxed to min_child_weight=5, gamma=0.3.

All other fixes from the previous version are preserved:
  - Time-based train/val split (no data leakage)
  - Intraday-high label support (INTRADAY_WIN_THRESHOLD)
  - Duplicate-date deduplication
  - Mistake learner integration
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

# FIX 1: Raise upper cap from 3.0 → 30.0 to reflect real class imbalance.
# Daily winners are ~0.5-1% of market → natural ratio is 100-200:1.
# With accumulated non-winner samples this is typically 5-20:1 in training data.
# A cap of 3.0 was causing the model to treat AVOID as nearly free → all AVOID.
SPW_MIN = 1.0   # was 0.5
SPW_MAX = 30.0  # was 3.0

XGBOOST_PARAMS = {
    "n_estimators":       300,
    "max_depth":          5,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    # FIX 2: Relaxed regularisation so t3_-only features can still discriminate.
    # Too-strong regularisation with weak features → all predictions = base rate.
    "min_child_weight":   5,       # was 10
    "gamma":              0.3,     # was 1.0
    "reg_alpha":          0.3,     # was 0.5
    "reg_lambda":         1.5,     # was 2.0
    "scale_pos_weight":   1,       # overridden at train time (clamped to SPW_MIN/MAX)
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

    if n_pos > 0 and (n_neg / n_pos) < 0.2:
        logger.warning(
            f"Class imbalance WARNING: {n_pos} positives vs {n_neg} negatives "
            f"(ratio {n_neg/n_pos:.2f}). scale_pos_weight will compensate, "
            "but consider accumulating more non-winner data."
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
        if abs(raw_spw - clamped_spw) > 0.01:
            logger.info(
                f"  scale_pos_weight: raw={raw_spw:.3f} → clamped to {clamped_spw:.3f} "
                f"(limits: [{SPW_MIN}, {SPW_MAX}])  ← FIX: cap raised from 3.0 to 30.0"
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
# Gain regressor
# ---------------------------------------------------------------------------

def train_gain_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    combined_df: pd.DataFrame,
    feature_names: list[str],
):
    from xgboost import XGBRegressor

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

    base_df     = load_base_training_data(client)
    t1_df       = load_t1_data(client)
    combined_df = combine_datasets(base_df, t1_df)

    # Enrich with intraday peak gain
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
    else:
        logger.warning("ml_mistake_learner not available — skipping mistake-learning step.")
        mistake_df = pd.DataFrame()

    X, y, w = prepare_features(combined_df)
    feature_names = list(X.columns)

    logger.info("Fitting scaler...")
    scaler, X_scaled = build_scaler(X)

    X_train, X_val, y_train, y_val, w_train, w_val = train_val_split(
        X_scaled, y, w, combined_df
    )

    model = train_model(X_train, y_train, w_train, X_val, y_val)

    fi_df = compute_feature_importance(model, feature_names)

    logger.info("\n" + "=" * 60)
    logger.info("GAIN REGRESSOR TRAINING")
    logger.info("=" * 60)
    gain_regressor = train_gain_regressor(
        X_train, y_train, w_train, combined_df, feature_names
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
    logger.info(f"  SPW cap             : [{SPW_MIN}, {SPW_MAX}]  ← was [0.5, 3.0]")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--use-all-timepoints", action="store_true", default=True)
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.exit(main())
