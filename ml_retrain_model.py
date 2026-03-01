#!/usr/bin/env python3
"""
ml_retrain_model.py  —  Weekly Full Retrain From Scratch

GAIN REGRESSOR FIXES (this version):

BUG 1 — Wrong feature matrix passed to regressor:
  Previously train_gain_regressor() rebuilt X_reg from combined_df columns,
  matching feature_names by name. But combined_df columns are raw/unprefixed
  from the base CSV (e.g. "RSI_14"), while feature_names are prefixed
  (e.g. "t3_RSI_14"). So `if col in combined_df.columns` NEVER matched,
  X_reg was entirely NaN, and the regressor learned nothing — it just
  predicted the training mean for every stock.
  FIX: Pass X_train (already built, scaled, correct features) directly
  to the regressor instead of rebuilding from combined_df.

BUG 2 — Gain target missing for most winners:
  actual_high_pct was fetched from daily_winners and merged using
  left_on=[symbol_col, date_col]. Base CSV rows have event_date; T-1 rows
  have detection_date. The merge right_on was always 'detection_date',
  so base CSV winner rows (which use event_date) got NaN for actual_high_pct.
  winner_mask then excluded them, leaving potentially <30 samples total
  (only T-1 winner rows), hitting the "need ≥30" guard and skipping training.
  FIX: Try merging on both date columns. Also fall back to change_pct from
  the base CSV itself when actual_high_pct is unavailable.

BUG 3 — Regressor silently nullified at prediction time:
  ExplosionPredictor checked self._regressor_n_features != classifier_n
  and set self.regressor = None if they differed. Since the regressor was
  trained on the same feature_names as the classifier, counts always matched —
  but the regressor was useless (BUG 1). Now that the regressor is trained
  correctly on X_train, counts will still match and the check still passes.

BUG 4 — predict_with_targets used scaled X for regressor:
  The regressor was trained on X_train (scaler-transformed). At prediction
  time predict_with_targets also passes X_scaled. This is consistent and
  correct — no change needed there. Documented here for clarity.
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

INTRADAY_WIN_THRESHOLD = 15.0

SPW_MIN = 0.5
SPW_MAX = 3.0

XGBOOST_PARAMS = {
    "n_estimators":       300,
    "max_depth":          5,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   10,
    "gamma":              1.0,
    "reg_alpha":          0.5,
    "reg_lambda":         2.0,
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
        logger.error(f"Table '{TABLE_BASE}' is empty!")
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
    """Load accumulated T-1 winner and non-winner samples."""
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
                    logger.warning(f"  {table}: dropping {len(dupes)} duplicate column(s)")
                    df = df.loc[:, ~df.columns.duplicated(keep="first")]
            else:
                logger.warning(f"  {table}: column map unavailable")

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
    """Re-label rows where actual_high_pct exceeds threshold as winners."""
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
        logger.info(f"Intraday-high relabelling: {after - before} rows upgraded to label=1")
    return combined_df


def combine_datasets(base_df: pd.DataFrame, t1_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate base + T-1 data with deduplication."""
    if t1_df.empty:
        logger.info("Combining: base data only (no T-1 data yet)")
        return base_df.copy()

    t1_count = len(t1_df)
    if t1_count >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT:
        logger.info(f"T-1 data ({t1_count} rows) >= threshold. Using equal sample weights.")
        base_df = base_df.copy()
        base_df["sample_weight"] = 1.0
    else:
        logger.info(f"T-1 data ({t1_count} rows) < threshold. Applying differential weights.")

    combined = pd.concat([t1_df, base_df], ignore_index=True, sort=False)

    sym_col = next((c for c in ["symbol", "ticker"] if c in combined.columns), None)
    date_cols_present = [c for c in ["detection_date", "event_date"] if c in combined.columns]

    if sym_col and len(date_cols_present) == 1:
        date_col = date_cols_present[0]
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=[sym_col, date_col], keep="first")
        n_dropped = before_dedup - len(combined)
        if n_dropped > 0:
            logger.info(f"Deduplication: removed {n_dropped} duplicate rows")
    elif sym_col and len(date_cols_present) == 2:
        logger.info("Skipping deduplication: base CSV (event_date) and T-1 (detection_date) use different date columns")

    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())
    logger.info(
        f"Combined dataset: {len(combined)} rows, "
        f"pos={n_pos}, neg={n_neg}, "
        f"pos_rate={n_pos/len(combined)*100:.1f}%"
    )

    if n_neg == 0:
        logger.error("CRITICAL: No negative samples found.")
        sys.exit(1)

    return combined


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract feature matrix X, labels y, and sample weights w."""
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
    logger.info(f"Overall NaN rate: {nan_pct:.1f}%")

    return X, y, w


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def build_scaler(X: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """Fit scaler and return scaled X with NaN filled by column means.

    NaN values (e.g. missing T-1 features for base-CSV rows) are filled with
    column means BEFORE scaling, producing 0.0 after StandardScaler transform.
    We do NOT restore NaN after scaling — this is intentional.

    At prediction time, _scale_features() applies the same mean-fill before
    scaling. This ensures the model sees exactly the same representation for
    'missing T-1 data' during training and prediction (both get 0.0 = scaled mean).

    If NaN were restored after scaling, XGBoost would receive NaN at prediction
    time but 0.0 at training time — different split paths -> equal probabilities
    for all stocks lacking T-1 data (the bimodal collapse bug).
    """
    scaler    = StandardScaler()
    col_means = X.mean()
    X_filled  = X.fillna(col_means)
    scaler.fit(X_filled)

    X_scaled_vals = scaler.transform(X_filled)
    X_scaled      = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)

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
        clamped_spw = max(SPW_MIN, min(SPW_MAX, raw_spw))
        params["scale_pos_weight"] = round(clamped_spw, 3)
        logger.info(f"  scale_pos_weight: {clamped_spw:.3f} (neg={n_neg}/pos={n_pos})")

    early_stopping = params.pop("early_stopping_rounds", 30)

    model = XGBClassifier(**params, early_stopping_rounds=early_stopping)

    logger.info("Training XGBoost classifier from scratch...")
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
        logger.warning(f"  ⚠️  Val logloss={model.best_score:.4f} is suspiciously low.")

    return model


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    df_with_dates: pd.DataFrame,
    val_fraction: float = 0.15,
) -> tuple:
    """Time-based train/val split to prevent data leakage."""
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
    else:
        date_col = None

    if date_col is None:
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
        logger.warning(f"  ⚠️  Only {val_pos} positive examples in validation set.")

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
# Gain regressor — FIXED VERSION
# ---------------------------------------------------------------------------

def train_gain_regressor(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    combined_df: pd.DataFrame,
    feature_names: list[str],
) -> "Optional[object]":
    """
    Train a regression model to predict actual % gain for winning stocks.

    FIX BUG 1: Previously rebuilt X_reg from combined_df using feature_names
    as column lookups. But combined_df columns are raw/unprefixed (e.g. "RSI_14")
    while feature_names are prefixed (e.g. "t3_RSI_14"). The lookup always
    failed → X_reg was entirely NaN → regressor learned only the mean.

    FIX: Use X_train_scaled directly (already built with correct prefixed names
    and scaler-transformed). The regressor trains on the same feature space as
    the classifier, which is what we want for consistent prediction at inference.

    FIX BUG 2: actual_high_pct merge only populated T-1 winner rows because
    the merge used right_on='detection_date' but base CSV rows have event_date.
    FIX: Try multiple gain column sources in priority order:
      1. actual_high_pct (intraday peak — best measure of opportunity)
      2. actual_gain_pct (close-to-close)
      3. change_pct (present in base CSV directly)
    This ensures base CSV winners contribute training samples too.
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    # --- Find best available gain target ---
    # Priority: actual_high_pct > actual_gain_pct > change_pct
    # We look in combined_df (which may have these from the merge or base CSV)
    gain_col = None
    for candidate in ("actual_high_pct", "actual_gain_pct", "change_pct"):
        if candidate in combined_df.columns:
            n_valid = pd.to_numeric(combined_df[candidate], errors="coerce").notna().sum()
            if n_valid > 0:
                gain_col = candidate
                logger.info(f"Gain regressor target: '{gain_col}' ({n_valid} non-null values)")
                break

    if gain_col is None:
        logger.warning("No gain column found in combined_df — skipping gain regressor training.")
        return None

    # --- Build gain target aligned to X_train_scaled index ---
    # X_train_scaled has a subset of combined_df's index (train split rows only)
    # We need the gain values for those same rows, for winner rows only
    gain_series = pd.to_numeric(combined_df[gain_col], errors="coerce")
    label_series = combined_df["label"].astype(int)

    # Align to training set indices
    train_idx = X_train_scaled.index
    gain_train = gain_series.reindex(train_idx)
    label_train = label_series.reindex(train_idx)
    w_train_aligned = w_train.reindex(train_idx)

    # Winner mask: label=1 AND gain is known
    winner_mask = (label_train == 1) & gain_train.notna()
    n_winners = int(winner_mask.sum())

    logger.info(f"Gain regressor: {n_winners} winner rows with '{gain_col}' in training set")

    if n_winners < 30:
        logger.warning(
            f"Only {n_winners} winner rows with gain data (need ≥30). "
            f"Skipping gain regressor. Accumulate more daily winner data to enable this."
        )
        return None

    # --- Extract winner subset ---
    X_winners = X_train_scaled[winner_mask]
    y_winners = gain_train[winner_mask]
    w_winners = w_train_aligned[winner_mask]

    # Fill any remaining NaN in X with column means (XGBRegressor handles NaN
    # but filling gives cleaner training signal)
    col_means = X_winners.mean()
    X_winners_filled = X_winners.fillna(col_means)

    # --- Train/val split for regressor ---
    if len(X_winners) >= 20:
        X_tr, X_va, y_tr, y_va, w_tr, _ = train_test_split(
            X_winners_filled, y_winners, w_winners,
            test_size=0.2, random_state=42,
        )
    else:
        X_tr = X_va = X_winners_filled
        y_tr = y_va = y_winners
        w_tr = w_winners

    logger.info(f"  Regressor train: {len(X_tr)} samples, val: {len(X_va)} samples")
    logger.info(f"  Target range: {y_winners.min():.1f}% – {y_winners.max():.1f}%  "
                f"mean={y_winners.mean():.1f}%  median={y_winners.median():.1f}%")

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
    mae = mean_absolute_error(y_va, val_pred)
    r2  = r2_score(y_va, val_pred) if len(y_va) > 1 else float("nan")
    logger.info(f"  Gain regressor val — MAE: {mae:.2f}%  R²: {r2:.3f}")
    logger.info(f"  Predicted range: {val_pred.min():.1f}% – {val_pred.max():.1f}%")

    if mae > 50:
        logger.warning(
            f"  ⚠️  MAE={mae:.1f}% is very high. The regressor may not be useful yet. "
            "This is expected early on — it improves as more winner data accumulates."
        )

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
        logger.info("Gain regressor not trained this run")

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
    # FIX BUG 2: Try merging on BOTH detection_date and event_date so base CSV
    # winner rows also get actual_high_pct populated (not just T-1 rows)
    logger.info("Fetching intraday peak gain data from daily_winners...")
    try:
        winners_response = fetch_table_paginated(client, "daily_winners")
        if not winners_response.empty:
            required = {"symbol", "detection_date", "high", "price"}
            if required.issubset(winners_response.columns):
                winners_gain = winners_response[["symbol", "detection_date", "high", "price"]].copy()
                winners_gain["actual_high_pct"] = (
                    (winners_gain["high"] / winners_gain["price"] - 1) * 100
                ).clip(lower=0)

                sym_col = next(
                    (c for c in ["symbol", "ticker"] if c in combined_df.columns), None
                )

                n_merged_total = 0

                # Try merging on detection_date (T-1 rows)
                if sym_col and "detection_date" in combined_df.columns:
                    merged = combined_df.merge(
                        winners_gain[["symbol", "detection_date", "actual_high_pct"]],
                        left_on=[sym_col, "detection_date"],
                        right_on=["symbol", "detection_date"],
                        how="left",
                        suffixes=("", "_gain"),
                    )
                    # Copy over actual_high_pct where it was found
                    if "actual_high_pct_gain" in merged.columns:
                        new_vals = merged["actual_high_pct_gain"].notna()
                        combined_df.loc[new_vals[new_vals].index, "actual_high_pct"] = (
                            merged.loc[new_vals[new_vals].index, "actual_high_pct_gain"].values
                        )
                        n_merged_total += new_vals.sum()
                    elif "actual_high_pct" in merged.columns:
                        combined_df["actual_high_pct"] = merged["actual_high_pct"].values
                        n_merged_total += merged["actual_high_pct"].notna().sum()

                # Try merging on event_date (base CSV rows)
                if sym_col and "event_date" in combined_df.columns:
                    # Only update rows that still don't have actual_high_pct
                    missing_mask = combined_df.get("actual_high_pct", pd.Series(dtype=float)).isna()
                    if missing_mask.any():
                        merged_event = combined_df[missing_mask].merge(
                            winners_gain[["symbol", "detection_date", "actual_high_pct"]].rename(
                                columns={"detection_date": "event_date"}
                            ),
                            left_on=[sym_col, "event_date"],
                            right_on=["symbol", "event_date"],
                            how="left",
                            suffixes=("", "_gain"),
                        )
                        if "actual_high_pct_gain" in merged_event.columns:
                            new_vals_ev = merged_event["actual_high_pct_gain"].notna()
                            combined_df.loc[
                                merged_event[new_vals_ev].index, "actual_high_pct"
                            ] = merged_event.loc[new_vals_ev, "actual_high_pct_gain"].values
                            n_merged_total += new_vals_ev.sum()

                logger.info(f"Enriched {n_merged_total} rows with intraday peak gain data")
            else:
                logger.warning(f"daily_winners missing required columns: {required - set(winners_response.columns)}")
    except Exception as e:
        logger.warning(f"Could not fetch gain data: {e} — gain regressor will use change_pct fallback")

    # ── FIX 4: Relabel rows with strong intraday moves as winners ─────────────
    combined_df = apply_intraday_high_labels(combined_df, threshold=INTRADAY_WIN_THRESHOLD)

    # ── Load mistake samples ──────────────────────────────────────────────────
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
            logger.info(f"Dataset after adding mistakes: {len(combined_df)} rows")
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

    # ── Time-based train/val split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val, w_train, w_val = train_val_split(
        X_scaled, y, w, combined_df
    )

    # ── Train classifier ───────────────────────────────────────────────────────
    model = train_model(X_train, y_train, w_train, X_val, y_val)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df = compute_feature_importance(model, feature_names)

    # ── Train gain regressor (FIXED) ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("GAIN REGRESSOR TRAINING")
    logger.info("=" * 60)
    gain_regressor = train_gain_regressor(
        X_train,        # FIX: pass already-built scaled feature matrix
        y_train,
        w_train,
        combined_df,    # still needed for gain target column lookup
        feature_names,
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
        logger.warning("Validation AUC-ROC: nan")

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
        logger.warning(f"  ⚠️  BIMODAL COLLAPSE: only {gap_count} predictions in 0.15–0.85 range.")
    else:
        logger.info(f"  ✅ {gap_count} predictions in mid-range — distribution looks healthy")

    # ── Training stats ────────────────────────────────────────────────────────
    n_mistakes = len(mistake_df) if not mistake_df.empty else 0
    n_with_gain = (
        pd.to_numeric(combined_df.get("actual_high_pct", pd.Series(dtype=float)), errors="coerce").notna().sum()
        + pd.to_numeric(combined_df.get("change_pct", pd.Series(dtype=float)), errors="coerce").notna().sum()
    )
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
        "gain_regressor_target":   "actual_high_pct or change_pct",
        "split_method":            "time_based",
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(model, scaler, fi_df, feature_names, training_stats, gain_regressor)

    # ── Verify gain regressor round-trips correctly ───────────────────────────
    if gain_regressor is not None:
        try:
            loaded = joblib.load(GAIN_REGRESSOR_PATH)
            test_input = pd.DataFrame(
                np.zeros((1, len(feature_names))), columns=feature_names
            )
            test_pred = loaded.predict(test_input)
            logger.info(
                f"  ✅ Gain regressor verified loadable — "
                f"test prediction: {test_pred[0]:.2f}%"
            )
            logger.info(
                f"     ExplosionPredictor should load from: "
                f"{GAIN_REGRESSOR_PATH.resolve()}"
            )
        except Exception as e:
            logger.error(f"  ❌ Gain regressor verification FAILED after save: {e}")

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
    logger.info(f"  Gain regressor      : {'✓ trained' if gain_regressor else '— skipped (need ≥30 winners with gain data)'}")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
