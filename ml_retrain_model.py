#!/usr/bin/env python3
"""
ml_retrain_model.py  —  Weekly Full Retrain From Scratch

Replaces the previous fine-tuning approach with a complete retrain every week.

DATA SOURCES (combined into one training dataset):
  1. ml_training_base    — original CSV data pivoted to wide format, both classes
                           Feature prefixes: t3_, t5_, t10_ only
                           (same_day excluded = leakage; day_before excluded = t1
                            comes from Supabase daily tables instead)
                           Uploaded once via upload_base_training_data.py
  2. winners_day_prior_close / winners_day_prior_open
                         — accumulating T-1 winner samples from daily runs (label=1)
  3. non_winners_day_prior_close / non_winners_day_prior_open
                         — accumulating T-1 non-winner samples from daily runs (label=0)

NOTE ON CLASS BALANCE:
  ml_training_base contains ONLY winners (label=1). Non-winners (label=0) come
  entirely from the accumulated Supabase T-1 tables. Until enough T-1 data
  accumulates, the model is trained on imbalanced data. scale_pos_weight in
  XGBoost params compensates for this automatically via sample_weight.

WHY FULL RETRAIN (not fine-tuning):
  - Only ~3,600 base rows — trivially fast to retrain (seconds, not minutes)
  - Fine-tuning with dummy-default T-3/T-7/T-14 values was corrupting new trees
  - NaN for genuinely missing columns is correct; XGBoost handles it natively
  - feature_importance.csv is regenerated each run — always accurate and current
  - Sample weights: base CSV rows weighted 1.5x initially (tapers automatically
    once T-1 data reaches >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT)

OUTPUTS (same paths as before, drop-in compatible with ml_weekly_retrain.yml):
  ml_models/best_model.pkl
  ml_models/scaler.pkl
  ml_models/model_metadata.json
  ml_models/feature_importance.csv
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Supabase table names
TABLE_BASE = "ml_training_base"
TABLE_WINNERS_CLOSE = "winners_day_prior_close"
TABLE_WINNERS_OPEN = "winners_day_prior_open"
TABLE_NON_WINNERS_CLOSE = "non_winners_day_prior_close"
TABLE_NON_WINNERS_OPEN = "non_winners_day_prior_open"

# Output paths (relative to repo root)
MODEL_DIR = Path("ml_models")
MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

# Sample weighting
BASE_CSV_WEIGHT = 1.5       # Weight for original CSV rows
T1_WEIGHT = 1.0             # Weight for accumulated T-1 rows
# Once T-1 data reaches this many rows, switch to equal weighting (1.0 / 1.0)
MIN_T1_ROWS_FOR_EQUAL_WEIGHT = 1800

# XGBoost hyperparameters (keep same as original training unless you tune)
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,   # Will be overridden by sample_weight
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}

# Columns to exclude from feature matrix (metadata / label columns)
NON_FEATURE_COLS = {
    "id", "created_at", "updated_at", "date", "symbol", "ticker",
    "label", "source", "sample_weight", "detection_date", "explosion_date",
    "change_pct", "rank", "notes",
}

# Columns that are T-1 specific (not present in base CSV)
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
    rows = []
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

    # Ensure label column exists
    if "label" not in df.columns:
        logger.error(f"'{TABLE_BASE}' has no 'label' column.")
        sys.exit(1)

    # Preserve sample_weight if present, otherwise assign default
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

    frames = []

    # Winners (label = 1)
    for table in [TABLE_WINNERS_CLOSE, TABLE_WINNERS_OPEN]:
        try:
            df = fetch_table_paginated(client, table)
            if not df.empty:
                df["label"] = 1
                df["source"] = table
                frames.append(df)
        except Exception as e:
            logger.warning(f"Could not load '{table}': {e}")

    # Non-winners (label = 0)
    for table in [TABLE_NON_WINNERS_CLOSE, TABLE_NON_WINNERS_OPEN]:
        try:
            df = fetch_table_paginated(client, table)
            if not df.empty:
                df["label"] = 0
                df["source"] = table
                frames.append(df)
        except Exception as e:
            logger.warning(f"Could not load '{table}': {e}")

    if not frames:
        logger.warning("No T-1 data found. Training on base data only.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["sample_weight"] = T1_WEIGHT

    logger.info(f"T-1 data: {len(combined)} rows, "
                f"pos={int((combined['label']==1).sum())}, "
                f"neg={int((combined['label']==0).sum())}")
    return combined


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def combine_datasets(base_df: pd.DataFrame, t1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate base + T-1 data.

    Columns present only in base → NaN in T-1 rows (XGBoost handles natively)
    Columns present only in T-1  → NaN in base rows (XGBoost handles natively)

    This is intentional and correct. NaN tells the model "this feature was
    genuinely not observed for this sample", which is different from a dummy
    value of 0.0 or 50.0 that implies a specific measurement.
    """
    if t1_df.empty:
        logger.info("Combining: base data only (no T-1 data yet)")
        return base_df.copy()

    # Adjust sample weights based on T-1 data volume
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

    combined = pd.concat([base_df, t1_df], ignore_index=True, sort=False)

    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())
    logger.info(
        f"Combined dataset: {len(combined)} rows, "
        f"{len(combined.columns)} columns, "
        f"pos={n_pos}, neg={n_neg}"
    )

    if n_neg == 0:
        logger.error(
            "CRITICAL: No negative (non-winner) samples found. "
            "The model cannot train without both classes. "
            "Ensure non_winners_day_prior_close/open tables have data in Supabase."
        )
        sys.exit(1)

    if n_pos > 0 and (n_neg / n_pos) < 0.2:
        logger.warning(
            f"Class imbalance WARNING: {n_pos} positives vs {n_neg} negatives "
            f"(ratio {n_neg/n_pos:.2f}). scale_pos_weight will compensate, "
            "but consider accumulating more non-winner data before relying on this model."
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
    # Extract labels and weights before dropping non-feature cols
    y = df["label"].astype(int)
    w = df["sample_weight"].astype(float) if "sample_weight" in df.columns else pd.Series(1.0, index=df.index)

    # Build feature columns: exclude all metadata/label/source columns
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not c.startswith("Unnamed")
    ]

    X = df[feature_cols].copy()

    # Convert all feature columns to numeric; non-numeric → NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Replace inf values with NaN (XGBoost handles NaN, not inf)
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

    Important: We fill NaN with column mean ONLY for scaling purposes.
    The scaled DataFrame retains NaN so XGBoost can use its missing-value
    routing logic correctly.
    """
    scaler = StandardScaler()

    # Fit on column means (ignoring NaN) — this gives a meaningful scale
    col_means = X.mean()
    X_filled_for_fit = X.fillna(col_means)
    scaler.fit(X_filled_for_fit)

    # Apply scaling but PRESERVE NaN positions
    nan_mask = X.isna()
    X_scaled_values = scaler.transform(X_filled_for_fit)
    X_scaled = pd.DataFrame(X_scaled_values, columns=X.columns, index=X.index)
    X_scaled[nan_mask] = np.nan  # Restore NaN positions

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

    # Auto-compute scale_pos_weight from training class balance
    # This compensates for the fact that base CSV is winners-only
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0 and n_neg > 0:
        params["scale_pos_weight"] = round(n_neg / n_pos, 3)
        logger.info(f"  scale_pos_weight set to {params['scale_pos_weight']:.3f} "
                    f"(neg={n_neg} / pos={n_pos})")

    model = XGBClassifier(**{k: v for k, v in params.items() if k != "early_stopping_rounds"})

    logger.info("Training XGBoost model from scratch...")
    logger.info(f"  Train: {len(X_train)} rows")
    logger.info(f"  Val:   {len(X_val)} rows")

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train.values,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=XGBOOST_PARAMS.get("early_stopping_rounds", 30),
        verbose=False,
    )

    best_iteration = model.best_iteration
    val_score = model.best_score
    logger.info(f"  Best iteration: {best_iteration}")
    logger.info(f"  Best val logloss: {val_score:.4f}")

    return model


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    val_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Stratified split preserving class balance."""
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w,
        test_size=val_fraction,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        f"Train/val split: {len(X_train)} train "
        f"(pos={int((y_train==1).sum())}, neg={int((y_train==0).sum())}), "
        f"{len(X_val)} val "
        f"(pos={int((y_val==1).sum())}, neg={int((y_val==0).sum())})"
    )
    return X_train, X_val, y_train, y_val, w_train, w_val


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Generate feature_importance.csv from freshly trained model.
    Uses 'gain' importance type — most informative for XGBoost.
    """
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")

    # Map internal fN names back to actual feature names
    # XGBoost uses f0, f1, ... internally when feature names aren't set
    # Since we pass a DataFrame, names should be preserved — but handle both cases
    importance_list = []
    for feat, score in scores.items():
        if feat.startswith("f") and feat[1:].isdigit():
            idx = int(feat[1:])
            name = feature_names[idx] if idx < len(feature_names) else feat
        else:
            name = feat
        importance_list.append({"feature": name, "importance": round(score, 6)})

    fi_df = pd.DataFrame(importance_list)
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Normalize to sum to 1.0
    total = fi_df["importance"].sum()
    if total > 0:
        fi_df["importance"] = (fi_df["importance"] / total).round(6)

    logger.info(f"Feature importance computed: {len(fi_df)} features")
    logger.info("Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return fi_df


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    model: XGBClassifier,
    scaler: StandardScaler,
    fi_df: pd.DataFrame,
    feature_names: list[str],
    training_stats: dict,
) -> None:
    """Save model, scaler, feature importance, and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model → {MODEL_PATH}")

    # Scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler → {SCALER_PATH}")

    # Feature importance
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Saved feature importance → {FEATURE_IMPORTANCE_PATH}")

    # Metadata
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": "ml_retrain_model.py",
        "training_approach": "full_retrain_from_scratch",
        "n_features": len(feature_names),
        "feature_names_sample": feature_names[:20],
        "best_iteration": int(model.best_iteration),
        "best_val_logloss": float(model.best_score),
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

    # ── Load data ─────────────────────────────────────────────────────────────
    base_df = load_base_training_data(client)
    t1_df = load_t1_data(client)
    combined_df = combine_datasets(base_df, t1_df)

    # ── Prepare features ──────────────────────────────────────────────────────
    X, y, w = prepare_features(combined_df)
    feature_names = list(X.columns)

    # ── Scale ─────────────────────────────────────────────────────────────────
    logger.info("Fitting scaler...")
    scaler, X_scaled = build_scaler(X)

    # ── Train/val split ───────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val, w_train, w_val = train_val_split(X_scaled, y, w)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train, w_train, X_val, y_val)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df = compute_feature_importance(model, feature_names)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, classification_report

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_val, val_proba)
    logger.info(f"Validation AUC-ROC: {auc:.4f}")
    logger.info("Classification report (val):")
    for line in classification_report(y_val, val_pred).split("\n"):
        if line.strip():
            logger.info(f"  {line}")

    # ── Training stats for metadata ───────────────────────────────────────────
    training_stats = {
        "n_total_samples": len(combined_df),
        "n_base_samples": len(base_df),
        "n_t1_samples": len(t1_df) if not t1_df.empty else 0,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": float((y == 1).mean()),
        "val_auc_roc": float(auc),
        "base_sample_weight": BASE_CSV_WEIGHT,
        "t1_sample_weight": T1_WEIGHT,
        "equal_weight_applied": len(t1_df) >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT if not t1_df.empty else False,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(model, scaler, fi_df, feature_names, training_stats)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RETRAIN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total samples    : {training_stats['n_total_samples']}")
    logger.info(f"  Base CSV samples : {training_stats['n_base_samples']}")
    logger.info(f"  T-1 samples      : {training_stats['n_t1_samples']}")
    logger.info(f"  Positive rate    : {training_stats['positive_rate']:.1%}")
    logger.info(f"  Validation AUC   : {auc:.4f}")
    logger.info(f"  Best iteration   : {model.best_iteration}")
    logger.info(f"  Features         : {len(feature_names)}")
    logger.info("")
    logger.info("Files written:")
    logger.info(f"  {MODEL_PATH}")
    logger.info(f"  {SCALER_PATH}")
    logger.info(f"  {METADATA_PATH}")
    logger.info(f"  {FEATURE_IMPORTANCE_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
