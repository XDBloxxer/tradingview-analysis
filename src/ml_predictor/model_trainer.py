#!/usr/bin/env python3
"""
model_trainer.py  —  Legacy T-1 Fine-Tuner

NOTE: This module is superseded by ml_retrain_model.py which performs a full
weekly retrain from scratch (faster, more correct, and includes mistake-learning).
Fine-tuning is retained here only as an emergency fallback for adding T-1 data
to a freshly restored model between scheduled retrains.

STRATEGY:
  1. Loads existing model (preserves T-3/T-5/T-10 knowledge).
  2. Fetches ONLY T-1 open/close data from Supabase.
  3. Uses XGBoost's xgb_model parameter to CONTINUE training (not start over).
  4. Model retains old knowledge while learning new T-1 patterns.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


# ---------------------------------------------------------------------------
# Helpers (previously imported from src.utils — now inlined)
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config, return empty dict if file not found."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.getLogger(__name__).warning(
            f"Config file '{config_path}' not found — using empty config."
        )
        return {}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_COLS = {
    "id", "created_at", "updated_at", "symbol", "exchange",
    "detection_date", "snapshot_type", "snapshot_time", "snapshot_date",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune model with T-1 data")
    parser.add_argument("--config",             default="config.yaml")
    parser.add_argument("--lookback-days",      type=int,   default=90)
    parser.add_argument("--use-all-timepoints", action="store_true",
                        help="Use both day_prior_close and day_prior_open")
    parser.add_argument("--test-size",          type=float, default=0.2)
    parser.add_argument("--verbose",            action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("ML MODEL FINE-TUNING (legacy fallback — prefer ml_retrain_model.py)")
    logger.info("=" * 80)

    config  = load_config(args.config)
    supabase = MLPredictionSupabaseClient(config)

    # ── STEP 1: Load existing model ───────────────────────────────────────
    model_dir     = Path("ml_models")
    model_path    = model_dir / "best_model.pkl"
    scaler_path   = model_dir / "scaler.pkl"
    metadata_path = model_dir / "model_metadata.json"

    if not model_path.exists():
        logger.error("❌ No existing model found. Run ml_retrain_model.py first.")
        return 1

    existing_model  = joblib.load(model_path)
    existing_scaler = joblib.load(scaler_path)

    with open(metadata_path) as f:
        existing_metadata = json.load(f)

    # "features" is the full list; "feature_names_sample" is 20-item preview
    existing_features = (
        existing_metadata.get("features")
        or existing_metadata.get("feature_names_sample")
        or []
    )

    logger.info(f"✓ Loaded existing model: {len(existing_features)} features")

    # ── STEP 2: Fetch T-1 data ────────────────────────────────────────────
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=args.lookback_days)
    logger.info(f"Date range: {start_date} → {end_date}")

    winners_close    = supabase.get_winners_day_prior_close(
        start_date.isoformat(), end_date.isoformat(), limit=5000)
    non_winners_close = supabase.get_non_winners_day_prior_close(
        start_date.isoformat(), end_date.isoformat(), limit=5000)

    winners_open = non_winners_open = pd.DataFrame()
    if args.use_all_timepoints:
        winners_open      = supabase.get_winners_day_prior_open(
            start_date.isoformat(), end_date.isoformat(), limit=5000)
        non_winners_open  = supabase.get_non_winners_day_prior_open(
            start_date.isoformat(), end_date.isoformat(), limit=5000)

    logger.info(f"Winners close: {len(winners_close)}, "
                f"non-winners close: {len(non_winners_close)}, "
                f"winners open: {len(winners_open)}, "
                f"non-winners open: {len(non_winners_open)}")

    # ── STEP 3: Apply column map so feature names match model schema ───────
    try:
        from t1_column_map import rename_t1_columns
        for df_var, prefix in [
            (winners_close,     "t1_close"),
            (non_winners_close, "t1_close"),
            (winners_open,      "t1_open"),
            (non_winners_open,  "t1_open"),
        ]:
            if not df_var.empty:
                df_var[:] = rename_t1_columns(df_var, prefix=prefix)
    except ImportError:
        logger.warning("t1_column_map.py not found — T-1 feature names won't be renamed")

    # ── STEP 4: Build training samples ────────────────────────────────────
    def make_samples(close_df, open_df, is_winner: bool) -> list[dict]:
        samples = []
        for _, row in close_df.iterrows():
            sample = {}
            for col in row.index:
                if col not in META_COLS:
                    sample[col] = row[col]

            if not open_df.empty:
                match = open_df[
                    (open_df["symbol"]         == row.get("symbol")) &
                    (open_df["detection_date"] == row.get("detection_date"))
                ]
                if not match.empty:
                    for col in match.columns:
                        if col not in META_COLS and col not in sample:
                            sample[col] = match.iloc[0][col]

            # Pad missing model features with intelligent defaults
            for feat in existing_features:
                if feat not in sample:
                    fl = feat.lower()
                    if any(x in fl for x in ("rsi", "stoch", "willr", "cci")):
                        sample[feat] = 50.0
                    elif "volume" in fl or "obv" in fl:
                        sample[feat] = 100_000.0
                    elif any(x in fl for x in ("price", "close", "open", "high", "low")):
                        sample[feat] = 50.0
                    else:
                        sample[feat] = 0.0

            sample["label"] = 1 if is_winner else 0
            samples.append(sample)
        return samples

    all_samples = (
        make_samples(winners_close,     winners_open,     is_winner=True)
        + make_samples(non_winners_close, non_winners_open, is_winner=False)
    )

    if not all_samples:
        logger.error("❌ No training samples created.")
        return 1

    df = pd.DataFrame(all_samples)
    logger.info(f"Training samples: {len(df)} "
                f"(pos={int(df['label'].sum())}, "
                f"neg={int(len(df)-df['label'].sum())})")

    # ── STEP 5: Feature matrix ────────────────────────────────────────────
    for feat in existing_features:
        if feat not in df.columns:
            df[feat] = 0.0

    X = df[existing_features].copy().fillna(0)
    y = df["label"].copy()

    # ── STEP 6: Train/test split (chronological) ──────────────────────────
    split = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── STEP 7: Scale (reuse existing scaler) ─────────────────────────────
    X_train_s = existing_scaler.transform(X_train)
    X_test_s  = existing_scaler.transform(X_test)

    # ── STEP 8: Fine-tune ─────────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = n_neg / n_pos if n_pos else 1.0

    fine_tuned = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=10,
    )
    fine_tuned.fit(
        X_train_s, y_train,
        eval_set=[(X_train_s, y_train), (X_test_s, y_test)],
        xgb_model=existing_model.get_booster(),
        verbose=False,
    )
    logger.info("✓ Fine-tuning complete")

    # ── STEP 9: Evaluate ──────────────────────────────────────────────────
    test_proba = fine_tuned.predict_proba(X_test_s)[:, 1]
    test_pred  = fine_tuned.predict(X_test_s)
    auc        = roc_auc_score(y_test, test_proba)
    logger.info(f"Test AUC: {auc:.4f}  "
                f"Acc: {accuracy_score(y_test, test_pred):.4f}  "
                f"F1: {f1_score(y_test, test_pred, zero_division=0):.4f}")

    # ── STEP 10: Save ─────────────────────────────────────────────────────
    version     = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = model_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    shutil.copy(model_path, archive_dir / f"best_model_{version}.pkl")

    joblib.dump(fine_tuned, model_path, protocol=4)

    existing_metadata.update({
        "last_fine_tuned":    datetime.now().isoformat(),
        "fine_tuning_version": version,
        "fine_tuning_performance": {
            "test_auc": float(auc),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_f1":  float(f1_score(y_test, test_pred, zero_division=0)),
        },
    })
    with open(metadata_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)

    logger.info(f"✓ Saved fine-tuned model  → {model_path}")
    logger.info(f"✓ Updated metadata        → {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
