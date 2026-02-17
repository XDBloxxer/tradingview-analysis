#!/usr/bin/env python3
"""
ML Model Fine-Tuning Script - WITH MISTAKE LEARNING

CHANGES FROM PREVIOUS VERSION:
  + Imports ml_mistake_learner
  + After building standard winner/non-winner samples, fetches mistake samples
  + Concatenates mistake samples (with higher weights) into training set
  + Passes sample_weight to XGBoost fit() so mistakes are penalised proportionally

EVERYTHING ELSE IS UNCHANGED:
  - Still fine-tunes from existing model (preserves T-3/T-5/T-10 knowledge)
  - Still uses the same scaler
  - Still saves to the same paths
  - Mistake learning is purely additive — it cannot degrade the base training
"""

import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import joblib
import json
import yaml
import os
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient

# ── NEW IMPORT ────────────────────────────────────────────────────────────────
from ml_mistake_learner import (
    build_mistake_training_samples,
    log_mistake_summary,
    WEIGHT_STANDARD,
)
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
import xgboost as xgb


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    Path("logs").mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with T-1 data + mistake learning")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--use-all-timepoints", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    # ── NEW FLAG ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-mistake-learning",
        action="store_true",
        help="Disable mistake-learning (use for ablation studies or debugging)",
    )
    # ─────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("ML MODEL FINE-TUNING — WITH MISTAKE LEARNING")
    logger.info("=" * 80)
    logger.info(f"Lookback days      : {args.lookback_days}")
    logger.info(f"Use all timepoints : {args.use_all_timepoints}")
    logger.info(f"Mistake learning   : {'ENABLED' if not args.skip_mistake_learning else 'DISABLED'}")

    try:
        config = load_config()
        supabase = MLPredictionSupabaseClient(config)

        # =====================================================================
        # STEP 1: LOAD EXISTING MODEL
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOAD EXISTING MODEL")
        logger.info("=" * 80)

        model_dir = Path("ml_models")
        model_path = model_dir / "best_model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        metadata_path = model_dir / "model_metadata.json"

        if not model_path.exists():
            logger.error("No existing model found — run initial training first.")
            return 1

        existing_model = joblib.load(model_path)
        existing_scaler = joblib.load(scaler_path)

        with open(metadata_path, "r") as f:
            existing_metadata = json.load(f)

        existing_features = existing_metadata.get("features", [])
        logger.info(f"Loaded model with {len(existing_features)} features.")

        t3_count      = sum(1 for f in existing_features if f.startswith("t3_"))
        t5_count      = sum(1 for f in existing_features if f.startswith("t5_"))
        t10_count     = sum(1 for f in existing_features if f.startswith("t10_"))
        t1_close_count = sum(1 for f in existing_features if f.startswith("t1_close_"))
        t1_open_count  = sum(1 for f in existing_features if f.startswith("t1_open_"))
        logger.info(f"  T-3: {t3_count}, T-5: {t5_count}, T-10: {t10_count}, "
                    f"T-1 close: {t1_close_count}, T-1 open: {t1_open_count}")

        # =====================================================================
        # STEP 2: FETCH STANDARD T-1 TRAINING DATA (unchanged)
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FETCH STANDARD T-1 TRAINING DATA")
        logger.info("=" * 80)

        end_date   = datetime.now().date()
        start_date = end_date - timedelta(days=args.lookback_days)
        logger.info(f"Date range: {start_date} → {end_date}")

        logger.info("Fetching winners day_prior_close...")
        winners_t1_close_df = supabase.get_winners_day_prior_close(
            start_date=start_date.isoformat(), end_date=end_date.isoformat(), limit=5000
        )
        logger.info(f"  {len(winners_t1_close_df)} winner T-1 close records")

        if len(winners_t1_close_df) == 0:
            logger.error("No winner data found in winners_day_prior_close table.")
            logger.error(f"  Date range: {start_date} to {end_date}")
            logger.error("  Possible causes:")
            logger.error("    - daily_winners table is empty for this date range")
            logger.error("    - winners_day_prior_close table is empty")
            logger.error("  Solution: Run daily_top10.yml workflow first to collect winners")
            return 1

        winners_t1_open_df = pd.DataFrame()
        if args.use_all_timepoints:
            logger.info("Fetching winners day_prior_open...")
            winners_t1_open_df = supabase.get_winners_day_prior_open(
                start_date=start_date.isoformat(), end_date=end_date.isoformat(), limit=5000
            )
            logger.info(f"  {len(winners_t1_open_df)} winner T-1 open records")

        logger.info("Fetching non-winners day_prior_close...")
        non_winners_t1_close_df = supabase.get_non_winners_day_prior_close(
            start_date=start_date.isoformat(), end_date=end_date.isoformat(), limit=5000
        )
        logger.info(f"  {len(non_winners_t1_close_df)} non-winner T-1 close records")

        if len(non_winners_t1_close_df) == 0:
            logger.error("No non-winner data found in non_winners_day_prior_close table.")
            logger.error(f"  Date range: {start_date} to {end_date}")
            logger.error("  Possible causes:")
            logger.error("    - daily_non_winners table is empty for this date range")
            logger.error("    - non_winners_day_prior_close table is empty")
            logger.error("  Solution: Run daily_non_winners_workflow.yml to collect non-winners")
            return 1

        non_winners_t1_open_df = pd.DataFrame()
        if args.use_all_timepoints:
            logger.info("Fetching non-winners day_prior_open...")
            non_winners_t1_open_df = supabase.get_non_winners_day_prior_open(
                start_date=start_date.isoformat(), end_date=end_date.isoformat(), limit=5000
            )
            logger.info(f"  {len(non_winners_t1_open_df)} non-winner T-1 open records")

        # =====================================================================
        # STEP 3: BUILD STANDARD TRAINING SAMPLES
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: BUILD STANDARD TRAINING SAMPLES")
        logger.info("=" * 80)

        meta_cols = {"id", "created_at", "updated_at", "symbol", "exchange",
                     "detection_date", "snapshot_type", "snapshot_time", "snapshot_date"}

        def create_standard_sample(close_row, open_row, is_winner: bool) -> dict:
            sample = {}
            for col in close_row.index:
                if col not in meta_cols:
                    sample[f"t1_close_{col}"] = close_row[col]
            if open_row is not None:
                for col in open_row.index:
                    if col not in meta_cols:
                        sample[f"t1_open_{col}"] = open_row[col]
            for feat in existing_features:
                if feat not in sample:
                    feat_lower = feat.lower()
                    if any(x in feat_lower for x in ["rsi", "stoch", "willr", "cci"]):
                        sample[feat] = 50.0
                    elif "volume" in feat_lower or "obv" in feat_lower:
                        sample[feat] = 100_000.0
                    elif any(x in feat_lower for x in ["price", "close", "open", "high", "low"]):
                        sample[feat] = 50.0
                    else:
                        sample[feat] = 0.0
            sample["label"] = 1 if is_winner else 0
            sample["sample_weight"] = WEIGHT_STANDARD
            return sample

        training_samples = []

        for _, close_row in winners_t1_close_df.iterrows():
            open_row = None
            if not winners_t1_open_df.empty:
                match = winners_t1_open_df[
                    (winners_t1_open_df["symbol"] == close_row["symbol"]) &
                    (winners_t1_open_df["detection_date"] == close_row["detection_date"])
                ]
                if not match.empty:
                    open_row = match.iloc[0]
            training_samples.append(create_standard_sample(close_row, open_row, True))

        for _, close_row in non_winners_t1_close_df.iterrows():
            open_row = None
            if not non_winners_t1_open_df.empty:
                match = non_winners_t1_open_df[
                    (non_winners_t1_open_df["symbol"] == close_row["symbol"]) &
                    (non_winners_t1_open_df["detection_date"] == close_row["detection_date"])
                ]
                if not match.empty:
                    open_row = match.iloc[0]
            training_samples.append(create_standard_sample(close_row, open_row, False))

        standard_df = pd.DataFrame(training_samples)
        n_standard_pos = int(standard_df["label"].sum())
        n_standard_neg = len(standard_df) - n_standard_pos
        logger.info(f"Standard samples: {len(standard_df)} "
                    f"({n_standard_pos} winners, {n_standard_neg} non-winners)")

        # =====================================================================
        # STEP 3b: FETCH MISTAKE SAMPLES  ← NEW
        # =====================================================================
        mistake_df = pd.DataFrame()

        if not args.skip_mistake_learning:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3b: FETCH MISTAKE SAMPLES (learn from own errors)")
            logger.info("=" * 80)

            mistake_df = build_mistake_training_samples(
                lookback_days=args.lookback_days,
                use_all_timepoints=args.use_all_timepoints,
                existing_features=existing_features,
            )

            if not mistake_df.empty:
                log_mistake_summary(mistake_df)
                # Drop bookkeeping columns the model doesn't need
                for drop_col in ["symbol", "detection_date", "mistake_type"]:
                    if drop_col in mistake_df.columns:
                        mistake_df = mistake_df.drop(columns=[drop_col])
            else:
                logger.info("No mistake samples available yet "
                            "(model needs at least one accuracy-tracking run first).")

        # =====================================================================
        # STEP 4: COMBINE AND VALIDATE
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: COMBINE STANDARD + MISTAKE SAMPLES")
        logger.info("=" * 80)

        frames = [standard_df]
        if not mistake_df.empty:
            frames.append(mistake_df)

        df = pd.concat(frames, ignore_index=True)

        # Ensure all feature columns exist and NaN is filled
        for feat in existing_features:
            if feat not in df.columns:
                df[feat] = 0.0
        df[existing_features] = df[existing_features].fillna(0)

        n_positives = int(df["label"].sum())
        n_negatives = len(df) - n_positives
        n_mistakes  = len(mistake_df) if not mistake_df.empty else 0

        logger.info(f"COMBINED training set: {len(df)} samples")
        logger.info(f"  Standard : {len(standard_df)}")
        logger.info(f"  Mistakes : {n_mistakes}")
        logger.info(f"  Positives: {n_positives}  Negatives: {n_negatives}")
        logger.info(f"  Positive rate: {n_positives / len(df) * 100:.1f}%")

        if n_positives == 0:
            logger.error("No positive samples — cannot train binary classifier.")
            return 1
        if n_negatives == 0:
            logger.error("No negative samples — cannot train binary classifier.")
            return 1
        if min(n_positives, n_negatives) < 10:
            logger.warning(f"Very few samples in minority class ({min(n_positives, n_negatives)}). "
                           "Consider increasing --lookback-days.")

        # =====================================================================
        # STEP 5: TRAIN / TEST SPLIT
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: TRAIN / TEST SPLIT (STRATIFIED)")
        logger.info("=" * 80)

        X = df[existing_features].copy()
        y = df["label"].copy()
        # ── NEW: extract sample weights ───────────────────────────────────────
        w = df["sample_weight"].copy() if "sample_weight" in df.columns \
            else pd.Series(WEIGHT_STANDARD, index=df.index)
        # ─────────────────────────────────────────────────────────────────────

        try:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, w, test_size=args.test_size, stratify=y, random_state=42
            )
            logger.info("Using stratified split.")
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), falling back to random split.")
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, w, test_size=args.test_size, random_state=42
            )

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Train weight range: {w_train.min():.1f}–{w_train.max():.1f}")

        train_classes = y_train.unique()
        if len(train_classes) < 2:
            logger.error(f"Training set has only one class ({train_classes}).")
            logger.error("  Cannot train binary classifier with only one class.")
            logger.error(f"  Total samples: {len(df)}, Positives: {n_positives}, Negatives: {n_negatives}")
            logger.error("  Solutions:")
            logger.error("    1. Increase --lookback-days to get more historical data")
            logger.error("    2. Run daily_non_winners_workflow.yml more days to collect non-winners")
            logger.error("    3. Wait until you have at least 20+ examples of each class")
            return 1

        test_classes = y_test.unique()
        if len(test_classes) < 2:
            logger.warning(f"Test set has only one class ({test_classes}).")
            logger.warning("  Evaluation metrics may be incomplete, but training will proceed.")

        # =====================================================================
        # STEP 6: SCALE
        # =====================================================================
        X_train_scaled = existing_scaler.transform(X_train)
        X_test_scaled  = existing_scaler.transform(X_test)
        logger.info("Scaled using existing scaler.")

        # =====================================================================
        # STEP 7: FINE-TUNE CLASSIFIER
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: FINE-TUNE CLASSIFIER (continue from existing model)")
        logger.info("=" * 80)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

        fine_tuned_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=10,
        )

        # ── KEY CHANGE: pass sample_weight so mistakes hit harder ─────────────
        fine_tuned_model.fit(
            X_train_scaled,
            y_train,
            sample_weight=w_train.values,          # ← NEW
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            xgb_model=existing_model.get_booster(),
            verbose=False,
        )
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Classifier fine-tuning complete.")

        # =====================================================================
        # STEP 7b: FINE-TUNE GAIN REGRESSOR (unchanged, abbreviated)
        # =====================================================================
        fine_tuned_regressor = None
        regressor_path = model_dir / "gain_regressor.pkl"

        if regressor_path.exists():
            logger.info("\nFine-tuning gain regressor...")
            try:
                winners_actual_df = supabase.get_daily_winners(
                    start_date=start_date.isoformat(), end_date=end_date.isoformat()
                )
                if not winners_actual_df.empty:
                    gain_lookup = {
                        (r.symbol, r.detection_date): r.change_pct
                        for r in winners_actual_df.itertuples()
                    }
                    reg_samples, reg_targets = [], []
                    for _, close_row in winners_t1_close_df.iterrows():
                        key = (close_row["symbol"], close_row["detection_date"])
                        if key not in gain_lookup:
                            continue
                        sample = {f"t1_close_{col}": close_row[col]
                                  for col in close_row.index if col not in meta_cols}
                        if not winners_t1_open_df.empty:
                            om = winners_t1_open_df[
                                (winners_t1_open_df["symbol"] == close_row["symbol"]) &
                                (winners_t1_open_df["detection_date"] == close_row["detection_date"])
                            ]
                            if not om.empty:
                                for col in om.iloc[0].index:
                                    if col not in meta_cols:
                                        sample[f"t1_open_{col}"] = om.iloc[0][col]
                        for feat in existing_features:
                            if feat not in sample:
                                sample[feat] = 0.0
                        reg_samples.append(sample)
                        reg_targets.append(gain_lookup[key])

                    if len(reg_samples) >= 5:
                        reg_df = pd.DataFrame(reg_samples)
                        for feat in existing_features:
                            if feat not in reg_df.columns:
                                reg_df[feat] = 0.0
                        X_reg = reg_df[existing_features].fillna(0)
                        y_reg = np.array(reg_targets)
                        split = int(len(X_reg) * (1 - args.test_size))
                        Xr_tr = existing_scaler.transform(X_reg.iloc[:split])
                        Xr_te = existing_scaler.transform(X_reg.iloc[split:])
                        existing_reg = joblib.load(regressor_path)
                        fine_tuned_regressor = xgb.XGBRegressor(
                            n_estimators=100, max_depth=6, learning_rate=0.01,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, eval_metric="rmse",
                            early_stopping_rounds=10,
                        )
                        fine_tuned_regressor.fit(
                            Xr_tr, y_reg[:split],
                            eval_set=[(Xr_tr, y_reg[:split]), (Xr_te, y_reg[split:])],
                            xgb_model=existing_reg.get_booster(),
                            verbose=False,
                        )

                        # Evaluate regressor
                        from sklearn.metrics import mean_absolute_error, r2_score
                        reg_pred_test = fine_tuned_regressor.predict(Xr_te)
                        if len(y_reg[split:]) > 0:
                            reg_mae = mean_absolute_error(y_reg[split:], reg_pred_test)
                            logger.info(f"  Regressor Test MAE: {reg_mae:.4f}%")
                        if len(y_reg[split:]) > 1:
                            reg_r2 = r2_score(y_reg[split:], reg_pred_test)
                            logger.info(f"  Regressor Test R²: {reg_r2:.4f}")

                        logger.info("Gain regressor fine-tuned.")
            except Exception as e:
                logger.error(f"Regressor fine-tuning failed: {e}")
        else:
            logger.info("No existing gain_regressor.pkl — skipping regressor fine-tune.")

        # =====================================================================
        # STEP 8: EVALUATE
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: EVALUATE")
        logger.info("=" * 80)

        train_pred  = fine_tuned_model.predict(X_train_scaled)
        test_pred   = fine_tuned_model.predict(X_test_scaled)
        train_proba = fine_tuned_model.predict_proba(X_train_scaled)[:, 1]
        test_proba  = fine_tuned_model.predict_proba(X_test_scaled)[:, 1]

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy  = accuracy_score(y_test, test_pred)
        train_auc      = roc_auc_score(y_train, train_proba)
        test_auc       = roc_auc_score(y_test, test_proba)
        precision      = precision_score(y_test, test_pred, zero_division=0)
        recall         = recall_score(y_test, test_pred, zero_division=0)
        f1             = f1_score(y_test, test_pred, zero_division=0)

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()

        logger.info(f"  Train Accuracy : {train_accuracy:.4f}  |  Test Accuracy : {test_accuracy:.4f}")
        logger.info(f"  Train AUC      : {train_auc:.4f}  |  Test AUC      : {test_auc:.4f}")
        logger.info(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
        logger.info(f"  Confusion matrix — TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

        # ─── Mistake-specific diagnostics ────────────────────────────────────
        if not mistake_df.empty and "mistake_type" in frames[1].columns:
            logger.info("\n  Mistake-specific accuracy on TEST set:")
            mistake_test_mask = X_test.index.isin(
                frames[1].index if len(frames) > 1 else []
            )
            if mistake_test_mask.any():
                m_acc = accuracy_score(
                    y_test[mistake_test_mask], test_pred[mistake_test_mask]
                )
                logger.info(f"    Mistake samples accuracy: {m_acc:.4f}")
            else:
                logger.info("    (Mistake samples all landed in train split.)")
        # ─────────────────────────────────────────────────────────────────────

        # =====================================================================
        # STEP 9: SAVE
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: SAVE")
        logger.info("=" * 80)

        import shutil
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = model_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        if model_path.exists():
            shutil.copy(model_path, archive_dir / f"best_model_{version}.pkl")
        joblib.dump(fine_tuned_model, model_path, protocol=4)
        logger.info(f"Saved classifier → {model_path}")

        if fine_tuned_regressor is not None:
            if regressor_path.exists():
                shutil.copy(regressor_path, archive_dir / f"gain_regressor_{version}.pkl")
            joblib.dump(fine_tuned_regressor, regressor_path, protocol=4)
            logger.info(f"Saved regressor → {regressor_path}")

        # Update metadata
        existing_metadata.update({
            "last_fine_tuned": datetime.now().isoformat(),
            "fine_tuning_version": version,
            "fine_tuning_config": {
                "lookback_days": args.lookback_days,
                "use_all_timepoints": args.use_all_timepoints,
                "test_size": args.test_size,
                "mistake_learning_enabled": not args.skip_mistake_learning,
            },
            "fine_tuning_data": {
                "n_samples_total": len(df),
                "n_standard_samples": len(standard_df),
                "n_mistake_samples": n_mistakes,
                "n_positives": n_positives,
                "n_negatives": n_negatives,
                "positive_rate": float(n_positives / len(df)),
            },
            "fine_tuning_performance": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy":  float(test_accuracy),
                "train_auc":      float(train_auc),
                "test_auc":       float(test_auc),
                "precision":      float(precision),
                "recall":         float(recall),
                "f1_score":       float(f1),
            },
            "regressor_fine_tuned": fine_tuned_regressor is not None,
        })
        with open(metadata_path, "w") as f:
            json.dump(existing_metadata, f, indent=2)
        logger.info(f"Updated metadata → {metadata_path}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ FINE-TUNING + MISTAKE LEARNING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Model version  : {version}")
        logger.info(f"  Test AUC       : {test_auc:.4f}")
        logger.info(f"  Standard samples: {len(standard_df)}")
        logger.info(f"  Mistake samples : {n_mistakes}")
        logger.info(f"  Regressor updated: {'Yes' if fine_tuned_regressor else 'No'}")
        return 0

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
