#!/usr/bin/env python3
"""
Retrain ML model using recent actual data
SELF-LEARNING: Uses all available historical data
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config
from src.ml_predictor.model_trainer import ModelTrainer
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


def main():
    parser = argparse.ArgumentParser(description="Retrain ML explosion model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lookback-days", type=int, default=90, 
                       help="Days of data to use for training")
    parser.add_argument("--use-all-timepoints", action="store_true",
                       help="Use all timepoints (day_prior_close + day_prior_open)")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, config.get("logging", {}))
    
    logger.info("="*60)
    logger.info("ML MODEL RETRAINING - SELF-LEARNING")
    logger.info("="*60)
    
    # Initialize
    trainer = ModelTrainer(config)
    supabase = MLPredictionSupabaseClient(config)
    
    # Prepare training data from Daily Winners system
    logger.info(f"\nPreparing training data (last {args.lookback_days} days)...")
    logger.info(f"Using all timepoints: {args.use_all_timepoints}")
    
    try:
        X, y, metadata = trainer.prepare_training_data_from_daily_winners(
            supabase,
            lookback_days=args.lookback_days,
            use_all_timepoints=args.use_all_timepoints
        )
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        return 1
    
    logger.info(f"\nTraining data prepared:")
    logger.info(f"  Samples: {metadata['n_samples']}")
    logger.info(f"  Features: {metadata['feature_count']}")
    logger.info(f"  Positive samples: {metadata['n_positives']}")
    logger.info(f"  Negative samples: {metadata['n_negatives']}")
    logger.info(f"  Positive rate: {metadata['positive_rate']*100:.2f}%")
    logger.info(f"  Date range: {metadata['date_range']}")
    logger.info(f"  Timepoints: {', '.join(metadata['timepoints_used'])}")
    
    # Train model
    logger.info("\nTraining new model...")
    try:
        results = trainer.train_model(X, y, use_time_series_split=True)
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return 1
    
    # Calculate feature importance
    logger.info("\nCalculating feature importance...")
    importance_df = trainer.calculate_feature_importance(
        results['model'],
        results['feature_names']
    )
    
    logger.info("\nTop 20 Important Features:")
    for idx, row in importance_df.head(20).iterrows():
        logger.info(f"  {idx+1:2d}. {row['feature']:35s}: {row['importance']:.6f}")
    
    # Save model
    logger.info("\nSaving new model...")
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        trainer.save_model(
            results['model'],
            results['scaler'],
            metadata,
            version
        )
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return 1
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("✓ MODEL RETRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"\nModel Version: {version}")
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Train Accuracy: {results['train_accuracy']:.2%}")
    logger.info(f"  Test Accuracy: {results['test_accuracy']:.2%}")
    logger.info(f"  Train AUC: {results['train_auc']:.4f}")
    logger.info(f"  Test AUC: {results['test_auc']:.4f}")
    logger.info(f"  Precision: {results['precision']:.2%}")
    logger.info(f"  Recall: {results['recall']:.2%}")
    logger.info(f"  F1 Score: {results['f1_score']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Positives:  {results['true_positives']}")
    logger.info(f"  False Positives: {results['false_positives']}")
    logger.info(f"  True Negatives:  {results['true_negatives']}")
    logger.info(f"  False Negatives: {results['false_negatives']}")
    
    logger.info(f"\nModel files saved to: {trainer.model_dir}")
    logger.info(f"Old model archived to: {trainer.archive_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
