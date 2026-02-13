#!/usr/bin/env python3
"""
Retrain ML model using INCREMENTAL LEARNING
Combines historical research data + new daily winners data
PREVENTS CATASTROPHIC FORGETTING
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
    parser = argparse.ArgumentParser(description="Retrain ML explosion model with incremental learning")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lookback-days", type=int, default=90, 
                       help="Days of NEW data to use from daily winners")
    parser.add_argument("--use-all-timepoints", action="store_true",
                       help="Use all timepoints (day_prior_close + day_prior_open)")
    parser.add_argument("--historical-weight", type=float, default=0.7,
                       help="Weight for historical data (0.7 = 70%% importance)")
    parser.add_argument("--skip-historical", action="store_true",
                       help="Skip historical data (NOT RECOMMENDED)")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, config.get("logging", {}))
    
    logger.info("="*80)
    logger.info("ML MODEL RETRAINING - INCREMENTAL LEARNING")
    logger.info("="*80)
    logger.info("\nSTRATEGY:")
    logger.info("  1. Load original historical data (10k stocks × 2 years)")
    logger.info("  2. Add new data from daily winners (real-world performance)")
    logger.info("  3. Train on COMBINED dataset with weighted importance")
    logger.info("  4. Preserve time-lag patterns while learning from mistakes")
    
    # Initialize
    trainer = ModelTrainer(config)
    supabase = MLPredictionSupabaseClient(config)
    
    # ===== STEP 1: Load historical research data =====
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD HISTORICAL RESEARCH DATA")
    logger.info("="*80)
    
    if args.skip_historical:
        logger.warning("⚠️  SKIPPING HISTORICAL DATA (--skip-historical flag)")
        logger.warning("⚠️  Model will ONLY train on recent daily winners")
        logger.warning("⚠️  May lose time-lag analysis and deep pattern insights!")
        historical_X, historical_y, historical_metadata = None, None, None
    else:
        historical_X, historical_y, historical_metadata = trainer.load_historical_training_data()
        
        if historical_X is None:
            logger.warning("\n⚠️  NO HISTORICAL DATA FOUND!")
            logger.warning("⚠️  To preserve your original research:")
            logger.warning("⚠️  Save your 10k stocks × 2 years dataset as:")
            logger.warning("⚠️    ml_models/historical_data/original_training_data.pkl")
            logger.warning("⚠️  Format: {'X': DataFrame, 'y': Series, 'metadata': dict}")
            logger.warning("\n⚠️  Continuing with ONLY daily winners data...")
    
    # ===== STEP 2: Prepare new data from Daily Winners =====
    logger.info("\n" + "="*80)
    logger.info("STEP 2: PREPARE NEW DATA FROM DAILY WINNERS")
    logger.info("="*80)
    logger.info(f"Looking back {args.lookback_days} days")
    logger.info(f"Using all timepoints: {args.use_all_timepoints}")
    
    try:
        new_X, new_y, new_metadata = trainer.prepare_training_data_from_daily_winners(
            supabase,
            lookback_days=args.lookback_days,
            use_all_timepoints=args.use_all_timepoints
        )
    except Exception as e:
        logger.error(f"Failed to prepare new training data: {e}")
        return 1
    
    if new_X.empty:
        logger.error("No new training data available from daily winners")
        logger.error("Cannot train model without any data")
        return 1
    
    logger.info(f"\nNew training data prepared:")
    logger.info(f"  Samples: {new_metadata['n_samples']}")
    logger.info(f"  Features: {new_metadata['feature_count']}")
    logger.info(f"  Positive samples: {new_metadata['n_positives']}")
    logger.info(f"  Negative samples: {new_metadata['n_negatives']}")
    logger.info(f"  Positive rate: {new_metadata['positive_rate']*100:.2f}%")
    logger.info(f"  Date range: {new_metadata['date_range']}")
    logger.info(f"  Timepoints: {', '.join(new_metadata['timepoints_used'])}")
    
    # ===== STEP 3: Combine historical + new data =====
    logger.info("\n" + "="*80)
    logger.info("STEP 3: COMBINE HISTORICAL + NEW DATA")
    logger.info("="*80)
    
    if historical_X is not None and not historical_X.empty:
        logger.info(f"Historical weight: {args.historical_weight} (preserves original insights)")
        logger.info(f"New data weight: {1.0 - args.historical_weight} (real-world calibration)")
        
        try:
            X, y, combined_metadata, sample_weights = trainer.combine_training_data(
                historical_X, historical_y,
                new_X, new_y,
                historical_weight=args.historical_weight
            )
        except Exception as e:
            logger.error(f"Failed to combine data: {e}")
            return 1
    else:
        logger.warning("Using ONLY new data (no historical data available)")
        X = new_X
        y = new_y
        combined_metadata = new_metadata
        sample_weights = None
    
    # ===== STEP 4: Train model =====
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAIN MODEL")
    logger.info("="*80)
    
    try:
        results = trainer.train_model(
            X, y, 
            sample_weights=sample_weights,
            use_time_series_split=True
        )
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return 1
    
    # ===== STEP 5: Calculate feature importance =====
    logger.info("\n" + "="*80)
    logger.info("STEP 5: ANALYZE FEATURE IMPORTANCE")
    logger.info("="*80)
    
    importance_df = trainer.calculate_feature_importance(
        results['model'],
        results['feature_names']
    )
    
    logger.info("\nTop 20 Important Features:")
    for idx, row in importance_df.head(20).iterrows():
        logger.info(f"  {idx+1:2d}. {row['feature']:35s}: {row['importance']:.6f}")
    
    # ===== STEP 6: Save model =====
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SAVE MODEL")
    logger.info("="*80)
    
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine metadata
    final_metadata = {
        **combined_metadata,
        'features': results['feature_names'],
        'trained_at': datetime.now().isoformat(),
        'training_mode': 'incremental_learning',
        'historical_weight': args.historical_weight if sample_weights is not None else None,
        'lookback_days': args.lookback_days,
        'use_all_timepoints': args.use_all_timepoints
    }
    
    try:
        trainer.save_model(
            results['model'],
            results['scaler'],
            final_metadata,
            version
        )
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return 1
    
    # ===== Final summary =====
    logger.info("\n" + "="*80)
    logger.info("✓ INCREMENTAL LEARNING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nModel Version: {version}")
    
    if historical_X is not None and not historical_X.empty:
        logger.info(f"\nTraining Data Composition:")
        logger.info(f"  Historical samples: {combined_metadata.get('n_historical', 0)} ({args.historical_weight*100:.0f}% weight)")
        logger.info(f"  New samples: {combined_metadata.get('n_new', 0)} ({(1-args.historical_weight)*100:.0f}% weight)")
        logger.info(f"  Total samples: {combined_metadata['n_samples']}")
        logger.info(f"\n✓ Model preserves original research insights")
        logger.info(f"✓ Model learns from recent real-world performance")
    else:
        logger.info(f"\nTraining Data:")
        logger.info(f"  Total samples: {combined_metadata['n_samples']}")
        logger.info(f"\n⚠️  Model trained ONLY on recent data")
        logger.info(f"⚠️  May have lost time-lag pattern insights")
    
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
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS:")
    logger.info("="*80)
    logger.info("1. Test predictions: python ml_screen_and_predict.py --verbose")
    logger.info("2. Track accuracy: python ml_track_comprehensive_accuracy.py --verbose")
    logger.info("3. Monitor feature importance in ml_models/feature_importance.csv")
    
    if historical_X is None or historical_X.empty:
        logger.info("\n⚠️  IMPORTANT: Save your historical training data!")
        logger.info("   Format: {'X': DataFrame, 'y': Series, 'metadata': dict}")
        logger.info("   Location: ml_models/historical_data/original_training_data.pkl")
        logger.info("   This preserves your 10k stocks × 2 years research")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
