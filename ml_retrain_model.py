"""
ML Model Retraining Script - FIXED VERSION
Implements incremental learning with correct sample weighting

Key fixes:
1. Only uses T-1 data (no same-day leakage)
2. Includes non-winners (negative examples)
3. Historical weight = 10.0 (preserves T-3, T-5, T-10 patterns)
4. Uses joblib for model persistence
"""

import logging
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import load_config, setup_logging
from src.ml_predictor.model_trainer import ModelTrainer
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


def main():
    parser = argparse.ArgumentParser(description='Retrain ML prediction model')
    parser.add_argument('--lookback-days', type=int, default=90,
                       help='Days of new data to include (default: 90)')
    parser.add_argument('--use-all-timepoints', action='store_true', default=True,
                       help='Use both T-1 close and T-1 open data')
    parser.add_argument('--no-non-winners', action='store_true',
                       help='Exclude non-winners (NOT recommended - causes bias)')
    parser.add_argument('--historical-weight', type=float, default=10.0,
                       help='Weight for historical samples (default: 10.0 to preserve T-lag patterns)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("ML MODEL RETRAINING - INCREMENTAL LEARNING")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Lookback days: {args.lookback_days}")
    logger.info(f"Use all timepoints: {args.use_all_timepoints}")
    logger.info(f"Include non-winners: {not args.no_non_winners}")
    logger.info(f"Historical weight: {args.historical_weight}")
    
    if args.no_non_winners:
        logger.warning("⚠️  Training WITHOUT non-winners - this causes selection bias!")
        logger.warning("⚠️  Model will have high false positive rate!")
    
    if args.historical_weight < 5.0:
        logger.warning(f"⚠️  Historical weight {args.historical_weight} is LOW!")
        logger.warning("⚠️  Model may forget T-3, T-5, T-10 patterns!")
        logger.warning("⚠️  Recommended: 10.0 or higher")
    
    try:
        # Load config
        config = load_config()
        
        # Initialize trainer and database client
        trainer = ModelTrainer(config)
        supabase = MLPredictionSupabaseClient(config)
        
        # ===== STEP 1: Load Historical Training Data =====
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Loading Historical Training Data")
        logger.info("="*80)
        
        historical_X, historical_y, historical_metadata = trainer.load_historical_training_data()
        
        if historical_X is None or historical_X.empty:
            logger.warning("No historical data found!")
            logger.warning("Proceeding with ONLY new daily data (not ideal)")
            logger.info("\nTo add historical data:")
            logger.info("  1. Save your original research data as:")
            logger.info("     ml_models/historical_data/original_training_data.pkl")
            logger.info("  2. Format: joblib dump with {'X': df, 'y': series, 'metadata': dict}")
        else:
            logger.info(f"✓ Historical data loaded: {len(historical_X)} samples")
            logger.info(f"  This preserves your T-3, T-5, T-10 lag patterns")
        
        # ===== STEP 2: Prepare New Training Data from Daily Systems =====
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Preparing New Training Data")
        logger.info("="*80)
        logger.info("Source: Daily Winners + Daily Non-Winners")
        logger.info("Strategy: T-1 data ONLY (no same-day leakage)")
        
        new_X, new_y, new_metadata = trainer.prepare_training_data_from_daily_winners(
            supabase,
            lookback_days=args.lookback_days,
            use_all_timepoints=args.use_all_timepoints,
            include_non_winners=not args.no_non_winners
        )
        
        if new_X.empty:
            logger.error("No new training data available!")
            logger.error("Possible causes:")
            logger.error("  - No daily winners data collected")
            logger.error("  - No non-winners data collected (if enabled)")
            logger.error("  - Database connection issue")
            return 1
        
        logger.info(f"✓ New training data prepared: {len(new_X)} samples")
        logger.info(f"  Data source: {new_metadata.get('source', 'unknown')}")
        logger.info(f"  Includes negatives: {new_metadata.get('includes_negative_examples', False)}")
        logger.info(f"  Uses T-1 only: {new_metadata.get('uses_only_t1_data', False)}")
        logger.info(f"  No leakage: {new_metadata.get('same_day_data_excluded', False)}")
        
        # ===== STEP 3: Combine Historical and New Data =====
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Combining Historical + New Data")
        logger.info("="*80)
        logger.info(f"Historical weight: {args.historical_weight}x")
        logger.info(f"New data weight: 1.0x")
        logger.info("This preserves T-3, T-5, T-10 patterns while learning T-1")
        
        X_combined, y_combined, combined_metadata, sample_weights = trainer.combine_training_data(
            historical_X,
            historical_y,
            new_X,
            new_y,
            historical_weight=args.historical_weight
        )
        
        if X_combined.empty:
            logger.error("Failed to combine training data!")
            return 1
        
        logger.info(f"✓ Combined training data: {len(X_combined)} samples")
        
        # ===== STEP 4: Train Model =====
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Training XGBoost Model")
        logger.info("="*80)
        
        results = trainer.train_model(
            X_combined,
            y_combined,
            sample_weights=sample_weights,
            use_time_series_split=True,
            test_size=args.test_size
        )
        
        # ===== STEP 5: Calculate Feature Importance =====
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Analyzing Feature Importance")
        logger.info("="*80)
        
        importance_df = trainer.calculate_feature_importance(
            results['model'],
            results['feature_names']
        )
        
        logger.info(f"\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        # ===== STEP 6: Save Model =====
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Saving Model")
        logger.info("="*80)
        
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine all metadata
        full_metadata = {
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'training_config': {
                'lookback_days': args.lookback_days,
                'use_all_timepoints': args.use_all_timepoints,
                'include_non_winners': not args.no_non_winners,
                'historical_weight': args.historical_weight,
                'test_size': args.test_size
            },
            'data_sources': {
                'historical': historical_metadata if historical_metadata else {},
                'new_daily': new_metadata,
                'combined': combined_metadata
            },
            'performance': {
                'train_accuracy': results['train_accuracy'],
                'test_accuracy': results['test_accuracy'],
                'train_auc': results['train_auc'],
                'test_auc': results['test_auc'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'true_positives': results['true_positives'],
                'false_positives': results['false_positives'],
                'true_negatives': results['true_negatives'],
                'false_negatives': results['false_negatives']
            },
            'feature_names': results['feature_names'],
            'n_features': len(results['feature_names'])
        }
        
        trainer.save_model(
            results['model'],
            results['scaler'],
            full_metadata,
            version=version
        )
        
        # ===== SUMMARY =====
        logger.info("\n" + "="*80)
        logger.info("RETRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Model version: {version}")
        logger.info(f"✓ Total training samples: {len(X_combined)}")
        logger.info(f"✓ Test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"✓ Test AUC: {results['test_auc']:.4f}")
        logger.info(f"✓ Precision: {results['precision']:.4f}")
        logger.info(f"✓ Recall: {results['recall']:.4f}")
        logger.info(f"✓ F1 Score: {results['f1_score']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"  True Positives:  {results['true_positives']}")
        logger.info(f"  False Positives: {results['false_positives']}")
        logger.info(f"  True Negatives:  {results['true_negatives']}")
        logger.info(f"  False Negatives: {results['false_negatives']}")
        
        if not args.no_non_winners and new_metadata.get('includes_negative_examples'):
            logger.info("\n✓ Model trained with negative examples (non-winners)")
            logger.info("  Expected improvements:")
            logger.info("  - Higher precision (fewer false positives)")
            logger.info("  - Better discrimination")
            logger.info("  - More reliable predictions")
        else:
            logger.warning("\n⚠️  Model trained WITHOUT negative examples")
            logger.warning("  This may result in:")
            logger.warning("  - Lower precision (more false positives)")
            logger.warning("  - Poor discrimination")
            logger.warning("  - Less reliable predictions")
        
        if args.historical_weight >= 10.0:
            logger.info("\n✓ Historical weight is optimal (10.0+)")
            logger.info("  T-3, T-5, T-10 patterns preserved")
        else:
            logger.warning(f"\n⚠️  Historical weight is {args.historical_weight}")
            logger.warning("  Model may gradually forget time-lag patterns")
        
        logger.info(f"\nFinished at: {datetime.now()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
