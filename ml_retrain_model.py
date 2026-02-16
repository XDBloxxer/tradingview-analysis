#!/usr/bin/env python3
"""
ML Model Fine-Tuning Script
PRESERVES existing T-3/T-5/T-10 knowledge
ADDS new T-1 open/close knowledge from database
"""

import logging
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging
from src.ml_predictor.model_trainer import ModelTrainer
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model with T-1 data')
    parser.add_argument('--lookback-days', type=int, default=90)
    parser.add_argument('--no-non-winners', action='store_true')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("ML MODEL FINE-TUNING")
    logger.info("="*80)
    logger.info(f"Lookback days: {args.lookback_days}")
    logger.info(f"Include non-winners: {not args.no_non_winners}")
    logger.info("")
    logger.info("STRATEGY:")
    logger.info("  1. Load existing model (has T-3/T-5/T-10 from CSV)")
    logger.info("  2. Fetch T-1 open/close from database (winners + non-winners)")
    logger.info("  3. Fine-tune model to ADD T-1 knowledge")
    logger.info("  4. Save expanded model (knows BOTH old and new features)")
    
    try:
        config = load_config()
        
        trainer = ModelTrainer(config)
        supabase = MLPredictionSupabaseClient(config)
        
        # Prepare T-1 fine-tuning data from database
        logger.info("\n" + "="*80)
        logger.info("STEP 1: FETCH T-1 DATA FROM DATABASE")
        logger.info("="*80)
        
        X, y, metadata = trainer.prepare_fine_tuning_data(
            supabase,
            lookback_days=args.lookback_days,
            include_non_winners=not args.no_non_winners
        )
        
        if X.empty:
            logger.error("No training data available!")
            return 1
        
        # Fine-tune model
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FINE-TUNE MODEL")
        logger.info("="*80)
        
        results = trainer.fine_tune_model(X, y, test_size=args.test_size)
        
        # Calculate feature importance
        importance_df = trainer.calculate_feature_importance(
            results['model'],
            results['feature_names']
        )
        
        # Save model
        logger.info("\n" + "="*80)
        logger.info("STEP 3: SAVE FINE-TUNED MODEL")
        logger.info("="*80)
        
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        full_metadata = {
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'training_type': 'fine_tuning',
            'training_config': {
                'lookback_days': args.lookback_days,
                'include_non_winners': not args.no_non_winners,
                'test_size': args.test_size
            },
            'data_source': metadata,
            'performance': {
                'train_accuracy': results['train_accuracy'],
                'test_accuracy': results['test_accuracy'],
                'train_auc': results['train_auc'],
                'test_auc': results['test_auc'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            },
            'features': results['feature_names'],
            'n_features': len(results['feature_names']),
            'existing_features': results['existing_features'],
            'new_features': results['new_features'],
            'n_existing_features': len(results['existing_features']),
            'n_new_features': len(results['new_features'])
        }
        
        trainer.save_model(
            results['model'],
            results['scaler'],
            full_metadata,
            version=version
        )
        
        logger.info("\n" + "="*80)
        logger.info("✓ FINE-TUNING COMPLETE")
        logger.info("="*80)
        logger.info(f"Model version: {version}")
        logger.info(f"Total features: {len(results['feature_names'])}")
        logger.info(f"  - Preserved (T-3/T-5/T-10): {len(results['existing_features'])}")
        logger.info(f"  - Added (T-1 open/close): {len(results['new_features'])}")
        logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Test AUC: {results['test_auc']:.4f}")
        logger.info("")
        logger.info("Model now accepts BOTH:")
        logger.info("  - Old CSV features (flat): Close, RSI_14, MACD_12_26_9, etc.")
        logger.info("  - New T-1 features (prefixed): t1_open_rsi, t1_close_macd, etc.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
