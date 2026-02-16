#!/usr/bin/env python3
"""
ML Model Retraining Script - MULTI-TIMEPOINT VERSION
Trains model on ALL timepoint data (T-1 close + open)
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
    parser = argparse.ArgumentParser(description='Retrain MULTI-TIMEPOINT model')
    parser.add_argument('--lookback-days', type=int, default=90)
    parser.add_argument('--no-non-winners', action='store_true')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("ML MODEL RETRAINING - MULTI-TIMEPOINT")
    logger.info("="*80)
    logger.info(f"Lookback days: {args.lookback_days}")
    logger.info(f"Include non-winners: {not args.no_non_winners}")
    logger.info("Model will learn from T-1 open AND close timepoints")
    
    try:
        config = load_config()
        
        trainer = ModelTrainer(config)
        supabase = MLPredictionSupabaseClient(config)
        
        # Prepare multi-timepoint training data
        X, y, metadata = trainer.prepare_multi_timepoint_training_data(
            supabase,
            lookback_days=args.lookback_days,
            include_non_winners=not args.no_non_winners
        )
        
        if X.empty:
            logger.error("No training data available!")
            return 1
        
        # Train model
        results = trainer.train_model(X, y, test_size=args.test_size)
        
        # Calculate feature importance
        importance_df = trainer.calculate_feature_importance(
            results['model'],
            results['feature_names']
        )
        
        logger.info(f"\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']:40s}: {row['importance']:.4f}")
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        full_metadata = {
            'version': version,
            'trained_at': datetime.now().isoformat(),
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
            'is_multi_timepoint': True
        }
        
        trainer.save_model(
            results['model'],
            results['scaler'],
            full_metadata,
            version=version
        )
        
        logger.info("\n" + "="*80)
        logger.info("✓ MULTI-TIMEPOINT MODEL TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Model version: {version}")
        logger.info(f"Total features: {len(results['feature_names'])}")
        logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Test AUC: {results['test_auc']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
