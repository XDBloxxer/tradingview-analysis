"""
ML Screening and Prediction Script - FIXED VERSION
Uses trained model to predict explosion candidates

Key changes:
1. Uses joblib for model loading (not pickle)
2. Proper error handling for missing models
3. Clear logging of prediction confidence
"""

import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import load_config, setup_logging
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient
from src.screener import TradingViewScreener
from src.intraday_data_collector import IntradayDataCollector


class MLPredictor:
    """ML-based stock explosion predictor"""
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_dir = Path("ml_models")
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metadata = None
        
    def load_model(self):
        """Load trained model and scaler - FIXED to use joblib"""
        
        model_path = self.model_dir / "best_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        metadata_path = self.model_dir / "model_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        # Load model and scaler using joblib (not pickle)
        self.logger.info("Loading model and scaler...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                self.model_metadata = json.load(f)
            
            self.feature_names = self.model_metadata.get('feature_names', [])
            
            self.logger.info(f"✓ Loaded model version: {self.model_metadata.get('version', 'unknown')}")
            self.logger.info(f"  Trained at: {self.model_metadata.get('trained_at', 'unknown')}")
            self.logger.info(f"  Test accuracy: {self.model_metadata.get('performance', {}).get('test_accuracy', 0):.4f}")
            self.logger.info(f"  Precision: {self.model_metadata.get('performance', {}).get('precision', 0):.4f}")
            self.logger.info(f"  Recall: {self.model_metadata.get('performance', {}).get('recall', 0):.4f}")
            
            # Check if model includes negative examples
            training_config = self.model_metadata.get('training_config', {})
            includes_negatives = training_config.get('include_non_winners', False)
            
            if includes_negatives:
                self.logger.info("  ✓ Model trained with negative examples (better discrimination)")
            else:
                self.logger.warning("  ⚠️  Model trained WITHOUT negative examples (may have high false positives)")
        else:
            self.logger.warning("Model metadata not found - using defaults")
            self.feature_names = []
    
    def predict(self, indicators: dict) -> tuple:
        """
        Predict explosion probability for a stock
        
        Returns:
            (probability, confidence_level, prediction_binary)
        """
        
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare features
        features = {}
        for feat in self.feature_names:
            features[feat] = indicators.get(feat, 0)
        
        # Create DataFrame
        X = pd.DataFrame([features])
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        probability = self.model.predict_proba(X_scaled)[0, 1]
        prediction = self.model.predict(X_scaled)[0]
        
        # Determine confidence level
        if probability >= 0.8:
            confidence = "VERY_HIGH"
        elif probability >= 0.7:
            confidence = "HIGH"
        elif probability >= 0.6:
            confidence = "MEDIUM"
        elif probability >= 0.5:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"
        
        return probability, confidence, prediction


def main():
    parser = argparse.ArgumentParser(description='ML-based stock explosion screening')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Number of top stocks to screen (default: 50)')
    parser.add_argument('--min-probability', type=float, default=0.6,
                       help='Minimum prediction probability (default: 0.6)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to database')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("ML STOCK EXPLOSION SCREENING")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Screening top {args.top_n} stocks")
    logger.info(f"Minimum probability: {args.min_probability}")
    
    try:
        # Load config
        config = load_config()
        
        # ===== STEP 1: Load ML Model =====
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Loading ML Model")
        logger.info("="*80)
        
        predictor = MLPredictor(config)
        predictor.load_model()
        
        # ===== STEP 2: Screen Stocks =====
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Screening Stocks")
        logger.info("="*80)
        
        screener = TradingViewScreener(config)
        
        logger.info("Running TradingView screener...")
        candidates = screener.screen_candidates(top_n=args.top_n)
        
        if not candidates:
            logger.warning("No candidates found from screener")
            return 1
        
        logger.info(f"✓ Found {len(candidates)} candidates")
        
        # ===== STEP 3: Collect Indicators =====
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Collecting Indicators")
        logger.info("="*80)
        
        collector = IntradayDataCollector(config)
        
        logger.info("Collecting current indicators...")
        symbols = [c['symbol'] for c in candidates]
        
        # Collect market close indicators (current snapshot)
        indicators_data = collector.collect_indicators(
            symbols,
            snapshot_type='market_close'
        )
        
        if not indicators_data:
            logger.warning("Failed to collect indicators")
            return 1
        
        logger.info(f"✓ Collected indicators for {len(indicators_data)} symbols")
        
        # ===== STEP 4: Make Predictions =====
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Making Predictions")
        logger.info("="*80)
        
        predictions = []
        
        for symbol, indicators in indicators_data.items():
            try:
                probability, confidence, prediction = predictor.predict(indicators)
                
                # Only include if probability meets threshold
                if probability >= args.min_probability:
                    predictions.append({
                        'symbol': symbol,
                        'probability': probability,
                        'confidence': confidence,
                        'prediction': prediction,
                        'indicators': indicators,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                continue
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        logger.info(f"\n✓ Made predictions for {len(predictions)} stocks")
        logger.info(f"  (Filtered by min probability: {args.min_probability})")
        
        # ===== STEP 5: Display Results =====
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Prediction Results")
        logger.info("="*80)
        
        if not predictions:
            logger.warning("No stocks met the minimum probability threshold")
            return 0
        
        logger.info(f"\nTop {min(20, len(predictions))} Predictions:")
        logger.info("-" * 80)
        logger.info(f"{'Symbol':<10} {'Probability':<12} {'Confidence':<15} {'Price':<10} {'Volume':<12}")
        logger.info("-" * 80)
        
        for pred in predictions[:20]:
            symbol = pred['symbol']
            prob = pred['probability']
            conf = pred['confidence']
            price = pred['indicators'].get('close', 0)
            volume = pred['indicators'].get('volume', 0)
            
            logger.info(f"{symbol:<10} {prob:>11.2%} {conf:<15} ${price:>8.2f} {volume:>11,d}")
        
        # ===== STEP 6: Save Predictions (Optional) =====
        if args.save_predictions:
            logger.info("\n" + "="*80)
            logger.info("STEP 6: Saving Predictions")
            logger.info("="*80)
            
            supabase = MLPredictionSupabaseClient(config)
            
            # Prepare prediction records
            prediction_records = []
            for pred in predictions:
                record = {
                    'symbol': pred['symbol'],
                    'prediction_date': datetime.now().date().isoformat(),
                    'probability': pred['probability'],
                    'confidence': pred['confidence'],
                    'prediction': pred['prediction'],
                    'model_version': predictor.model_metadata.get('version', 'unknown'),
                    'indicators': pred['indicators']
                }
                prediction_records.append(record)
            
            # Write to database
            count = supabase.write_predictions(prediction_records)
            logger.info(f"✓ Saved {count} predictions to database")
        
        # ===== SUMMARY =====
        logger.info("\n" + "="*80)
        logger.info("SCREENING COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Screened: {len(candidates)} stocks")
        logger.info(f"✓ Predictions made: {len(predictions)}")
        logger.info(f"✓ Above {args.min_probability:.0%} threshold: {len(predictions)}")
        
        confidence_breakdown = {}
        for pred in predictions:
            conf = pred['confidence']
            confidence_breakdown[conf] = confidence_breakdown.get(conf, 0) + 1
        
        logger.info("\nConfidence Breakdown:")
        for conf in ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW']:
            count = confidence_breakdown.get(conf, 0)
            if count > 0:
                logger.info(f"  {conf:12s}: {count}")
        
        logger.info(f"\nFinished at: {datetime.now()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Screening failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
