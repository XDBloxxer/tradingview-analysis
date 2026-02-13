#!/usr/bin/env python3
"""
Daily ML Prediction Runner - WITH TARGET GAINS
Generates explosion predictions with estimated target gains
Optimized for minimal egress and proper feature handling
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config
from src.ml_predictor.explosion_predictor import ExplosionPredictor
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


def main():
    parser = argparse.ArgumentParser(description="Generate ML explosion predictions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--use-all-timepoints", action="store_true", 
                       help="Use all available timepoints (not just day_prior_close)")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, config.get("logging", {}))
    
    logger.info("="*60)
    logger.info("ML EXPLOSION PREDICTION WITH TARGET GAINS")
    logger.info("="*60)
    
    # Initialize
    try:
        predictor = ExplosionPredictor()
        supabase = MLPredictionSupabaseClient(config)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # Determine prediction date
    if args.date:
        prediction_date = args.date
    else:
        prediction_date = datetime.now().date().isoformat()
    
    logger.info(f"Prediction date: {prediction_date}")
    
    # Get T-1 close data (BEST timepoint, minimal leakage)
    logger.info("Fetching T-1 close data...")
    day_prior_data = supabase.get_latest_day_prior_close(args.date)
    
    if day_prior_data.empty:
        logger.warning("No T-1 close data found")
        return 1
    
    logger.info(f"Found {len(day_prior_data)} stocks")
    
    # Prepare features
    logger.info("Preparing features with adaptive mapping...")
    features_df = predictor.prepare_features_from_daily_winners(day_prior_data)
    
    # Get historical gains for calibration
    logger.info("Loading historical gains for calibration...")
    historical_gains = supabase.get_historical_prediction_accuracy(days_back=30)
    
    if historical_gains.empty:
        logger.warning("No historical gains available - using rule-based estimates")
    else:
        logger.info(f"Loaded {len(historical_gains)} historical gain records")
    
    # Make predictions WITH target gains
    logger.info("Running ML model with target gain estimation...")
    predictions_df = predictor.predict_with_targets(features_df, historical_gains)
    
    logger.info(f"Generated {len(predictions_df)} predictions")
    
    # Display summary statistics
    logger.info("\n" + "="*60)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*60)
    
    signal_counts = predictions_df['signal'].value_counts()
    for signal, count in signal_counts.items():
        logger.info(f"  {signal}: {count}")
    
    avg_prob = predictions_df['explosion_probability'].mean()
    max_prob = predictions_df['explosion_probability'].max()
    logger.info(f"\n  Average Probability: {avg_prob*100:.2f}%")
    logger.info(f"  Max Probability: {max_prob*100:.2f}%")
    
    if 'target_gain_pct' in predictions_df.columns:
        avg_target = predictions_df['target_gain_pct'].mean()
        logger.info(f"  Average Target Gain: {avg_target:.2f}%")
    
    # Display top predictions
    logger.info("\n" + "="*80)
    logger.info("TOP 10 PREDICTIONS WITH TARGET GAINS")
    logger.info("="*80)
    
    top_10 = predictions_df.head(10)
    for idx, row in top_10.iterrows():
        current = row.get('current_price', 0)
        target = row.get('target_price', 0)
        target_low = row.get('target_price_low', 0)
        target_high = row.get('target_price_high', 0)
        target_gain = row.get('target_gain_pct', 0)
        
        logger.info(
            f"{idx+1:2d}. {row['symbol']:6s} | {row['signal']:12s} "
            f"({row['explosion_probability']*100:5.2f}%) | "
            f"Current: ${current:6.2f} → Target: ${target:6.2f} "
            f"(+{target_gain:5.1f}%) [${target_low:.2f}-${target_high:.2f}]"
        )
    
    # Prepare for database
    logger.info("\n" + "="*60)
    logger.info("WRITING TO DATABASE")
    logger.info("="*60)
    
    predictions_list = []
    
    for _, row in predictions_df.iterrows():
        # Get original data for this stock
        original_data = day_prior_data[day_prior_data['symbol'] == row['symbol']]
        
        if original_data.empty:
            logger.warning(f"No original data for {row['symbol']}, skipping")
            continue
        
        original_row = original_data.iloc[0]
        
        prediction_record = {
            'symbol': row['symbol'],
            'exchange': original_row.get('exchange', 'UNKNOWN'),
            'prediction_date': prediction_date,
            'explosion_probability': float(row['explosion_probability']),
            'prediction': int(row['prediction']),
            'signal': row['signal'],
            
            # Target gains
            'target_gain_pct': float(row.get('target_gain_pct', 0)),
            'target_gain_low': float(row.get('target_gain_low', 0)),
            'target_gain_high': float(row.get('target_gain_high', 0)),
            'current_price': float(row.get('current_price', 0)),
            'target_price': float(row.get('target_price', 0)),
            'target_price_low': float(row.get('target_price_low', 0)),
            'target_price_high': float(row.get('target_price_high', 0)),
            
            # Key indicators for reference
            'rsi': float(original_row.get('rsi', 0)) if pd.notna(original_row.get('rsi')) else None,
            'macd': float(original_row.get('macd.macd', 0)) if pd.notna(original_row.get('macd.macd')) else None,
            'adx': float(original_row.get('adx', 0)) if pd.notna(original_row.get('adx')) else None,
            'volume_ratio': float(original_row.get('volume_ratio', 0)) if pd.notna(original_row.get('volume_ratio')) else None,
            'hv_20': float(original_row.get('volatility_20d', 0)) if pd.notna(original_row.get('volatility_20d')) else None,
            'bb_width': float(original_row.get('bb_width', 0)) if pd.notna(original_row.get('bb_width')) else None,
        }
        
        predictions_list.append(prediction_record)
    
    # Write to database
    if predictions_list:
        logger.info(f"Writing {len(predictions_list)} predictions to database...")
        try:
            count = supabase.write_predictions(predictions_list)
            logger.info(f"✓ Successfully wrote {count} predictions")
        except Exception as e:
            logger.error(f"Failed to write predictions: {e}")
            return 1
    else:
        logger.warning("No predictions to write")
    
    logger.info("\n" + "="*60)
    logger.info("✓ PREDICTION COMPLETE")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())
