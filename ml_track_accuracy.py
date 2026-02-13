#!/usr/bin/env python3
"""
Track prediction accuracy - COMPREHENSIVE VERSION
Compares predictions vs actual winners AND tracks self-discovered stocks
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Track ML prediction accuracy")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", type=str, help="Date to check (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--track-self-discovered", action="store_true",
                       help="Also track stocks model found on its own")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, config.get("logging", {}))
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE PREDICTION ACCURACY TRACKING")
    logger.info("="*60)
    
    supabase = MLPredictionSupabaseClient(config)
    client = supabase.client
    
    # Get date to check (yesterday's predictions)
    if args.date:
        check_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        check_date = (datetime.now() - timedelta(days=1)).date()
    
    logger.info(f"Checking predictions for: {check_date}")
    
    # ===== PART 1: Get predictions for that date =====
    logger.info("\nFetching predictions...")
    predictions_response = client.table("ml_explosion_predictions")\
        .select("*")\
        .eq("prediction_date", check_date.isoformat())\
        .execute()
    
    if not predictions_response.data:
        logger.warning(f"No predictions found for {check_date}")
        return 1
    
    predictions = predictions_response.data
    logger.info(f"Found {len(predictions)} predictions to check")
    
    # ===== PART 2: Get actual winners for that date =====
    logger.info("Fetching actual winners...")
    winners_response = client.table("daily_winners")\
        .select("symbol,change_pct,price,volume")\
        .eq("detection_date", check_date.isoformat())\
        .execute()
    
    if not winners_response.data:
        logger.warning(f"No winners data for {check_date}")
        # Continue anyway - we still want to track false positives
    
    winners = {row['symbol']: row for row in (winners_response.data or [])}
    logger.info(f"Found {len(winners)} actual winners")
    
    # ===== PART 3: Track accuracy for predictions =====
    logger.info("\nAnalyzing prediction accuracy...")
    
    accuracy_records = []
    correct_predictions = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    gain_errors = []
    
    for pred in predictions:
        symbol = pred['symbol']
        
        # Check if stock became a winner
        became_winner = symbol in winners
        
        if became_winner:
            actual_gain = winners[symbol]['change_pct']
            actual_price = winners[symbol]['price']
        else:
            actual_gain = 0  # Didn't explode
            actual_price = pred.get('current_price', 0)
        
        # Was prediction correct?
        predicted_positive = pred['prediction'] == 1
        
        if predicted_positive and became_winner:
            true_positives += 1
            correct_predictions += 1
        elif predicted_positive and not became_winner:
            false_positives += 1
        elif not predicted_positive and became_winner:
            false_negatives += 1
        elif not predicted_positive and not became_winner:
            true_negatives += 1
            correct_predictions += 1
        
        prediction_correct = (predicted_positive and became_winner) or \
                           (not predicted_positive and not became_winner)
        
        # Calculate gain error
        predicted_gain = pred.get('target_gain_pct', 0)
        if became_winner and predicted_gain > 0:
            gain_error = abs(predicted_gain - actual_gain)
            gain_errors.append({
                'symbol': symbol,
                'predicted': predicted_gain,
                'actual': actual_gain,
                'error': gain_error,
                'error_pct': (gain_error / actual_gain * 100) if actual_gain != 0 else 0
            })
        else:
            gain_error = None
        
        # Create accuracy record
        accuracy_record = {
            'symbol': symbol,
            'prediction_date': check_date.isoformat(),
            'predicted_probability': pred['explosion_probability'],
            'predicted_signal': pred['signal'],
            'predicted_target_gain': pred.get('target_gain_pct'),
            'predicted_target_price': pred.get('target_price'),
            'became_winner': became_winner,
            'actual_gain_pct': actual_gain if became_winner else None,
            'actual_price': actual_price,
            'prediction_correct': prediction_correct,
            'gain_error_pct': gain_error,
            'actual_recorded_at': datetime.now().isoformat()
        }
        
        accuracy_records.append(accuracy_record)
    
    # ===== PART 4: Check for missed opportunities (stocks in winners but not predicted) =====
    logger.info("\nChecking for missed opportunities...")
    
    predicted_symbols = {p['symbol'] for p in predictions}
    winner_symbols = set(winners.keys())
    missed_symbols = winner_symbols - predicted_symbols
    
    if missed_symbols:
        logger.warning(f"Model missed {len(missed_symbols)} winners:")
        for symbol in list(missed_symbols)[:10]:
            logger.warning(f"  - {symbol}: +{winners[symbol]['change_pct']:.2f}%")
        
        # These are false negatives that weren't even predicted
        false_negatives += len(missed_symbols)
    
    # ===== PART 5: Write accuracy records =====
    if accuracy_records:
        logger.info(f"\nWriting {len(accuracy_records)} accuracy records...")
        try:
            supabase.write_accuracy_records(accuracy_records)
        except Exception as e:
            logger.error(f"Failed to write accuracy records: {e}")
    
    # ===== PART 6: Calculate and display detailed metrics =====
    total_predictions = len(predictions)
    total_winners = len(winners)
    
    logger.info("\n" + "="*60)
    logger.info("ACCURACY ANALYSIS")
    logger.info("="*60)
    
    # Overall accuracy
    accuracy_pct = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Total Predictions: {total_predictions}")
    logger.info(f"  Actual Winners: {total_winners}")
    logger.info(f"  Accuracy: {accuracy_pct:.2f}%")
    
    # Confusion matrix
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Positives:  {true_positives:4d} (Correctly predicted winners)")
    logger.info(f"  False Positives: {false_positives:4d} (Predicted winner, didn't explode)")
    logger.info(f"  True Negatives:  {true_negatives:4d} (Correctly predicted no explosion)")
    logger.info(f"  False Negatives: {false_negatives:4d} (Missed winners)")
    
    # Precision, Recall, F1
    precision = (true_positives / (true_positives + false_positives)) * 100 \
                if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) * 100 \
             if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"\nDetailed Metrics:")
    logger.info(f"  Precision: {precision:.2f}% (Of predicted winners, how many actually exploded)")
    logger.info(f"  Recall: {recall:.2f}% (Of actual winners, how many did we predict)")
    logger.info(f"  F1 Score: {f1:.2f}%")
    
    # Breakdown by signal
    logger.info(f"\nAccuracy by Signal:")
    for signal in ['STRONG BUY', 'BUY', 'HOLD', 'AVOID']:
        signal_records = [r for r in accuracy_records if r['predicted_signal'] == signal]
        if signal_records:
            signal_correct = sum(1 for r in signal_records if r['prediction_correct'])
            signal_total = len(signal_records)
            signal_acc = (signal_correct / signal_total) * 100
            
            signal_winners = sum(1 for r in signal_records if r['became_winner'])
            signal_precision = (signal_winners / signal_total) * 100
            
            logger.info(f"  {signal:12s}: {signal_acc:5.1f}% accuracy, "
                       f"{signal_precision:5.1f}% precision ({signal_correct}/{signal_total} correct, "
                       f"{signal_winners} became winners)")
    
    # Gain prediction accuracy
    if gain_errors:
        logger.info(f"\nGain Prediction Analysis ({len(gain_errors)} winners):")
        avg_error = sum(e['error'] for e in gain_errors) / len(gain_errors)
        avg_error_pct = sum(e['error_pct'] for e in gain_errors) / len(gain_errors)
        
        logger.info(f"  Average Gain Error: {avg_error:.2f} percentage points")
        logger.info(f"  Average Error %: {avg_error_pct:.1f}%")
        
        # Show worst predictions
        worst_errors = sorted(gain_errors, key=lambda x: x['error'], reverse=True)[:5]
        logger.info(f"\n  Top 5 Worst Gain Predictions:")
        for err in worst_errors:
            logger.info(f"    {err['symbol']}: Predicted {err['predicted']:.1f}%, "
                       f"Actual {err['actual']:.1f}% (Error: {err['error']:.1f}pp)")
    
    # ===== PART 7: Track self-discovered stocks (model's own findings) =====
    if args.track_self_discovered:
        logger.info("\n" + "="*60)
        logger.info("TRACKING SELF-DISCOVERED STOCKS")
        logger.info("="*60)
        
        # Get high-confidence predictions that WEREN'T in daily winners
        self_discovered = []
        
        for pred in predictions:
            # High confidence predictions (BUY or STRONG BUY)
            if pred['signal'] in ['BUY', 'STRONG BUY']:
                symbol = pred['symbol']
                
                # Check if it became a winner
                became_winner = symbol in winners
                
                if became_winner:
                    actual_gain = winners[symbol]['change_pct']
                    actual_price = winners[symbol]['price']
                else:
                    # Need to fetch actual price change even if not in winners
                    # This requires additional data - for now just mark as 0
                    actual_gain = 0
                    actual_price = pred.get('current_price', 0)
                
                self_discovered.append({
                    'symbol': symbol,
                    'prediction_date': check_date.isoformat(),
                    'predicted_probability': pred['explosion_probability'],
                    'predicted_signal': pred['signal'],
                    'predicted_target_gain': pred.get('target_gain_pct'),
                    'became_winner': became_winner,
                    'actual_gain_pct': actual_gain if became_winner else None,
                    'discovered_at': datetime.now().isoformat()
                })
        
        if self_discovered:
            logger.info(f"Found {len(self_discovered)} self-discovered stocks")
            
            # Write to database
            try:
                supabase.write_self_discovered_stocks(self_discovered)
                logger.info(f"✓ Wrote {len(self_discovered)} self-discovered records")
            except Exception as e:
                logger.error(f"Failed to write self-discovered stocks: {e}")
            
            # Show summary
            self_winners = sum(1 for s in self_discovered if s['became_winner'])
            self_accuracy = (self_winners / len(self_discovered)) * 100 if self_discovered else 0
            
            logger.info(f"\nSelf-Discovery Performance:")
            logger.info(f"  High-confidence predictions: {len(self_discovered)}")
            logger.info(f"  Actually exploded: {self_winners}")
            logger.info(f"  Success rate: {self_accuracy:.1f}%")
    
    logger.info("\n" + "="*60)
    logger.info("✓ ACCURACY TRACKING COMPLETE")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
