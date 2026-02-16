#!/usr/bin/env python3
"""
Comprehensive ML Accuracy Tracker with LEARNING - IMPROVED VERSION

IMPROVEMENTS:
1. ✅ Finds most recent prediction date automatically (safer than assuming yesterday)
2. ✅ Validates predictions exist before fetching winners (saves egress)
3. ✅ Better error handling to prevent wasted API calls
4. ✅ Early exit if no data to process (saves resources)

This script:
1. Compares predictions vs actual winners
2. Analyzes missed opportunities
3. Learns from mistakes
4. Updates learned filters for future screening
5. Stores insights for model retraining
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import yaml
import json
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


def load_config(config_path: str) -> dict:
    """Load config from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    """Setup basic logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_most_recent_prediction_date(tracker) -> Optional[str]:
    """
    Find the most recent date that has predictions in the database
    This is MUCH safer than assuming yesterday had predictions
    
    Returns:
        Most recent prediction date as ISO string, or None if no predictions found
    """
    try:
        # Single lightweight query - just get the most recent date
        response = tracker.client.table("ml_explosion_predictions")\
            .select("prediction_date")\
            .order("prediction_date", desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data:
            return None
        
        return response.data[0]['prediction_date']
        
    except Exception as e:
        tracker.logger.error(f"Error finding most recent prediction date: {e}")
        return None


def validate_data_exists(tracker, check_date: str) -> dict:
    """
    Validate that both predictions AND winners exist for the date
    Returns early if data is missing to save egress
    
    Returns:
        Dict with 'predictions_exist', 'winners_exist', 'should_proceed', 'prediction_count', 'winner_count'
    """
    result = {
        'predictions_exist': False,
        'winners_exist': False,
        'should_proceed': False,
        'prediction_count': 0,
        'winner_count': 0
    }
    
    try:
        # Check predictions (lightweight count query)
        pred_response = tracker.client.table("ml_explosion_predictions")\
            .select("*", count="exact")\
            .eq("prediction_date", check_date)\
            .limit(1)\
            .execute()
        
        result['prediction_count'] = pred_response.count if pred_response.count else 0
        result['predictions_exist'] = result['prediction_count'] > 0
        
        if not result['predictions_exist']:
            tracker.logger.warning(f"⚠️ No predictions found for {check_date}")
            tracker.logger.info("Make sure ml_screen_and_predict.yml ran successfully for this date")
            return result
        
        tracker.logger.info(f"✓ Found {result['prediction_count']} predictions for {check_date}")
        
        # Check winners (lightweight count query)
        winner_response = tracker.client.table("daily_winners")\
            .select("*", count="exact")\
            .eq("detection_date", check_date)\
            .limit(1)\
            .execute()
        
        result['winner_count'] = winner_response.count if winner_response.count else 0
        result['winners_exist'] = result['winner_count'] > 0
        
        if not result['winners_exist']:
            tracker.logger.warning(f"⚠️ No winners found for {check_date}")
            tracker.logger.info("Make sure daily_top10.yml ran successfully for this date")
            return result
        
        tracker.logger.info(f"✓ Found {result['winner_count']} winners for {check_date}")
        
        # Both exist - safe to proceed
        result['should_proceed'] = True
        return result
        
    except Exception as e:
        tracker.logger.error(f"Error validating data: {e}")
        return result


class ComprehensiveAccuracyTracker:
    """
    Tracks accuracy AND learns from mistakes
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.supabase = MLPredictionSupabaseClient(config)
        self.client = self.supabase.client
        
        # Learning storage
        self.learned_insights = {
            'screening_patterns': [],
            'prediction_patterns': [],
            'missed_winner_patterns': [],
            'false_positive_patterns': []
        }
    
    def get_predictions_for_date(self, check_date: str) -> pd.DataFrame:
        """Get all predictions made for a specific date"""
        
        self.logger.info(f"Fetching predictions for {check_date}...")
        
        response = self.client.table("ml_explosion_predictions")\
            .select("*")\
            .eq("prediction_date", check_date)\
            .execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        self.logger.info(f"Found {len(df)} predictions")
        return df
    
    def get_actual_winners_for_date(self, check_date: str) -> pd.DataFrame:
        """Get all actual winners for a specific date"""
        
        self.logger.info(f"Fetching actual winners for {check_date}...")
        
        response = self.client.table("daily_winners")\
            .select("symbol,change_pct,price,volume,high,low,open,close")\
            .eq("detection_date", check_date)\
            .execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        self.logger.info(f"Found {len(df)} actual winners")
        return df
    
    def get_actual_non_winners_for_date(self, check_date: str) -> pd.DataFrame:
        """Get non-winners (negative examples)"""
        
        self.logger.info(f"Fetching non-winners for {check_date}...")
        
        response = self.client.table("daily_non_winners")\
            .select("symbol,change_pct,price,volume")\
            .eq("detection_date", check_date)\
            .execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        self.logger.info(f"Found {len(df)} non-winners")
        return df
    
    def analyze_prediction_accuracy(
        self, 
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame
    ) -> tuple:
        """Analyze prediction accuracy"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("ANALYZING PREDICTION ACCURACY")
        self.logger.info("="*80)
        
        winners_set = set(winners_df['symbol'].tolist())
        
        accuracy_records = []
        details_records = []
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        
        for _, pred in predictions_df.iterrows():
            symbol = pred['symbol']
            predicted_positive = pred['prediction'] == 1
            became_winner = symbol in winners_set
            
            if became_winner:
                winner_data = winners_df[winners_df['symbol'] == symbol].iloc[0]
                actual_gain = winner_data['change_pct']
                actual_price = winner_data['price']
                actual_high = winner_data.get('high', actual_price)
                actual_high_pct = ((actual_high / actual_price) - 1) * 100 if actual_price > 0 else 0
            else:
                actual_gain = 0
                actual_price = pred.get('current_price', 0)
                actual_high_pct = 0
            
            prediction_correct = (predicted_positive and became_winner) or \
                               (not predicted_positive and not became_winner)
            
            predicted_gain = pred.get('target_gain_pct', 0)
            if became_winner and predicted_gain > 0:
                gain_error = abs(predicted_gain - actual_gain)
                gain_error_ratio = gain_error / actual_gain if actual_gain != 0 else 0
            else:
                gain_error = None
                gain_error_ratio = None
            
            if predicted_positive and became_winner:
                outcome_type = 'true_positive'
                true_positives += 1
            elif predicted_positive and not became_winner:
                outcome_type = 'false_positive'
                false_positives += 1
            elif not predicted_positive and not became_winner:
                outcome_type = 'true_negative'
                true_negatives += 1
            else:
                outcome_type = 'false_negative'
            
            accuracy_record = {
                'symbol': symbol,
                'prediction_date': pred['prediction_date'],
                'predicted_probability': pred['explosion_probability'],
                'predicted_signal': pred['signal'],
                'predicted_target_gain': pred.get('target_gain_pct'),
                'predicted_target_price': pred.get('target_price'),
                'became_winner': became_winner,
                'actual_gain_pct': actual_gain if became_winner else None,
                'actual_high_pct': actual_high_pct if became_winner else None,
                'actual_price': actual_price,
                'prediction_correct': prediction_correct,
                'gain_error_pct': gain_error,
                'gain_error_ratio': gain_error_ratio,
                'actual_recorded_at': datetime.now().isoformat()
            }
            
            accuracy_records.append(accuracy_record)
            
            details_record = {
                'symbol': symbol,
                'prediction_date': pred['prediction_date'],
                'predicted_probability': pred['explosion_probability'],
                'predicted_signal': pred['signal'],
                'outcome_type': outcome_type,
                'became_winner': became_winner,
                'actual_gain_pct': actual_gain if became_winner else None,
                'actual_high_pct': actual_high_pct if became_winner else None,
                'actual_volume': int(winner_data['volume']) if became_winner else None,
                'predicted_rsi': pred.get('rsi'),
                'predicted_macd': pred.get('macd'),
                'predicted_volume_ratio': pred.get('volume_ratio'),
                'failure_reason': None
            }
            
            details_records.append(details_record)
        
        total = len(predictions_df)
        correct = true_positives + true_negatives
        accuracy_pct = (correct / total * 100) if total > 0 else 0
        
        self.logger.info(f"\nPrediction Accuracy:")
        self.logger.info(f"  Total: {total}")
        self.logger.info(f"  True Positives: {true_positives}")
        self.logger.info(f"  False Positives: {false_positives}")
        self.logger.info(f"  True Negatives: {true_negatives}")
        self.logger.info(f"  Accuracy: {accuracy_pct:.2f}%")
        
        predicted_winners = true_positives + false_positives
        if predicted_winners > 0:
            precision = (true_positives / predicted_winners) * 100
            self.logger.info(f"  Precision: {precision:.2f}%")
        
        return accuracy_records, details_records
    
    def analyze_missed_opportunities(
        self,
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame,
        check_date: str
    ) -> list:
        """Analyze winners we missed - LEARN FROM THIS"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("ANALYZING MISSED OPPORTUNITIES (LEARNING)")
        self.logger.info("="*80)
        
        predicted_symbols = set(predictions_df['symbol'].tolist())
        winner_symbols = set(winners_df['symbol'].tolist())
        
        missed_symbols = winner_symbols - predicted_symbols
        
        self.logger.info(f"\nMissed {len(missed_symbols)} winners")
        
        missed_records = []
        
        for symbol in missed_symbols:
            winner_data = winners_df[winners_df['symbol'] == symbol].iloc[0]
            
            missed_record = {
                'symbol': symbol,
                'detection_date': check_date,
                'exchange': winner_data.get('exchange', 'UNKNOWN'),
                'actual_gain_pct': winner_data['change_pct'],
                'actual_high_pct': ((winner_data.get('high', winner_data['price']) / winner_data['price']) - 1) * 100,
                'actual_price': winner_data['price'],
                'actual_volume': int(winner_data['volume']),
                'was_screened': False,
                'screening_failure_reason': self._determine_screening_failure(winner_data),
                'predicted_probability': None,
                'predicted_signal': None,
            }
            
            missed_records.append(missed_record)
            
            # LEARN: Store pattern for future screening improvements
            self.learned_insights['missed_winner_patterns'].append({
                'price': winner_data['price'],
                'volume': winner_data['volume'],
                'gain': winner_data['change_pct'],
                'reason': missed_record['screening_failure_reason']
            })
        
        return missed_records
    
    def _determine_screening_failure(self, winner_data: pd.Series) -> str:
        """Determine why a winner wasn't screened"""
        
        price = winner_data.get('price', 0)
        volume = winner_data.get('volume', 0)
        
        if price < 0.50:
            return 'price_too_low'
        elif price > 500.0:
            return 'price_too_high'
        elif volume < 100000:
            return 'volume_too_low'
        else:
            return 'not_in_screener_results'
    
    def learn_from_mistakes(
        self,
        missed_records: list,
        false_positive_records: list
    ) -> dict:
        """
        CRITICAL: Learn from mistakes and update screening filters
        """
        
        self.logger.info("\n" + "="*80)
        self.logger.info("LEARNING FROM MISTAKES")
        self.logger.info("="*80)
        
        learned_filters = {}
        
        if missed_records:
            # Analyze missed winners to adjust filters
            missed_df = pd.DataFrame(missed_records)
            
            # Price range analysis
            missed_prices = missed_df[missed_df['screening_failure_reason'] != 'not_in_screener_results']['actual_price']
            
            if len(missed_prices) > 0:
                min_missed_price = missed_prices.min()
                max_missed_price = missed_prices.max()
                
                self.logger.info(f"\nMissed winner price range: ${min_missed_price:.2f} - ${max_missed_price:.2f}")
                
                # Adjust filters to catch more
                learned_filters['min_price'] = min(0.25, min_missed_price * 0.8)
                learned_filters['max_price'] = max(500.0, max_missed_price * 1.2)
            
            # Volume analysis
            missed_volumes = missed_df[missed_df['screening_failure_reason'] == 'volume_too_low']['actual_volume']
            
            if len(missed_volumes) > 0:
                min_missed_volume = missed_volumes.min()
                
                self.logger.info(f"\nMissed winner min volume: {min_missed_volume:,}")
                
                # Lower volume threshold slightly
                learned_filters['min_volume'] = max(50000, int(min_missed_volume * 0.7))
        
        # Save learned filters
        filter_path = Path('ml_models/learned_filters.json')
        filter_path.parent.mkdir(exist_ok=True)
        
        with open(filter_path, 'w') as f:
            json.dump(learned_filters, f, indent=2)
        
        self.logger.info(f"\n✓ Saved learned filters: {learned_filters}")
        
        return learned_filters
    
    def write_all_records(
        self,
        accuracy_records: list,
        details_records: list,
        missed_records: list
    ):
        """Write all analysis records to database"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("WRITING RECORDS TO DATABASE")
        self.logger.info("="*80)
        
        if accuracy_records:
            self.logger.info(f"Writing {len(accuracy_records)} accuracy records...")
            self.supabase.write_accuracy_records(accuracy_records)
        
        if details_records:
            self.logger.info(f"Writing {len(details_records)} detail records...")
            try:
                self.client.table("ml_accuracy_details").upsert(
                    details_records,
                    on_conflict="symbol,prediction_date"
                ).execute()
            except Exception as e:
                self.logger.error(f"Failed to write details: {e}")
        
        if missed_records:
            self.logger.info(f"Writing {len(missed_records)} missed opportunity records...")
            try:
                self.client.table("ml_missed_opportunities").upsert(
                    missed_records,
                    on_conflict="symbol,detection_date"
                ).execute()
            except Exception as e:
                self.logger.error(f"Failed to write missed opportunities: {e}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive ML accuracy tracking with learning")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", type=str, help="Date to check (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE ML ACCURACY TRACKING WITH LEARNING")
    logger.info("="*80)
    
    tracker = ComprehensiveAccuracyTracker(config)
    
    if args.date:
        check_date = args.date
        logger.info(f"Using manually specified date: {check_date}")
    else:
        # IMPROVED: Find most recent prediction date instead of assuming yesterday
        check_date = get_most_recent_prediction_date(tracker)
        
        if not check_date:
            logger.warning("⚠️ No predictions found in database. Nothing to track.")
            logger.info("Make sure ml_screen_and_predict.yml has run successfully first.")
            return 0  # Exit gracefully - not an error, just nothing to do
        
        logger.info(f"✓ Found most recent prediction date: {check_date}")
        
        date_obj = datetime.fromisoformat(check_date)
        day_name = date_obj.strftime("%A")
        logger.info(f"  ({day_name})")
    
    # IMPROVED: Validate data exists before doing expensive queries
    validation = validate_data_exists(tracker, check_date)
    
    if not validation['should_proceed']:
        logger.warning("="*80)
        logger.warning("DATA VALIDATION FAILED - EXITING EARLY")
        logger.warning("="*80)
        logger.warning(f"  Predictions exist: {validation['predictions_exist']}")
        logger.warning(f"  Winners exist: {validation['winners_exist']}")
        logger.info("\nThis saves egress by not fetching data that doesn't exist.")
        logger.info("Workflows will retry automatically when data is available.")
        return 0  # Exit gracefully
    
    logger.info("")
    logger.info("="*80)
    logger.info("✓ DATA VALIDATION PASSED - PROCEEDING WITH ANALYSIS")
    logger.info("="*80)
    logger.info(f"  Predictions: {validation['prediction_count']}")
    logger.info(f"  Winners: {validation['winner_count']}")
    
    # Get predictions and actual winners
    predictions_df = tracker.get_predictions_for_date(check_date)
    winners_df = tracker.get_actual_winners_for_date(check_date)
    non_winners_df = tracker.get_actual_non_winners_for_date(check_date)
    
    # Run comprehensive analysis
    logger.info("\n" + "="*80)
    logger.info("STARTING COMPREHENSIVE ANALYSIS")
    logger.info("="*80)
    
    # 1. Prediction accuracy
    accuracy_records, details_records = tracker.analyze_prediction_accuracy(
        predictions_df, 
        winners_df
    )
    
    # 2. Missed opportunities
    missed_records = tracker.analyze_missed_opportunities(
        predictions_df,
        winners_df,
        check_date
    )
    
    # 3. LEARN from mistakes
    learned_filters = tracker.learn_from_mistakes(
        missed_records,
        []  # TODO: Add false positive records
    )
    
    # 4. Write everything to database
    tracker.write_all_records(
        accuracy_records,
        details_records,
        missed_records
    )
    
    # FINAL SUMMARY
    logger.info("\n" + "="*80)
    logger.info("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    logger.info("="*80)
    
    logger.info(f"\nRecords Written:")
    logger.info(f"  Accuracy: {len(accuracy_records)}")
    logger.info(f"  Details: {len(details_records)}")
    logger.info(f"  Missed: {len(missed_records)}")
    
    logger.info(f"\nLearning Applied:")
    logger.info(f"  Updated screening filters: {learned_filters}")
    logger.info(f"  These will be used in next prediction run")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
