#!/usr/bin/env python3
"""
Comprehensive ML Accuracy Tracker - FIXED WEEKEND HANDLING
Tracks BOTH prediction accuracy AND missed opportunities
Analyzes failures in detail for continuous learning
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from tradingview_ta import TA_Handler, Interval

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


def get_last_trading_day(from_date: datetime = None) -> str:
    """
    Get last trading day (skip weekends)
    
    Args:
        from_date: Start date (defaults to today)
    
    Returns:
        Last trading day as ISO string (YYYY-MM-DD)
    """
    if from_date is None:
        from_date = datetime.now().date()
    elif isinstance(from_date, datetime):
        from_date = from_date.date()
    
    # Start with yesterday
    check_date = from_date - timedelta(days=1)
    
    # Skip backwards until we find a weekday
    while check_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        check_date = check_date - timedelta(days=1)
    
    return check_date.isoformat()


class ComprehensiveAccuracyTracker:
    """
    Tracks comprehensive accuracy:
    1. Did our predictions come true? (Precision)
    2. Did we catch actual winners? (Recall)
    3. Why did we fail? (Failure analysis)
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.supabase = MLPredictionSupabaseClient(config)
        self.client = self.supabase.client
    
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
        """Get all actual winners (20%+ gainers) for a specific date"""
        
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
    
    def fetch_current_indicators(self, symbol: str, exchange: str = 'NASDAQ') -> dict:
        """Fetch current indicators for a stock (for comparison)"""
        
        try:
            handler = TA_Handler(
                symbol=symbol,
                exchange=exchange,
                screener="america",
                interval=Interval.INTERVAL_1_DAY,
                timeout=10
            )
            
            analysis = handler.get_analysis()
            return analysis.indicators
        except Exception as e:
            self.logger.debug(f"Failed to fetch indicators for {symbol}: {e}")
            return {}
    
    def analyze_prediction_accuracy(
        self, 
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame
    ) -> tuple:
        """
        Analyze prediction accuracy
        
        Returns:
            (accuracy_records, details_records)
        """
        
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
            
            # Get actual outcome data
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
            
            # Calculate accuracy
            prediction_correct = (predicted_positive and became_winner) or \
                               (not predicted_positive and not became_winner)
            
            # Calculate gain error
            predicted_gain = pred.get('target_gain_pct', 0)
            if became_winner and predicted_gain > 0:
                gain_error = abs(predicted_gain - actual_gain)
                gain_error_ratio = gain_error / actual_gain if actual_gain != 0 else 0
            else:
                gain_error = None
                gain_error_ratio = None
            
            # Classify outcome
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
            
            # Accuracy record
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
            
            # Details record
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
                'failure_reason': None  # Will be filled in failure analysis
            }
            
            details_records.append(details_record)
        
        # Summary
        total = len(predictions_df)
        correct = true_positives + true_negatives
        accuracy_pct = (correct / total * 100) if total > 0 else 0
        
        self.logger.info(f"\nPrediction Accuracy Results:")
        self.logger.info(f"  Total Predictions: {total}")
        self.logger.info(f"  True Positives: {true_positives}")
        self.logger.info(f"  False Positives: {false_positives}")
        self.logger.info(f"  True Negatives: {true_negatives}")
        self.logger.info(f"  Overall Accuracy: {accuracy_pct:.2f}%")
        
        # Precision & Recall
        predicted_winners = true_positives + false_positives
        if predicted_winners > 0:
            precision = (true_positives / predicted_winners) * 100
            self.logger.info(f"  Precision: {precision:.2f}% (of predictions, how many were correct)")
        
        return accuracy_records, details_records
    
    def analyze_missed_opportunities(
        self,
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame,
        check_date: str
    ) -> list:
        """
        Analyze winners we missed (false negatives)
        """
        
        self.logger.info("\n" + "="*80)
        self.logger.info("ANALYZING MISSED OPPORTUNITIES")
        self.logger.info("="*80)
        
        predicted_symbols = set(predictions_df['symbol'].tolist())
        winner_symbols = set(winners_df['symbol'].tolist())
        
        missed_symbols = winner_symbols - predicted_symbols
        
        self.logger.info(f"\nMissed {len(missed_symbols)} winners:")
        
        missed_records = []
        
        for symbol in missed_symbols:
            winner_data = winners_df[winners_df['symbol'] == symbol].iloc[0]
            
            # Check if it was even in our predictions (with low probability)
            was_predicted = symbol in predicted_symbols
            
            if was_predicted:
                pred_data = predictions_df[predictions_df['symbol'] == symbol].iloc[0]
                predicted_probability = pred_data['explosion_probability']
                predicted_signal = pred_data['signal']
                screening_failure_reason = None
                was_screened = True
            else:
                predicted_probability = None
                predicted_signal = None
                was_screened = False
                # Try to determine why it wasn't screened
                screening_failure_reason = self._determine_screening_failure(winner_data)
            
            missed_record = {
                'symbol': symbol,
                'detection_date': check_date,
                'exchange': winner_data.get('exchange', 'UNKNOWN'),
                'actual_gain_pct': winner_data['change_pct'],
                'actual_high_pct': ((winner_data.get('high', winner_data['price']) / winner_data['price']) - 1) * 100,
                'actual_price': winner_data['price'],
                'actual_volume': int(winner_data['volume']),
                'was_screened': was_screened,
                'screening_failure_reason': screening_failure_reason,
                'predicted_probability': predicted_probability,
                'predicted_signal': predicted_signal,
                # Would need to fetch T-1 indicators here for full analysis
                'rsi_at_t1': None,
                'macd_at_t1': None,
                'adx_at_t1': None,
                'volume_ratio_at_t1': None,
                'hv_20_at_t1': None,
                'pattern_analysis': None
            }
            
            missed_records.append(missed_record)
            
            self.logger.info(f"  - {symbol}: +{winner_data['change_pct']:.2f}% "
                           f"({'NOT screened - ' + screening_failure_reason if not was_screened else 'Screened but low probability'})")
        
        # Summary by failure reason
        if missed_records:
            failure_reasons = {}
            for rec in missed_records:
                reason = rec['screening_failure_reason'] or 'screened_but_not_predicted'
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            self.logger.info(f"\nMissed Opportunities by Reason:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {reason}: {count}")
        
        return missed_records
    
    def _determine_screening_failure(self, winner_data: pd.Series) -> str:
        """Determine why a winner wasn't screened"""
        
        price = winner_data.get('price', 0)
        volume = winner_data.get('volume', 0)
        
        # Check against typical screening filters
        if price < 3.0:
            return 'price_too_low'
        elif price > 500.0:
            return 'price_too_high'
        elif volume < 500000:
            return 'volume_too_low'
        else:
            return 'unknown'
    
    def analyze_false_positives(
        self,
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame,
        check_date: str
    ) -> list:
        """
        Deep analysis of false positives - why did predictions fail?
        """
        
        self.logger.info("\n" + "="*80)
        self.logger.info("ANALYZING FALSE POSITIVES")
        self.logger.info("="*80)
        
        winners_set = set(winners_df['symbol'].tolist())
        
        # Get false positives (predicted positive but didn't win)
        false_positives = predictions_df[
            (predictions_df['prediction'] == 1) &
            (~predictions_df['symbol'].isin(winners_set))
        ]
        
        self.logger.info(f"\nAnalyzing {len(false_positives)} false positives...")
        
        fp_records = []
        
        for _, pred in false_positives.iterrows():
            symbol = pred['symbol']
            
            # Fetch current day's actual performance
            indicators = self.fetch_current_indicators(symbol, pred.get('exchange', 'NASDAQ'))
            
            # Determine failure category
            failure_category = self._classify_failure(pred, indicators)
            
            fp_record = {
                'symbol': symbol,
                'prediction_date': check_date,
                'predicted_probability': pred['explosion_probability'],
                'predicted_signal': pred['signal'],
                'predicted_target_gain': pred.get('target_gain_pct'),
                'actual_gain_pct': indicators.get('change', 0),
                'actual_high_pct': None,  # Would need intraday data
                'actual_low_pct': None,
                'failure_category': failure_category,
                'volume_dropped': self._check_volume_drop(pred, indicators),
                'momentum_faded': self._check_momentum_fade(pred, indicators),
                'resistance_hit': None,  # Would need technical analysis
                'market_direction': None,  # Would need SPY/QQQ data
                'sector_performance': None,
                'lessons_learned': self._generate_lesson(failure_category)
            }
            
            fp_records.append(fp_record)
        
        # Summary by category
        if fp_records:
            categories = {}
            for rec in fp_records:
                cat = rec['failure_category']
                categories[cat] = categories.get(cat, 0) + 1
            
            self.logger.info(f"\nFalse Positives by Category:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {cat}: {count}")
        
        return fp_records
    
    def _classify_failure(self, prediction: pd.Series, indicators: dict) -> str:
        """Classify why a prediction failed"""
        
        actual_change = indicators.get('change', 0)
        volume = indicators.get('volume', 0)
        predicted_volume = prediction.get('volume_ratio', 1.0)
        
        # Classification logic
        if actual_change > 10:
            return 'early_peak'  # Did spike but not enough
        elif volume < predicted_volume * 0.5:
            return 'volume_insufficient'
        elif actual_change < -5:
            return 'market_reversal'
        elif 0 < actual_change < 5:
            return 'weak_followthrough'
        else:
            return 'other'
    
    def _check_volume_drop(self, prediction: pd.Series, indicators: dict) -> bool:
        """Check if volume failed to materialize"""
        predicted_volume_ratio = prediction.get('volume_ratio', 1.0)
        actual_volume = indicators.get('volume', 0)
        avg_volume = indicators.get('volume|20', 1)
        
        actual_ratio = actual_volume / avg_volume if avg_volume > 0 else 0
        
        return actual_ratio < predicted_volume_ratio * 0.7
    
    def _check_momentum_fade(self, prediction: pd.Series, indicators: dict) -> bool:
        """Check if momentum indicators weakened"""
        predicted_rsi = prediction.get('rsi', 50)
        actual_rsi = indicators.get('RSI', 50)
        
        # Simple check: RSI dropped significantly
        return actual_rsi < predicted_rsi - 10
    
    def _generate_lesson(self, failure_category: str) -> str:
        """Generate learning lesson from failure"""
        
        lessons = {
            'volume_insufficient': 'Add stronger volume confirmation filters',
            'weak_followthrough': 'Improve momentum strength indicators',
            'market_reversal': 'Add market trend filter (SPY/QQQ)',
            'early_peak': 'Consider intraday patterns, not just close',
            'other': 'Requires further investigation'
        }
        
        return lessons.get(failure_category, 'Unknown failure pattern')
    
    def write_all_records(
        self,
        accuracy_records: list,
        details_records: list,
        missed_records: list,
        fp_records: list
    ):
        """Write all analysis records to database"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("WRITING RECORDS TO DATABASE")
        self.logger.info("="*80)
        
        # Write accuracy records
        if accuracy_records:
            self.logger.info(f"Writing {len(accuracy_records)} accuracy records...")
            self.supabase.write_accuracy_records(accuracy_records)
        
        # Write details records
        if details_records:
            self.logger.info(f"Writing {len(details_records)} detail records...")
            try:
                self.client.table("ml_accuracy_details").upsert(
                    details_records,
                    on_conflict="symbol,prediction_date"
                ).execute()
            except Exception as e:
                self.logger.error(f"Failed to write details: {e}")
        
        # Write missed opportunities
        if missed_records:
            self.logger.info(f"Writing {len(missed_records)} missed opportunity records...")
            try:
                self.client.table("ml_missed_opportunities").upsert(
                    missed_records,
                    on_conflict="symbol,detection_date"
                ).execute()
            except Exception as e:
                self.logger.error(f"Failed to write missed opportunities: {e}")
        
        # Write false positives
        if fp_records:
            self.logger.info(f"Writing {len(fp_records)} false positive records...")
            try:
                self.client.table("ml_false_positives_analysis").upsert(
                    fp_records,
                    on_conflict="symbol,prediction_date"
                ).execute()
            except Exception as e:
                self.logger.error(f"Failed to write false positives: {e}")
        
        # Refresh materialized views
        self.logger.info("Refreshing materialized views...")
        try:
            self.client.rpc('refresh_ml_views').execute()
        except Exception as e:
            self.logger.warning(f"Failed to refresh views: {e}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive ML accuracy tracking")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", type=str, help="Date to check (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, config.get("logging", {}))
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE ML ACCURACY TRACKING")
    logger.info("="*80)
    
    # Get date to check - FIXED: Handle weekends properly
    if args.date:
        check_date = args.date
        logger.info(f"Using manually specified date: {check_date}")
    else:
        check_date = get_last_trading_day()
        logger.info(f"Auto-detected last trading day: {check_date}")
        
        # Show what day of week it is
        date_obj = datetime.fromisoformat(check_date)
        day_name = date_obj.strftime("%A")
        logger.info(f"  ({day_name})")
    
    # Initialize tracker
    tracker = ComprehensiveAccuracyTracker(config)
    
    # Get predictions and actual winners
    predictions_df = tracker.get_predictions_for_date(check_date)
    winners_df = tracker.get_actual_winners_for_date(check_date)
    
    if predictions_df.empty:
        logger.warning(f"No predictions found for {check_date}")
        logger.info("Make sure ml_screen_and_predict.py ran successfully for this date")
        return 1
    
    if winners_df.empty:
        logger.warning(f"No winners found for {check_date}")
        logger.info("This could mean:")
        logger.info("  1. No stocks gained 20%+ on this date")
        logger.info("  2. Daily winners workflow hasn't run yet")
        logger.info("  3. This was a weekend/holiday (no trading)")
        return 1
    
    # Run comprehensive analysis
    logger.info("\n" + "="*80)
    logger.info("STARTING COMPREHENSIVE ANALYSIS")
    logger.info("="*80)
    
    # 1. Prediction accuracy
    accuracy_records, details_records = tracker.analyze_prediction_accuracy(
        predictions_df, 
        winners_df
    )
    
    # 2. Missed opportunities (recall)
    missed_records = tracker.analyze_missed_opportunities(
        predictions_df,
        winners_df,
        check_date
    )
    
    # 3. False positive analysis
    fp_records = tracker.analyze_false_positives(
        predictions_df,
        winners_df,
        check_date
    )
    
    # 4. Write everything to database
    tracker.write_all_records(
        accuracy_records,
        details_records,
        missed_records,
        fp_records
    )
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✓ COMPREHENSIVE ACCURACY TRACKING COMPLETE")
    logger.info("="*80)
    
    logger.info(f"\nRecords Written:")
    logger.info(f"  Accuracy: {len(accuracy_records)}")
    logger.info(f"  Details: {len(details_records)}")
    logger.info(f"  Missed: {len(missed_records)}")
    logger.info(f"  False Positives: {len(fp_records)}")
    
    logger.info(f"\nDatabase Tables Updated:")
    logger.info(f"  - ml_prediction_accuracy")
    logger.info(f"  - ml_accuracy_details")
    logger.info(f"  - ml_missed_opportunities")
    logger.info(f"  - ml_false_positives_analysis")
    
    logger.info(f"\nViews refreshed - ready for dashboard")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
