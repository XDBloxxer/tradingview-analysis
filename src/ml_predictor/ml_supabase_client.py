"""
ML Supabase Client - COMPLETE VERSION
Handles ML prediction storage, accuracy tracking, and screening logs
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from supabase import create_client, Client
import os


class MLPredictionSupabaseClient:
    """
    Client for storing and retrieving ML predictions, accuracy tracking, and screening logs
    """
    
    def __init__(self, config: dict):
        """Initialize Supabase client"""
        self.logger = logging.getLogger(__name__)
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and KEY must be provided in environment variables")
        
        self.client: Client = create_client(supabase_url, supabase_key)
        
        # Table names
        self.predictions_table = "ml_explosion_predictions"
        self.accuracy_table = "ml_prediction_accuracy"
        self.screening_log_table = "ml_screening_logs"
        
        self.logger.info("ML Supabase client initialized")
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a value for Supabase/PostgreSQL"""
        if value is None:
            return None
        
        if pd.isna(value):
            return None
        
        if isinstance(value, np.integer):
            return int(value)
        
        if isinstance(value, np.floating):
            if np.isinf(value) or np.isnan(value):
                return None
            return float(value)
        
        if isinstance(value, np.bool_):
            return bool(value)
        
        if isinstance(value, float):
            if np.isinf(value) or np.isnan(value):
                return None
            return value
        
        return value
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all values in a dictionary"""
        return {k: self._sanitize_value(v) for k, v in data.items()}
    
    def write_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """
        Write ML predictions to database
        
        Args:
            predictions: List of prediction dictionaries with keys:
                - symbol
                - exchange
                - prediction_date
                - explosion_probability
                - prediction (0 or 1)
                - signal (STRONG BUY, BUY, HOLD, AVOID)
                - target_gain_pct
                - target_gain_low
                - target_gain_high
                - current_price
                - target_price
                - target_price_low
                - target_price_high
                - rsi, macd, adx, volume_ratio, bb_width (optional indicators)
                
        Returns:
            Number of records written
        """
        
        if not predictions:
            self.logger.warning("No predictions to write")
            return 0
        
        try:
            # Check for existing predictions (prevent duplicates)
            prediction_date = predictions[0].get('prediction_date')
            symbols = [p['symbol'] for p in predictions]
            
            existing = self.client.table(self.predictions_table)\
                .select("symbol")\
                .eq("prediction_date", prediction_date)\
                .in_("symbol", symbols)\
                .execute()
            
            existing_symbols = set(r['symbol'] for r in existing.data) if existing.data else set()
            
            # Filter out existing
            new_predictions = [p for p in predictions if p['symbol'] not in existing_symbols]
            
            if len(existing_symbols) > 0:
                self.logger.info(f"Skipping {len(existing_symbols)} predictions that already exist")
            
            if not new_predictions:
                self.logger.info("No new predictions to write")
                return 0
            
            # Sanitize data
            sanitized = [self._sanitize_dict(pred) for pred in new_predictions]
            
            # Write to database
            response = self.client.table(self.predictions_table).insert(sanitized).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"✓ Wrote {count} predictions to database")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to write predictions: {e}", exc_info=True)
            raise
    
    def write_screening_log(self, log_data: Dict[str, Any]) -> bool:
        """
        Write screening statistics log
        
        Args:
            log_data: Dictionary with screening statistics:
                - screening_date
                - total_symbols_attempted
                - symbols_fetched_successfully
                - symbols_after_price_filter
                - symbols_after_volume_filter
                - symbols_after_all_filters
                - total_predictions
                - strong_buy_count
                - buy_count
                - hold_count
                - avoid_count
                - avg_probability
                - max_probability
                - min_probability
                - model_version
                - screening_method
                
        Returns:
            True if successful
        """
        
        try:
            sanitized = self._sanitize_dict(log_data)
            
            response = self.client.table(self.screening_log_table)\
                .upsert(sanitized, on_conflict="screening_date")\
                .execute()
            
            self.logger.info("✓ Wrote screening log")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write screening log: {e}")
            return False
    
    def write_accuracy_records(self, accuracy_records: List[Dict[str, Any]]) -> int:
        """
        Write accuracy tracking records
        
        Args:
            accuracy_records: List of accuracy dictionaries with keys:
                - symbol
                - prediction_date
                - predicted_probability
                - predicted_signal
                - predicted_target_gain
                - predicted_target_price
                - became_winner
                - actual_gain_pct
                - actual_high_pct
                - actual_price
                - prediction_correct
                - gain_error_pct
                - gain_error_ratio
                - actual_recorded_at
                
        Returns:
            Number of records written
        """
        
        if not accuracy_records:
            return 0
        
        try:
            sanitized = [self._sanitize_dict(rec) for rec in accuracy_records]
            
            response = self.client.table(self.accuracy_table)\
                .upsert(sanitized, on_conflict="symbol,prediction_date")\
                .execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"✓ Wrote {count} accuracy records")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to write accuracy records: {e}", exc_info=True)
            raise
    
    def get_predictions_for_date(self, prediction_date: str) -> pd.DataFrame:
        """
        Get all predictions for a specific date
        
        Args:
            prediction_date: Date in ISO format (YYYY-MM-DD)
            
        Returns:
            DataFrame with predictions
        """
        
        try:
            response = self.client.table(self.predictions_table)\
                .select("*")\
                .eq("prediction_date", prediction_date)\
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Failed to get predictions: {e}")
            return pd.DataFrame()
    
    def get_historical_prediction_accuracy(
        self,
        days_back: int = 30,
        min_probability: float = 0.5
    ) -> pd.DataFrame:
        """
        Get historical prediction accuracy for calibration
        
        Args:
            days_back: Number of days to look back
            min_probability: Minimum probability threshold
            
        Returns:
            DataFrame with historical predictions and actual outcomes
        """
        
        try:
            start_date = (datetime.now().date() - timedelta(days=days_back)).isoformat()
            
            response = self.client.table(self.accuracy_table)\
                .select("*")\
                .gte("prediction_date", start_date)\
                .gte("predicted_probability", min_probability)\
                .is_("became_winner", "not.null")\
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            # Rename for consistency
            if 'predicted_probability' in df.columns:
                df['probability'] = df['predicted_probability']
            if 'actual_gain_pct' in df.columns:
                df['actual_gain_pct'] = df['actual_gain_pct'].fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical accuracy: {e}")
            return pd.DataFrame()
    
    def get_recent_screening_logs(self, limit: int = 10) -> pd.DataFrame:
        """Get recent screening logs"""
        
        try:
            response = self.client.table(self.screening_log_table)\
                .select("*")\
                .order("screening_date", desc=True)\
                .limit(limit)\
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Failed to get screening logs: {e}")
            return pd.DataFrame()
    
    def get_winners_day_prior_close(
        self,
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get winners day_prior_close data (T-1 4pm indicators)
        This is the PRIMARY training data source
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum rows to return
            
        Returns:
            DataFrame with T-1 close indicators
        """
        
        try:
            query = self.client.table("winners_day_prior_close").select("*")
            
            if start_date:
                query = query.gte("detection_date", start_date)
            
            if end_date:
                query = query.lte("detection_date", end_date)
            
            query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading winners day_prior_close: {e}")
            return pd.DataFrame()
    
    def get_winners_day_prior_open(
        self,
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get winners day_prior_open data (T-1 9:30am indicators)
        SECONDARY training data source
        """
        
        try:
            query = self.client.table("winners_day_prior_open").select("*")
            
            if start_date:
                query = query.gte("detection_date", start_date)
            
            if end_date:
                query = query.lte("detection_date", end_date)
            
            query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading winners day_prior_open: {e}")
            return pd.DataFrame()
    
    def get_non_winners_day_prior_close(
        self,
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get non-winners day_prior_close data (NEGATIVE EXAMPLES)
        Critical for preventing false positives
        """
        
        try:
            query = self.client.table("non_winners_day_prior_close").select("*")
            
            if start_date:
                query = query.gte("detection_date", start_date)
            
            if end_date:
                query = query.lte("detection_date", end_date)
            
            query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading non_winners day_prior_close: {e}")
            return pd.DataFrame()
    
    def get_non_winners_day_prior_open(
        self,
        start_date: str = None,
        end_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get non-winners day_prior_open data"""
        
        try:
            query = self.client.table("non_winners_day_prior_open").select("*")
            
            if start_date:
                query = query.gte("detection_date", start_date)
            
            if end_date:
                query = query.lte("detection_date", end_date)
            
            query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading non_winners day_prior_open: {e}")
            return pd.DataFrame()
    
    def get_daily_winners(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Get actual daily winners (for labeling training data)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with actual winners
        """
        
        try:
            query = self.client.table("daily_winners").select(
                "symbol,detection_date,change_pct,price,volume,high,low,open,close"
            )
            
            if start_date:
                query = query.gte("detection_date", start_date)
            
            if end_date:
                query = query.lte("detection_date", end_date)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading daily winners: {e}")
            return pd.DataFrame()
    
    def read_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        min_probability: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Read predictions from database
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Filter by symbol
            min_probability: Minimum probability threshold
            
        Returns:
            DataFrame with predictions
        """
        
        try:
            query = self.client.table(self.predictions_table).select("*")
            
            if start_date:
                query = query.gte("prediction_date", start_date)
            
            if end_date:
                query = query.lte("prediction_date", end_date)
            
            if symbol:
                query = query.eq("symbol", symbol)
            
            if min_probability:
                query = query.gte("explosion_probability", min_probability)
            
            response = query.execute()
            
            if not response.data:
                self.logger.info("No predictions found")
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            self.logger.info(f"Retrieved {len(df)} predictions")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read predictions: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_prediction_accuracy_stats(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """
        Calculate overall prediction accuracy statistics
        
        Returns:
            Dictionary with accuracy metrics
        """
        
        try:
            query = self.client.table(self.accuracy_table).select("*")
            
            if start_date:
                query = query.gte("prediction_date", start_date)
            
            if end_date:
                query = query.lte("prediction_date", end_date)
            
            response = query.execute()
            
            if not response.data:
                return {'error': 'No accuracy data found'}
            
            df = pd.DataFrame(response.data)
            
            # Calculate metrics
            total = len(df)
            correct = df['prediction_correct'].sum()
            
            winners = df[df['became_winner'] == True]
            non_winners = df[df['became_winner'] == False]
            
            predicted_winners = df[df['predicted_probability'] >= 0.5]
            true_positives = len(predicted_winners[predicted_winners['became_winner'] == True])
            false_positives = len(predicted_winners[predicted_winners['became_winner'] == False])
            false_negatives = len(winners[winners['predicted_probability'] < 0.5])
            
            precision = true_positives / len(predicted_winners) if len(predicted_winners) > 0 else 0
            recall = true_positives / len(winners) if len(winners) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'total_predictions': total,
                'correct_predictions': int(correct),
                'accuracy': correct / total if total > 0 else 0,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_actual_winners': len(winners),
                'predicted_winners': len(predicted_winners)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy stats: {e}")
            return {'error': str(e)}

