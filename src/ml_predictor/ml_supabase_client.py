"""
ML Supabase Client - Handles ML prediction storage and retrieval
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import pandas as pd
from supabase import create_client, Client


class MLPredictionSupabaseClient:
    """
    Client for storing and retrieving ML predictions
    """
    
    def __init__(self, config: dict):
        """Initialize Supabase client"""
        self.logger = logging.getLogger(__name__)
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and KEY must be provided in config")
        
        self.client: Client = create_client(supabase_url, supabase_key)
        
        # Table names
        self.predictions_table = "ml_predictions"
        
        self.logger.info("ML Supabase client initialized")
    
    def write_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """
        Write ML predictions to database
        
        Args:
            predictions: List of prediction dictionaries
            
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
            sanitized = []
            for pred in new_predictions:
                record = {
                    'symbol': pred['symbol'],
                    'prediction_date': pred['prediction_date'],
                    'probability': float(pred['probability']),
                    'confidence': pred['confidence'],
                    'prediction': int(pred['prediction']),
                    'model_version': pred.get('model_version', 'unknown'),
                    'indicators': pred.get('indicators', {})
                }
                sanitized.append(record)
            
            # Write to database
            response = self.client.table(self.predictions_table).insert(sanitized).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"✓ Wrote {count} predictions to database")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to write predictions: {e}", exc_info=True)
            raise
    
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
                query = query.gte("probability", min_probability)
            
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
    
    def get_prediction_accuracy(
        self,
        prediction_date: str
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy for a given date
        by comparing with actual winners
        
        Args:
            prediction_date: Date of predictions (YYYY-MM-DD)
            
        Returns:
            Dictionary with accuracy metrics
        """
        
        try:
            # Get predictions
            predictions = self.client.table(self.predictions_table)\
                .select("symbol,probability,prediction")\
                .eq("prediction_date", prediction_date)\
                .execute()
            
            if not predictions.data:
                return {'error': 'No predictions found'}
            
            pred_df = pd.DataFrame(predictions.data)
            
            # Get actual winners (stocks that exploded on this date)
            winners = self.client.table("daily_winners")\
                .select("symbol,change_pct")\
                .eq("detection_date", prediction_date)\
                .execute()
            
            if not winners.data:
                return {'error': 'No winners data available yet'}
            
            winner_symbols = set(w['symbol'] for w in winners.data)
            
            # Calculate accuracy
            tp = len(pred_df[pred_df['symbol'].isin(winner_symbols)])  # True positives
            fp = len(pred_df[~pred_df['symbol'].isin(winner_symbols)])  # False positives
            
            # Get false negatives (winners we didn't predict)
            fn = len(winner_symbols - set(pred_df['symbol']))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'prediction_date': prediction_date,
                'total_predictions': len(pred_df),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'actual_winners': len(winner_symbols),
                'predicted_symbols': list(pred_df['symbol']),
                'winner_symbols': list(winner_symbols)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy: {e}", exc_info=True)
            return {'error': str(e)}

