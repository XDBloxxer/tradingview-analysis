"""
ML Prediction Supabase Client - OPTIMIZED FOR MINIMAL EGRESS
Follows same patterns as daily_winners and backtesting clients
"""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from supabase import create_client, Client
import pandas as pd
import numpy as np


class MLPredictionSupabaseClient:
    """
    Handler for ML prediction data - EGRESS OPTIMIZED
    Only selects necessary columns, uses batching, implements caching
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Suppress httpx logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials")
        
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise
        
        # Table names
        self.tables = {
            "predictions": "ml_explosion_predictions",
            "accuracy": "ml_prediction_accuracy",
            "self_discovered": "ml_self_discovered_stocks"  # NEW - stocks model found itself
        }
    
    # ===== OPTIMIZED READ METHODS (MINIMAL EGRESS) =====
    
    def get_latest_day_prior_close(
        self, 
        detection_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get T-1 close data - ONLY COLUMNS NEEDED FOR PREDICTION
        EGRESS OPTIMIZATION: Select only indicator columns
        """
        try:
            # CRITICAL: Only select columns we actually need
            # This reduces egress by ~80% compared to SELECT *
            select_cols = (
                "symbol,exchange,detection_date,"
                # Momentum
                "rsi,\"rsi[1]\",\"rsi[2]\",mom,\"mom[1]\","
                "\"stoch.k\",\"stoch.d\",\"stoch.k[1]\",\"stoch.d[1]\","
                "\"w.r\",ao,uo,roc,kama,tsi,"
                # Trend
                "\"macd.macd\",\"macd.signal\",macd_diff,"
                "adx,\"adx+di\",\"adx-di\",cci20,"
                "aroon_up,aroon_down,aroon_indicator,"
                # Moving Averages
                "ema5,ema10,ema20,ema50,ema100,ema200,"
                "sma5,sma10,sma20,sma50,sma100,sma200,"
                # Volatility
                "atr,atr_pct,\"bb.upper\",\"bb.lower\",\"bb.middle\",bb_width,bbpower,"
                "volatility_20d,keltner_upper,keltner_lower,donchian_upper,donchian_lower,"
                # Volume
                "volume,volume_sma5,volume_sma10,volume_sma20,volume_ratio,obv,cmf,"
                # Price
                "close,open,high,low,vwap,"
                # Price Changes
                "price_change_1d,price_change_2d,price_change_3d,price_change_5d,price_change_10d,price_change_20d,"
                # Other
                "high_52w,low_52w,\"gap_%\""
            )
            
            query = self.client.table("winners_day_prior_close").select(select_cols)
            
            if detection_date:
                query = query.eq("detection_date", detection_date)
            else:
                # Get most recent date
                response = self.client.table("winners_day_prior_close")\
                    .select("detection_date")\
                    .order("detection_date", desc=True)\
                    .limit(1)\
                    .execute()
                
                if response.data:
                    latest_date = response.data[0]['detection_date']
                    query = query.eq("detection_date", latest_date)
            
            query = query.limit(limit)
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error fetching day prior close: {e}")
            return pd.DataFrame()
    
    def get_historical_prediction_accuracy(
        self, 
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Get historical accuracy data for gain calibration
        EGRESS OPTIMIZED: Only needed columns
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Only select columns needed for calibration
            response = self.client.table(self.tables["accuracy"])\
                .select("predicted_probability,actual_gain_pct,became_winner")\
                .gte("prediction_date", start_date.isoformat())\
                .lte("prediction_date", end_date.isoformat())\
                .eq("became_winner", True)\
                .limit(1000)\
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df = df.rename(columns={'predicted_probability': 'probability'})
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical accuracy: {e}")
            return pd.DataFrame()
    
    def get_predictions(
        self, 
        prediction_date: str = None, 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get predictions - OPTIMIZED COLUMN SELECTION
        """
        try:
            # Only select display columns
            select_cols = (
                "symbol,exchange,prediction_date,explosion_probability,prediction,signal,"
                "target_gain_pct,target_gain_low,target_gain_high,"
                "current_price,target_price,target_price_low,target_price_high,"
                "rsi,macd,adx,volume_ratio,hv_20,bb_width"
            )
            
            query = self.client.table(self.tables["predictions"]).select(select_cols)
            
            if prediction_date:
                query = query.eq("prediction_date", prediction_date)
            
            query = query.order("explosion_probability", desc=True).limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error fetching predictions: {e}")
            return pd.DataFrame()
    
    def get_accuracy_data(
        self,
        prediction_date: str = None,
        days_back: int = 30
    ) -> pd.DataFrame:
        """Get accuracy tracking data - OPTIMIZED"""
        try:
            query = self.client.table(self.tables["accuracy"]).select(
                "symbol,prediction_date,predicted_probability,predicted_signal,"
                "predicted_target_gain,predicted_target_price,"
                "became_winner,actual_gain_pct,actual_price,"
                "prediction_correct,gain_error_pct"
            )
            
            if prediction_date:
                query = query.eq("prediction_date", prediction_date)
            else:
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days_back)
                query = query.gte("prediction_date", start_date.isoformat())\
                             .lte("prediction_date", end_date.isoformat())
            
            query = query.limit(1000)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error fetching accuracy data: {e}")
            return pd.DataFrame()
    
    # ===== WRITE METHODS (OPTIMIZED BATCHING) =====
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize value for PostgreSQL"""
        if value is None or pd.isna(value):
            return None
        
        if isinstance(value, (np.integer, int)):
            return int(value)
        
        if isinstance(value, (np.floating, float)):
            if np.isinf(value) or np.isnan(value):
                return None
            return float(value)
        
        if isinstance(value, np.bool_):
            return bool(value)
        
        return value
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary and remove auto-fields"""
        auto_fields = {'id', 'created_at', 'updated_at'}
        return {
            k: self._sanitize_value(v) 
            for k, v in data.items() 
            if k not in auto_fields
        }
    
    def write_predictions(
        self, 
        predictions: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> int:
        """
        Write predictions - OPTIMIZED BATCHING
        """
        if not predictions:
            return 0
        
        try:
            sanitized = [self._sanitize_dict(p) for p in predictions]
            
            total_written = 0
            
            for i in range(0, len(sanitized), batch_size):
                batch = sanitized[i:i + batch_size]
                
                response = self.client.table(self.tables["predictions"]).upsert(
                    batch,
                    on_conflict="symbol,prediction_date"
                ).execute()
                
                total_written += len(batch)
            
            self.logger.info(f"✓ Wrote {total_written} predictions")
            return total_written
            
        except Exception as e:
            self.logger.error(f"Error writing predictions: {e}")
            raise
    
    def write_accuracy_records(
        self,
        accuracy_records: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> int:
        """Write accuracy tracking records - OPTIMIZED"""
        if not accuracy_records:
            return 0
        
        try:
            sanitized = [self._sanitize_dict(r) for r in accuracy_records]
            
            total_written = 0
            
            for i in range(0, len(sanitized), batch_size):
                batch = sanitized[i:i + batch_size]
                
                response = self.client.table(self.tables["accuracy"]).upsert(
                    batch,
                    on_conflict="symbol,prediction_date"
                ).execute()
                
                total_written += len(batch)
            
            self.logger.info(f"✓ Wrote {total_written} accuracy records")
            return total_written
            
        except Exception as e:
            self.logger.error(f"Error writing accuracy records: {e}")
            raise
    
    def write_self_discovered_stocks(
        self,
        stocks: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> int:
        """
        Write stocks that model discovered on its own (not in daily winners)
        Used for tracking model's independent discovery accuracy
        """
        if not stocks:
            return 0
        
        try:
            sanitized = [self._sanitize_dict(s) for s in stocks]
            
            total_written = 0
            
            for i in range(0, len(sanitized), batch_size):
                batch = sanitized[i:i + batch_size]
                
                response = self.client.table(self.tables["self_discovered"]).upsert(
                    batch,
                    on_conflict="symbol,prediction_date"
                ).execute()
                
                total_written += len(batch)
            
            self.logger.info(f"✓ Wrote {total_written} self-discovered stocks")
            return total_written
            
        except Exception as e:
            self.logger.error(f"Error writing self-discovered stocks: {e}")

            raise
    # Add to MLPredictionSupabaseClient class

def write_screening_log(self, log_data: dict) -> bool:
    """Write screening statistics log"""
    try:
        sanitized = self._sanitize_dict(log_data)
        
        self.client.table("ml_screening_logs").upsert(
            sanitized,
            on_conflict="screening_date"
        ).execute()
        
        return True
    except Exception as e:
        self.logger.error(f"Failed to write screening log: {e}")
        return False

def get_latest_screening_stats(self) -> dict:
    """Get latest screening statistics"""
    try:
        response = self.client.table("ml_screening_logs")\
            .select("*")\
            .order("screening_date", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data:
            return response.data[0]
        return {}
    except Exception as e:
        self.logger.error(f"Failed to get screening stats: {e}")
        return {}

def get_accuracy_summary(self, days_back: int = 30) -> pd.DataFrame:
    """Get accuracy summary from materialized view"""
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        response = self.client.table("v_ml_daily_accuracy_summary")\
            .select("*")\
            .gte("prediction_date", start_date.isoformat())\
            .lte("prediction_date", end_date.isoformat())\
            .execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        self.logger.error(f"Failed to get accuracy summary: {e}")
        return pd.DataFrame()

def get_signal_performance(self) -> pd.DataFrame:
    """Get signal performance from materialized view"""
    try:
        response = self.client.table("v_ml_signal_performance")\
            .select("*")\
            .execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        self.logger.error(f"Failed to get signal performance: {e}")
        return pd.DataFrame()

def get_missed_opportunities_summary(self, days_back: int = 30) -> pd.DataFrame:
    """Get missed opportunities summary"""
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        response = self.client.table("v_ml_missed_summary")\
            .select("*")\
            .gte("detection_date", start_date.isoformat())\
            .lte("detection_date", end_date.isoformat())\
            .execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        self.logger.error(f"Failed to get missed opportunities: {e}")
        return pd.DataFrame()

def get_false_positive_patterns(self) -> pd.DataFrame:
    """Get false positive patterns from materialized view"""
    try:
        response = self.client.table("v_ml_false_positive_patterns")\
            .select("*")\
            .execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        self.logger.error(f"Failed to get FP patterns: {e}")
        return pd.DataFrame()
