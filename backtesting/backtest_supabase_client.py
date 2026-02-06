"""
Supabase client for backtesting data
ENHANCED: Handles new exit analysis fields
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from supabase import create_client, Client


class BacktestSupabaseClient:
    """Handler for backtest data in Supabase - ENHANCED"""
    
    def __init__(self, config: dict):
        """Initialize Supabase client"""
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
        
        self.tables = {
            "strategies": "backtest_strategies",
            "results": "backtest_results",
            "trades": "backtest_trades",
            "historical": "historical_market_data"
        }
    
    def get_available_dates(self, start_date: date, end_date: date) -> List[date]:
        """Get all unique trading dates - OPTIMIZED to minimize egress"""
        try:
            self.logger.info(f"Querying unique dates from {start_date} to {end_date}")
            
            # KEY CHANGE: Only select the 'date' column, not all columns (*)
            # This reduces egress by ~95% since we don't fetch OHLCV data
            all_dates = set()
            offset = 0
            batch_size = 1000
            total_rows_processed = 0
            
            while True:
                response = self.client.table(self.tables["historical"]) \
                    .select("date") \  # <-- ONLY fetch date column!
                    .gte("date", start_date.isoformat()) \
                    .lte("date", end_date.isoformat()) \
                    .order("date") \
                    .range(offset, offset + batch_size - 1) \
                    .execute()
                
                if not response.data:
                    break
                
                rows_fetched = len(response.data)
                total_rows_processed += rows_fetched
                
                # Add dates to set (deduplicates automatically)
                for row in response.data:
                    all_dates.add(datetime.fromisoformat(row['date']).date())
                
                # If we got fewer rows than batch_size, we're done
                if rows_fetched < batch_size:
                    break
                
                offset += batch_size
                
                # Safety limit (keep this)
                if offset > 5000000:
                    self.logger.warning("Hit 5M row safety limit")
                    break
            
            available_dates = sorted(list(all_dates))
            
            self.logger.info(
                f"Processed {total_rows_processed} total rows, "
                f"found {len(available_dates)} unique dates"
            )
            
            if not available_dates:
                self.logger.warning("No dates found, using business day fallback")
                business_days = pd.bdate_range(start=start_date, end=end_date)
                return [d.date() for d in business_days]
            
            self.logger.info(f"Date range: {available_dates[0]} to {available_dates[-1]}")
            return available_dates
            
        except Exception as e:
            self.logger.error(f"Error getting dates: {e}", exc_info=True)
            business_days = pd.bdate_range(start=start_date, end=end_date)
            return [d.date() for d in business_days]
    
    def get_top_gainers(self, target_date: date, top_n: int = 5) -> List[str]:
        """Get top N gainers for a date"""
        try:
            response = self.client.table(self.tables["historical"]) \
                .select("symbol") \
                .eq("date", target_date.isoformat()) \
                .order("change_pct", desc=True) \
                .limit(top_n) \
                .execute()
            
            if not response.data:
                return []
            
            return [row['symbol'] for row in response.data]
            
        except:
            return []
    
    def get_stock_data(self, symbol: str, target_date: date) -> Optional[Dict]:
        """Get OHLCV data for a stock on a specific date"""
        try:
            response = self.client.table(self.tables["historical"]) \
                .select("*") \
                .eq("symbol", symbol) \
                .eq("date", target_date.isoformat()) \
                .limit(1) \
                .execute()
            
            if not response.data:
                # Try closest prior date
                prior_date = target_date - timedelta(days=7)
                response = self.client.table(self.tables["historical"]) \
                    .select("*") \
                    .eq("symbol", symbol) \
                    .gte("date", prior_date.isoformat()) \
                    .lte("date", target_date.isoformat()) \
                    .order("date", desc=True) \
                    .limit(1) \
                    .execute()
                
                if not response.data:
                    return None
            
            return response.data[0]
            
        except:
            return None
    
    def get_stock_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> List[Dict]:
        """Get historical data - OPTIMIZED with column selection"""
        try:
            all_data = []
            offset = 0
            batch_size = 1000  # Reduced from 5000 for better streaming
            
            while True:
                # KEY CHANGE: Only select columns needed for indicators
                # Add any other columns you need here (volume, etc.)
                response = self.client.table(self.tables["historical"]) \
                    .select("date,open,high,low,close,volume") \  # <-- Specific columns only!
                    .eq("symbol", symbol) \
                    .gte("date", start_date.isoformat()) \
                    .lte("date", end_date.isoformat()) \
                    .order("date") \
                    .range(offset, offset + batch_size - 1) \
                    .execute()
                
                if not response.data:
                    break
                
                all_data.extend(response.data)
                
                if len(response.data) < batch_size:
                    break
                
                offset += batch_size
            
            return all_data
            
        except:
            return []
    
    # ========================================================================
    # WRITE METHODS - OPTIMIZED
    # ========================================================================
    
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
        """Sanitize all values in a dictionary"""
        return {k: self._sanitize_value(v) for k, v in data.items()}
    
    def create_strategy(self, strategy_config: Dict[str, Any]) -> int:
        """Create a new strategy record"""
        try:
            data = {
                'name': strategy_config['name'],
                'description': strategy_config.get('description', ''),
                'start_date': strategy_config['start_date'],
                'end_date': strategy_config['end_date'],
                'target_min_gain_pct': strategy_config['target_min_gain_pct'],
                'target_days': strategy_config.get('target_days', 1),
                'indicator_criteria': json.dumps(strategy_config['indicator_criteria']),
                'min_price': strategy_config.get('min_price', 0.25),
                'max_price': strategy_config.get('max_price'),
                'min_volume': strategy_config.get('min_volume', 100000),
                'exchanges': strategy_config.get('exchanges', ['NASDAQ', 'NYSE', 'AMEX']),
                'run_status': 'pending'
            }
            
            response = self.client.table(self.tables["strategies"]).insert(data).execute()
            return response.data[0]['id']
            
        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            raise
    
    def get_strategy(self, strategy_id: int) -> Optional[Dict]:
        """Get strategy by ID"""
        try:
            response = self.client.table(self.tables["strategies"]) \
                .select("*") \
                .eq("id", strategy_id) \
                .execute()
            
            if not response.data:
                return None
            
            strategy = response.data[0]
            
            if 'indicator_criteria' in strategy:
                strategy['indicator_criteria'] = json.loads(strategy['indicator_criteria'])
            
            return strategy
            
        except:
            return None
    
    def update_strategy_status(self, strategy_id: int, status: str):
        """Update strategy run status"""
        try:
            self.client.table(self.tables["strategies"]) \
                .update({
                    'run_status': status,
                    'last_run_at': datetime.now().isoformat()
                }) \
                .eq("id", strategy_id) \
                .execute()
            
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    def write_daily_results(self, strategy_id: int, daily_results: List[Dict]):
        """Write daily results - OPTIMIZED with larger batches"""
        if not daily_results:
            return
        
        try:
            self.logger.info(f"Writing {len(daily_results)} daily results for strategy {strategy_id}")
            
            for result in daily_results:
                result['strategy_id'] = strategy_id
            
            sanitized = [self._sanitize_dict(r) for r in daily_results]
            
            # Larger batches
            batch_size = 500
            total_written = 0
            
            for i in range(0, len(sanitized), batch_size):
                batch = sanitized[i:i + batch_size]
                
                response = self.client.table(self.tables["results"]).upsert(
                    batch,
                    on_conflict="strategy_id,test_date"
                ).execute()
                
                total_written += len(batch)
            
            self.logger.info(f"✓ Successfully wrote {total_written} daily results")
            
        except Exception as e:
            self.logger.error(f"Error writing daily results: {e}", exc_info=True)
            raise
    
    def write_trades(self, strategy_id: int, trades: List[Dict]):
        """Write trade records - handles new fields gracefully"""
        if not trades:
            return
        
        try:
            self.logger.info(f"Writing {len(trades)} trades for strategy {strategy_id}")
            
            for trade in trades:
                trade['strategy_id'] = strategy_id
                
                if 'indicator_values' in trade:
                    trade['indicator_values'] = json.dumps(trade['indicator_values'])
            
            sanitized = [self._sanitize_dict(t) for t in trades]
            
            # Larger batches
            batch_size = 500
            total_written = 0
            
            for i in range(0, len(sanitized), batch_size):
                batch = sanitized[i:i + batch_size]
                
                response = self.client.table(self.tables["trades"]).upsert(
                    batch,
                    on_conflict="strategy_id,symbol,signal_date"
                ).execute()
                
                total_written += len(batch)
            
            self.logger.info(f"✓ Successfully wrote {total_written} trades")
            
        except Exception as e:
            self.logger.error(f"Error writing trades: {e}", exc_info=True)
            raise
    
    def update_strategy_summary(self, strategy_id: int, stats: Dict):
        """Update strategy with overall statistics - includes new metrics"""
        try:
            update_data = {
                'total_matches': stats.get('total_matches', 0),
                'true_positives': stats.get('true_positives', 0),
                'false_positives': stats.get('false_positives', 0),
                'missed_opportunities': stats.get('missed_opportunities', 0),
                'accuracy_pct': stats.get('accuracy_pct'),
                'avg_gain_pct': stats.get('avg_gain_pct'),
                'updated_at': datetime.now().isoformat()
            }
            
            self.client.table(self.tables["strategies"]) \
                .update(update_data) \
                .eq("id", strategy_id) \
                .execute()
            
        except Exception as e:
            self.logger.error(f"Error updating summary: {e}")
            raise
    
    def get_daily_results(self, strategy_id: int) -> pd.DataFrame:
        """Get daily results for a strategy"""
        try:
            response = self.client.table(self.tables["results"]) \
                .select("*") \
                .eq("strategy_id", strategy_id) \
                .order("test_date") \
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except:
            return pd.DataFrame()
    
    def get_trades(self, strategy_id: int, limit: Optional[int] = None) -> pd.DataFrame:
        """Get trades for a strategy"""
        try:
            query = self.client.table(self.tables["trades"]) \
                .select("*") \
                .eq("strategy_id", strategy_id) \
                .order("signal_date", desc=True)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            if 'indicator_values' in df.columns:
                df['indicator_values'] = df['indicator_values'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            
            return df
            
        except:
            return pd.DataFrame()
