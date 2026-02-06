"""
Supabase client for backtesting data
UPDATED: Includes methods to query historical_market_data table
FIXED: Better date range queries with distinct dates
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
    """Handler for backtest data in Supabase"""
    
    def __init__(self, config: dict):
        """Initialize Supabase client"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Get credentials from environment
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            self.logger.info(f"✓ Connected to Supabase")
        except Exception as e:
            self.logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise
        
        # Table names
        self.tables = {
            "strategies": "backtest_strategies",
            "results": "backtest_results",
            "trades": "backtest_trades",
            "historical": "historical_market_data"  # NEW!
        }
    
    # ========================================================================
    # HISTORICAL DATA QUERY METHODS (FIXED)
    # ========================================================================
    
    def get_available_dates(self, start_date: date, end_date: date) -> List[date]:
        """
        Get list of available trading dates in database within range
        FIXED: Uses proper query to get distinct dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dates
        """
        try:
            # Query with large limit to get all dates, then dedupe in Python
            response = self.client.table(self.tables["historical"]) \
                .select("date") \
                .gte("date", start_date.isoformat()) \
                .lte("date", end_date.isoformat()) \
                .order("date") \
                .limit(100000) \
                .execute()
            
            if not response.data:
                self.logger.warning(f"No data found for date range {start_date} to {end_date}")
                return []
            
            # Get unique dates and sort
            dates = sorted(list(set(
                datetime.fromisoformat(row['date']).date() 
                for row in response.data
            )))
            
            self.logger.info(f"Found {len(dates)} unique dates between {start_date} and {end_date}")
            
            return dates
            
        except Exception as e:
            self.logger.error(f"Error getting available dates: {e}")
            return []
    
    def get_top_gainers(self, target_date: date, top_n: int = 20) -> List[str]:
        """
        Get top N gainers for a specific date from database
        
        Args:
            target_date: Date to query
            top_n: Number of top gainers
            
        Returns:
            List of symbols
        """
        try:
            response = self.client.table(self.tables["historical"]) \
                .select("symbol, change_pct") \
                .eq("date", target_date.isoformat()) \
                .order("change_pct", desc=True) \
                .limit(top_n) \
                .execute()
            
            if not response.data:
                return []
            
            return [row['symbol'] for row in response.data]
            
        except Exception as e:
            self.logger.error(f"Error getting top gainers: {e}")
            return []
    
    def get_all_stocks_for_date(
        self,
        target_date: date,
        min_price: float = 0.25,
        max_price: Optional[float] = None,
        min_volume: int = 100000
    ) -> List[str]:
        """
        Get all stocks trading on a date that meet filters
        FIXED: Increased limit to get more stocks
        
        Args:
            target_date: Date to query
            min_price: Minimum price
            max_price: Maximum price (optional)
            min_volume: Minimum volume
            
        Returns:
            List of symbols
        """
        try:
            query = self.client.table(self.tables["historical"]) \
                .select("symbol") \
                .eq("date", target_date.isoformat()) \
                .gte("close", min_price) \
                .gte("volume", min_volume) \
                .limit(10000)  # Increased limit to get all stocks
            
            if max_price:
                query = query.lte("close", max_price)
            
            response = query.execute()
            
            if not response.data:
                return []
            
            symbols = [row['symbol'] for row in response.data]
            self.logger.debug(f"Found {len(symbols)} stocks on {target_date} matching filters")
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting stocks for date: {e}")
            return []
    
    def get_stock_data(self, symbol: str, target_date: date) -> Optional[Dict]:
        """
        Get OHLCV data for a stock on a specific date
        
        Args:
            symbol: Stock symbol
            target_date: Date to query
            
        Returns:
            Dictionary with OHLCV data or None
        """
        try:
            response = self.client.table(self.tables["historical"]) \
                .select("*") \
                .eq("symbol", symbol) \
                .eq("date", target_date.isoformat()) \
                .limit(1) \
                .execute()
            
            if not response.data:
                # Try closest prior date within 7 days
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
            
        except Exception as e:
            self.logger.debug(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def get_stock_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> List[Dict]:
        """
        Get historical OHLCV data for a stock over a date range
        Used for indicator calculation
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dictionaries with OHLCV data
        """
        try:
            response = self.client.table(self.tables["historical"]) \
                .select("*") \
                .eq("symbol", symbol) \
                .gte("date", start_date.isoformat()) \
                .lte("date", end_date.isoformat()) \
                .order("date") \
                .limit(500) \
                .execute()
            
            if not response.data:
                return []
            
            return response.data
            
        except Exception as e:
            self.logger.debug(f"Error getting stock history for {symbol}: {e}")
            return []
    
    # ========================================================================
    # EXISTING METHODS (UNCHANGED)
    # ========================================================================
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize value for PostgreSQL"""
        if value is None:
            return None
        
        if pd.isna(value):
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
        """
        Create a new strategy record
        
        Returns:
            Strategy ID
        """
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
            
            strategy_id = response.data[0]['id']
            self.logger.info(f"Created strategy {strategy_id}: {strategy_config['name']}")
            
            return strategy_id
            
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
            
            # Parse JSON field
            if 'indicator_criteria' in strategy:
                strategy['indicator_criteria'] = json.loads(strategy['indicator_criteria'])
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error getting strategy: {e}")
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
            
            self.logger.info(f"Updated strategy {strategy_id} status to {status}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy status: {e}")
    
    def write_daily_results(self, strategy_id: int, daily_results: List[Dict]):
        """Write daily results"""
        if not daily_results:
            return
        
        try:
            # Add strategy_id to each record
            for result in daily_results:
                result['strategy_id'] = strategy_id
            
            # Sanitize
            sanitized = [self._sanitize_dict(r) for r in daily_results]
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(sanitized), batch_size):
                batch = sanitized[i:i + batch_size]
                
                self.client.table(self.tables["results"]).upsert(
                    batch,
                    on_conflict="strategy_id,test_date"
                ).execute()
            
            self.logger.info(f"Wrote {len(daily_results)} daily results")
            
        except Exception as e:
            self.logger.error(f"Error writing daily results: {e}")
            raise
    
    def write_trades(self, strategy_id: int, trades: List[Dict]):
        """Write trade records"""
        if not trades:
            return
        
        try:
            # Add strategy_id to each trade
            for trade in trades:
                trade['strategy_id'] = strategy_id
                
                # Convert indicator_values dict to JSON string
                if 'indicator_values' in trade:
                    trade['indicator_values'] = json.dumps(trade['indicator_values'])
            
            # Sanitize
            sanitized = [self._sanitize_dict(t) for t in trades]
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(sanitized), batch_size):
                batch = sanitized[i:i + batch_size]
                
                self.client.table(self.tables["trades"]).upsert(
                    batch,
                    on_conflict="strategy_id,symbol,signal_date"
                ).execute()
            
            self.logger.info(f"Wrote {len(trades)} trades")
            
        except Exception as e:
            self.logger.error(f"Error writing trades: {e}")
            raise
    
    def update_strategy_summary(self, strategy_id: int, stats: Dict):
        """Update strategy with overall statistics"""
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
            
            self.logger.info(f"Updated strategy {strategy_id} summary")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy summary: {e}")
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
            
        except Exception as e:
            self.logger.error(f"Error getting daily results: {e}")
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
            
            # Parse indicator_values JSON
            if 'indicator_values' in df.columns:
                df['indicator_values'] = df['indicator_values'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return pd.DataFrame()
