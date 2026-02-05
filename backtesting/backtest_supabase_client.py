"""
Supabase client for backtesting data
Handles reading/writing backtest strategies, results, and trades
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from supabase import create_client, Client
import json


class BacktestSupabaseClient:
    """
    Handler for backtesting data in Supabase
    """
    
    def __init__(self, config: dict):
        """
        Initialize Supabase client for backtesting
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Get credentials
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            self.logger.info(f"✓ Connected to Supabase for backtesting")
        except Exception as e:
            self.logger.error(f"Failed to connect to Supabase: {e}")
            raise
        
        # Table names
        self.tables = {
            "strategies": "backtest_strategies",
            "results": "backtest_results",
            "trades": "backtest_trades"
        }
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize value for Supabase"""
    
        if value is None:
            return None
    
        # Handle pandas / numpy arrays or Series
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            # Convert numpy arrays / Series to Python lists
            if isinstance(value, (np.ndarray, pd.Series)):
                value = value.tolist()
    
            # If list is empty, store NULL
            if len(value) == 0:
                return None
    
            return value
    
        # Handle pandas scalar NA (must be AFTER array handling)
        if pd.isna(value):
            return None
    
        # NumPy scalars
        if isinstance(value, np.integer):
            return int(value)
    
        if isinstance(value, np.floating):
            if np.isinf(value) or np.isnan(value):
                return None
            return float(value)
    
        if isinstance(value, np.bool_):
            return bool(value)
    
        # Native floats
        if isinstance(value, float):
            if np.isinf(value) or np.isnan(value):
                return None
    
        return value

    
    # ========================================================================
    # STRATEGIES
    # ========================================================================
    
    def create_strategy(self, strategy_config: Dict[str, Any]) -> int:
        """
        Create a new backtest strategy
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            Strategy ID
        """
        try:
            # Prepare data
            data = {
                'name': strategy_config['name'],
                'description': strategy_config.get('description', ''),
                'start_date': strategy_config['start_date'],
                'end_date': strategy_config['end_date'],
                'target_min_gain_pct': strategy_config['target_min_gain_pct'],
                'target_days': strategy_config.get('target_days', 1),
                'indicator_criteria': json.dumps(strategy_config['indicator_criteria']),
                'min_price': strategy_config.get('min_price', 0.50),
                'max_price': strategy_config.get('max_price'),
                'min_volume': strategy_config.get('min_volume', 100000),
                'exchanges': strategy_config.get('exchanges', ['NASDAQ', 'NYSE', 'AMEX']),
                'run_status': 'pending'
            }
            
            data = self._sanitize_dict(data)
            
            response = self.client.table(self.tables["strategies"]).insert(data).execute()
            
            strategy_id = response.data[0]['id']
            self.logger.info(f"Created strategy {strategy_id}: {strategy_config['name']}")
            
            return strategy_id
            
        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            raise
    
    def update_strategy_status(self, strategy_id: int, status: str):
        """Update strategy run status"""
        try:
            self.client.table(self.tables["strategies"]).update({
                'run_status': status,
                'last_run_at': 'NOW()' if status == 'completed' else None
            }).eq('id', strategy_id).execute()
        except Exception as e:
            self.logger.error(f"Error updating strategy status: {e}")
    
    def get_strategy(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """Get strategy by ID"""
        try:
            response = self.client.table(self.tables["strategies"]) \
                .select("*") \
                .eq("id", strategy_id) \
                .execute()
            
            if response.data:
                strategy = response.data[0]
                # Parse JSON fields
                if 'indicator_criteria' in strategy and isinstance(strategy['indicator_criteria'], str):
                    strategy['indicator_criteria'] = json.loads(strategy['indicator_criteria'])
                return strategy
            return None
        except Exception as e:
            self.logger.error(f"Error getting strategy: {e}")
            return None
    
    def list_strategies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all strategies"""
        try:
            response = self.client.table(self.tables["strategies"]) \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            
            strategies = response.data if response.data else []
            
            # Parse JSON fields
            for strategy in strategies:
                if 'indicator_criteria' in strategy and isinstance(strategy['indicator_criteria'], str):
                    strategy['indicator_criteria'] = json.loads(strategy['indicator_criteria'])
            
            return strategies
        except Exception as e:
            self.logger.error(f"Error listing strategies: {e}")
            return []
    
    def delete_strategy(self, strategy_id: int):
        """Delete strategy and all related data"""
        try:
            # Cascade delete will handle results and trades
            self.client.table(self.tables["strategies"]) \
                .delete() \
                .eq("id", strategy_id) \
                .execute()
            
            self.logger.info(f"Deleted strategy {strategy_id}")
        except Exception as e:
            self.logger.error(f"Error deleting strategy: {e}")
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    def write_daily_results(
        self,
        strategy_id: int,
        daily_results: List[Dict[str, Any]]
    ) -> int:
        """
        Write daily backtest results
        
        Args:
            strategy_id: Strategy ID
            daily_results: List of daily result dictionaries
            
        Returns:
            Number of rows written
        """
        if not daily_results:
            return 0
        
        try:
            # Add strategy_id to each result
            data = []
            for result in daily_results:
                row = {
                    'strategy_id': strategy_id,
                    'test_date': result['test_date'],
                    'total_scanned': result['total_scanned'],
                    'criteria_matches': result['criteria_matches'],
                    'true_positives': result['true_positives'],
                    'false_positives': result['false_positives'],
                    'missed_opportunities': result['missed_opportunities'],
                    'avg_match_gain_pct': result['avg_match_gain_pct'],
                    'avg_miss_gain_pct': result['avg_miss_gain_pct'],
                    'max_gain_pct': result['max_gain_pct'],
                    'min_gain_pct': result['min_gain_pct']
                }
                data.append(self._sanitize_dict(row))
            
            # Upsert (handles duplicates)
            response = self.client.table(self.tables["results"]).upsert(
                data,
                on_conflict="strategy_id,test_date"
            ).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} daily results")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing daily results: {e}")
            raise
    
    def get_daily_results(
        self,
        strategy_id: int
    ) -> pd.DataFrame:
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
    
    # ========================================================================
    # TRADES
    # ========================================================================
    
    def write_trades(
        self,
        strategy_id: int,
        trades: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Write backtest trades
        
        Args:
            strategy_id: Strategy ID
            trades: List of trade dictionaries
            batch_size: Batch size for writing
            
        Returns:
            Number of rows written
        """
        if not trades:
            return 0
        
        try:
            total_written = 0
            
            # Process in batches
            for i in range(0, len(trades), batch_size):
                batch = trades[i:i + batch_size]
                
                # Add strategy_id and sanitize
                data = []
                for trade in batch:
                    row = {
                        'strategy_id': strategy_id,
                        'symbol': trade['symbol'],
                        'exchange': trade['exchange'],
                        'signal_date': trade['signal_date'],
                        'entry_price': trade['entry_price'],
                        'entry_volume': trade.get('entry_volume'),
                        'indicator_values': json.dumps(trade.get('indicator_values', {})),
                        'matched_criteria': trade['matched_criteria'],
                        'hit_target': trade['hit_target'],
                        'actual_gain_pct': trade.get('actual_gain_pct'),
                        'exit_price': trade.get('exit_price'),
                        'trade_type': trade['trade_type']
                    }
                    data.append(self._sanitize_dict(row))
                
                # Upsert batch
                response = self.client.table(self.tables["trades"]).upsert(
                    data,
                    on_conflict="strategy_id,symbol,signal_date"
                ).execute()
                
                count = len(response.data) if response.data else 0
                total_written += count
                
                self.logger.info(f"Wrote batch {i//batch_size + 1}: {count} trades")
            
            self.logger.info(f"Total trades written: {total_written}")
            return total_written
            
        except Exception as e:
            self.logger.error(f"Error writing trades: {e}")
            raise
    
    def get_trades(
        self,
        strategy_id: int,
        trade_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get trades for a strategy
        
        Args:
            strategy_id: Strategy ID
            trade_type: Optional filter by trade type
            limit: Optional limit
            
        Returns:
            DataFrame of trades
        """
        try:
            query = self.client.table(self.tables["trades"]) \
                .select("*") \
                .eq("strategy_id", strategy_id)
            
            if trade_type:
                query = query.eq("trade_type", trade_type)
            
            if limit:
                query = query.limit(limit)
            
            query = query.order("signal_date", desc=True)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            # Parse JSON fields
            if 'indicator_values' in df.columns:
                df['indicator_values'] = df['indicator_values'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # SUMMARY UPDATES
    # ========================================================================
    
    def update_strategy_summary(
        self,
        strategy_id: int,
        overall_stats: Dict[str, Any]
    ):
        """
        Update strategy with summary statistics
        
        Args:
            strategy_id: Strategy ID
            overall_stats: Overall statistics dictionary
        """
        try:
            data = {
                'total_matches': overall_stats['total_matches'],
                'true_positives': overall_stats['true_positives'],
                'false_positives': overall_stats['false_positives'],
                'missed_opportunities': overall_stats['missed_opportunities'],
                'accuracy_pct': overall_stats['accuracy_pct'],
                'avg_gain_pct': overall_stats.get('avg_gain_pct'),
                'run_status': 'completed',
                'last_run_at': 'NOW()'
            }
            
            data = self._sanitize_dict(data)
            
            self.client.table(self.tables["strategies"]) \
                .update(data) \
                .eq("id", strategy_id) \
                .execute()
            
            self.logger.info(f"Updated strategy {strategy_id} summary")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy summary: {e}")
