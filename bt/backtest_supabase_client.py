"""
Supabase client for Strategy Backtest results
Stores backtest configurations, results, and detailed trades
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client


class BacktestSupabaseClient:
    """
    Handler for writing and reading strategy backtest data to/from Supabase
    """
    
    def __init__(self, config: dict):
        """
        Initialize Supabase client for backtests
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Get credentials from environment
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            self.logger.error(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set."
            )
            raise ValueError(
                "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        
        try:
            # Initialize Supabase client
            self.client: Client = create_client(supabase_url, supabase_key)
            self.logger.info(f"✓ Connected to Supabase for backtests")
        except Exception as e:
            self.logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise
        
        # Table names for backtests
        self.tables = {
            "backtests": "strategy_backtests",
            "trades": "backtest_trades",
            "missed_opps": "backtest_missed_opportunities"
        }
        
        self.logger.info(f"Using backtest tables: {', '.join(self.tables.values())}")
    
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
        
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        
        if isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        
        return value
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all values in a dictionary"""
        return {k: self._sanitize_value(v) for k, v in data.items()}
    
    def write_backtest_results(self, backtest_data: Dict[str, Any]) -> str:
        """
        Write complete backtest results to Supabase
        Returns backtest_id for reference
        
        Args:
            backtest_data: Complete backtest results dictionary
            
        Returns:
            backtest_id
        """
        try:
            # Create backtest record
            backtest_record = {
                'created_at': datetime.now().isoformat(),
                'strategy_name': backtest_data.get('strategy_name', 'Unnamed Strategy'),
                'start_date': backtest_data['date_range']['start'],
                'end_date': backtest_data['date_range']['end'],
                'target_gain_pct': backtest_data['target_gain_pct'],
                'holding_days': backtest_data['holding_days'],
                'strategy_criteria': backtest_data['strategy_criteria'],
                'total_signals': backtest_data['summary']['total_signals'],
                'successful_hits': backtest_data['summary']['successful_hits'],
                'false_positives': backtest_data['summary']['false_positives'],
                'missed_opportunities': backtest_data['summary']['missed_opportunities'],
                'success_rate': backtest_data['summary']['success_rate'],
                'false_positive_rate': backtest_data['summary']['false_positive_rate'],
                'avg_gain_on_hits': backtest_data['summary']['avg_gain_on_hits'],
                'avg_loss_on_misses': backtest_data['summary']['avg_loss_on_misses'],
                'total_return': backtest_data['summary'].get('total_return', 0.0)
            }
            
            backtest_record = self._sanitize_dict(backtest_record)
            
            # Insert backtest record
            response = self.client.table(self.tables["backtests"]).insert(
                backtest_record
            ).execute()
            
            if not response.data or len(response.data) == 0:
                raise Exception("Failed to create backtest record")
            
            backtest_id = response.data[0]['id']
            self.logger.info(f"Created backtest record: {backtest_id}")
            
            # Write detailed trades
            if backtest_data.get('detailed_results'):
                self._write_trades(backtest_id, backtest_data['detailed_results'])
            
            # Write missed opportunities
            if backtest_data.get('missed_opportunities'):
                self._write_missed_opportunities(backtest_id, backtest_data['missed_opportunities'])
            
            return backtest_id
            
        except Exception as e:
            self.logger.error(f"Error writing backtest results: {str(e)}", exc_info=True)
            raise
    
    def _write_trades(self, backtest_id: str, trades: List[Dict[str, Any]]):
        """Write detailed trade records"""
        if not trades:
            return
        
        try:
            # Add backtest_id to each trade
            trade_records = []
            for trade in trades:
                record = {
                    'backtest_id': backtest_id,
                    'trade_date': trade['date'],
                    'symbol': trade['symbol'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'actual_gain_pct': trade['actual_gain_pct'],
                    'hit_target': trade['hit_target'],
                    'indicator_values': trade['indicator_values']
                }
                trade_records.append(self._sanitize_dict(record))
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(trade_records), batch_size):
                batch = trade_records[i:i + batch_size]
                self.client.table(self.tables["trades"]).insert(batch).execute()
            
            self.logger.info(f"Wrote {len(trade_records)} trade records")
            
        except Exception as e:
            self.logger.error(f"Error writing trades: {str(e)}")
    
    def _write_missed_opportunities(self, backtest_id: str, missed: List[Dict[str, Any]]):
        """Write missed opportunity records"""
        if not missed:
            return
        
        try:
            # Add backtest_id to each record
            missed_records = []
            for opp in missed:
                record = {
                    'backtest_id': backtest_id,
                    'trade_date': opp['date'],
                    'symbol': opp['symbol'],
                    'actual_gain_pct': opp['actual_gain_pct']
                }
                missed_records.append(self._sanitize_dict(record))
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(missed_records), batch_size):
                batch = missed_records[i:i + batch_size]
                self.client.table(self.tables["missed_opps"]).insert(batch).execute()
            
            self.logger.info(f"Wrote {len(missed_records)} missed opportunity records")
            
        except Exception as e:
            self.logger.error(f"Error writing missed opportunities: {str(e)}")
    
    def read_backtests(
        self,
        limit: Optional[int] = None,
        order_by: str = 'created_at',
        descending: bool = True
    ) -> pd.DataFrame:
        """
        Read all backtests
        
        Args:
            limit: Optional limit on results
            order_by: Column to sort by
            descending: Sort descending if True
            
        Returns:
            DataFrame of backtests
        """
        try:
            query = self.client.table(self.tables["backtests"]).select("*")
            
            query = query.order(order_by, desc=descending)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading backtests: {str(e)}")
            return pd.DataFrame()
    
    def read_backtest_trades(self, backtest_id: str) -> pd.DataFrame:
        """
        Read all trades for a specific backtest
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            DataFrame of trades
        """
        try:
            response = self.client.table(self.tables["trades"]) \
                .select("*") \
                .eq("backtest_id", backtest_id) \
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading trades: {str(e)}")
            return pd.DataFrame()
    
    def read_missed_opportunities(self, backtest_id: str) -> pd.DataFrame:
        """
        Read missed opportunities for a specific backtest
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            DataFrame of missed opportunities
        """
        try:
            response = self.client.table(self.tables["missed_opps"]) \
                .select("*") \
                .eq("backtest_id", backtest_id) \
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading missed opportunities: {str(e)}")
            return pd.DataFrame()
    
    def delete_backtest(self, backtest_id: str):
        """
        Delete a backtest and all associated data
        
        Args:
            backtest_id: Backtest ID to delete
        """
        try:
            # Delete trades
            self.client.table(self.tables["trades"]) \
                .delete() \
                .eq("backtest_id", backtest_id) \
                .execute()
            
            # Delete missed opportunities
            self.client.table(self.tables["missed_opps"]) \
                .delete() \
                .eq("backtest_id", backtest_id) \
                .execute()
            
            # Delete backtest
            self.client.table(self.tables["backtests"]) \
                .delete() \
                .eq("id", backtest_id) \
                .execute()
            
            self.logger.info(f"Deleted backtest: {backtest_id}")
            
        except Exception as e:
            self.logger.error(f"Error deleting backtest: {str(e)}")
