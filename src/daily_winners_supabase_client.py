"""
Supabase client for Daily Winners tracking
Completely separate tables from the spike/grinder analysis
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from supabase import create_client, Client


class DailyWinnersSupabaseClient:
    """
    Handler for writing and reading daily winners data to/from Supabase
    Uses separate tables from the main spike/grinder analysis
    """
    
    def __init__(self, config: dict):
        """
        Initialize Supabase client for daily winners
        
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
            self.logger.info(f"✓ Connected to Supabase: {supabase_url[:30]}...")
        except Exception as e:
            self.logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise
        
        # Table names for daily winners (separate from spike/grinder tables)
        daily_winners_config = config.get("daily_winners", {})
        self.tables = {
            "winners": daily_winners_config.get("winners_table", "daily_winners"),
            "market_open": daily_winners_config.get("market_open_table", "winners_market_open"),
            "market_close": daily_winners_config.get("market_close_table", "winners_market_close"),
            "day_prior": daily_winners_config.get("day_prior_table", "winners_day_prior")
        }
        
        self.logger.info(f"Using daily winners tables: {', '.join(self.tables.values())}")
    
    def _sanitize_value(self, value: Any) -> Any:
        """
        Sanitize a value for Supabase/PostgreSQL
        """
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
        """
        Sanitize all values in a dictionary
        """
        return {k: self._sanitize_value(v) for k, v in data.items()}
    
    def write_winners(self, winners: List[Dict[str, Any]]) -> int:
        """
        Write daily winners to Supabase
        Appends new data (does not overwrite)
        
        Args:
            winners: List of winner dictionaries
            
        Returns:
            Number of rows written
        """
        if not winners:
            self.logger.warning("No winners to write")
            return 0
        
        try:
            # Sanitize all data
            sanitized_winners = [self._sanitize_dict(w) for w in winners]
            
            # Insert data (append, don't overwrite)
            response = self.client.table(self.tables["winners"]).insert(
                sanitized_winners
            ).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} winners to Supabase")
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing winners: {str(e)}", exc_info=True)
            raise
    
    def write_intraday_data(self, intraday_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """
        Write intraday indicator data to Supabase
        Appends new data (does not overwrite)
        
        Args:
            intraday_data: Dictionary with 'market_open', 'market_close', 'day_prior' keys
            
        Returns:
            Dictionary with counts for each table
        """
        counts = {}
        
        for data_type, table_key in [
            ('market_open', 'market_open'),
            ('market_close', 'market_close'),
            ('day_prior', 'day_prior')
        ]:
            data = intraday_data.get(data_type, [])
            
            if not data:
                self.logger.warning(f"No {data_type} data to write")
                counts[data_type] = 0
                continue
            
            try:
                # Sanitize all data
                sanitized_data = [self._sanitize_dict(d) for d in data]
                
                # Insert data (append, don't overwrite)
                response = self.client.table(self.tables[table_key]).insert(
                    sanitized_data
                ).execute()
                
                count = len(response.data) if response.data else 0
                counts[data_type] = count
                self.logger.info(f"Wrote {count} rows to {self.tables[table_key]}")
                
            except Exception as e:
                self.logger.error(f"Error writing {data_type} data: {str(e)}", exc_info=True)
                counts[data_type] = 0
        
        return counts
    
    def read_winners(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read daily winners from Supabase
        
        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame of winners
        """
        try:
            query = self.client.table(self.tables["winners"]).select("*")
            
            if start_date:
                query = query.gte("detection_date", start_date)
            
            if end_date:
                query = query.lte("detection_date", end_date)
            
            if limit:
                query = query.limit(limit)
            
            # Order by detection date descending
            query = query.order("detection_date", desc=True)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading winners: {str(e)}")
            return pd.DataFrame()
    
    def read_intraday_data(
        self,
        snapshot_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read intraday indicator data from Supabase
        
        Args:
            snapshot_type: 'market_open', 'market_close', or 'day_prior'
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame of intraday data
        """
        try:
            table_name = self.tables.get(snapshot_type)
            if not table_name:
                self.logger.error(f"Invalid snapshot type: {snapshot_type}")
                return pd.DataFrame()
            
            query = self.client.table(table_name).select("*")
            
            if start_date:
                query = query.gte("detection_date", start_date)
            
            if end_date:
                query = query.lte("detection_date", end_date)
            
            if limit:
                query = query.limit(limit)
            
            # Order by detection date descending
            query = query.order("detection_date", desc=True)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading {snapshot_type} data: {str(e)}")
            return pd.DataFrame()
    
    def get_available_dates(self) -> List[str]:
        """
        Get list of all available detection dates
        
        Returns:
            List of date strings (ISO format)
        """
        try:
            response = self.client.table(self.tables["winners"]) \
                .select("detection_date") \
                .execute()
            
            if not response.data:
                return []
            
            dates = sorted(list(set(row["detection_date"] for row in response.data)), reverse=True)
            return dates
            
        except Exception as e:
            self.logger.error(f"Error getting available dates: {str(e)}")
            return []
    
    def get_winners_for_date(self, detection_date: str) -> pd.DataFrame:
        """
        Get all winners for a specific date
        
        Args:
            detection_date: Date in ISO format (YYYY-MM-DD)
            
        Returns:
            DataFrame of winners for that date
        """
        try:
            response = self.client.table(self.tables["winners"]) \
                .select("*") \
                .eq("detection_date", detection_date) \
                .order("change_pct", desc=True) \
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error getting winners for {detection_date}: {str(e)}")
            return pd.DataFrame()
