"""
Supabase client for Daily Winners tracking
Completely separate tables from the spike/grinder analysis
ONLY writes NEW symbols that don't already exist for the date
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
        Handles type conversions and invalid values
        """
        if value is None:
            return None
        
        if pd.isna(value):
            return None
        
        # Handle numpy integer types
        if isinstance(value, np.integer):
            return int(value)
        
        # Handle numpy floating types
        if isinstance(value, np.floating):
            if np.isinf(value) or np.isnan(value):
                return None
            return float(value)
        
        # Handle numpy bool
        if isinstance(value, np.bool_):
            return bool(value)
        
        # Handle regular Python floats
        if isinstance(value, float):
            if np.isinf(value) or np.isnan(value):
                return None
            return value
        
        return value
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all values in a dictionary"""
        return {k: self._sanitize_value(v) for k, v in data.items()}
    
    def _get_existing_symbols(self, table_name: str, detection_date: str) -> set:
        """
        Get set of symbols that already exist for a date in a table
        
        Args:
            table_name: Name of table to check
            detection_date: Date to check
            
        Returns:
            Set of existing symbols
        """
        try:
            response = self.client.table(table_name) \
                .select("symbol") \
                .eq("detection_date", detection_date) \
                .execute()
            
            if response.data:
                return {row['symbol'] for row in response.data}
            return set()
        except Exception as e:
            self.logger.debug(f"Could not check existing symbols in {table_name}: {e}")
            return set()
    
    def write_winners(self, winners: List[Dict[str, Any]]) -> int:
        """
        Write daily winners to Supabase
        ONLY writes NEW symbols that don't already exist for this date
        
        Args:
            winners: List of winner dictionaries
            
        Returns:
            Number of rows written
        """
        if not winners:
            self.logger.warning("No winners to write")
            return 0
        
        try:
            detection_date = winners[0].get('detection_date')
            
            # Get existing symbols for this date
            existing_symbols = self._get_existing_symbols(self.tables["winners"], detection_date)
            
            if existing_symbols:
                self.logger.info(f"Found {len(existing_symbols)} existing winners for {detection_date}")
            
            # Filter out symbols that already exist
            new_winners = [w for w in winners if w.get('symbol') not in existing_symbols]
            
            skipped_count = len(winners) - len(new_winners)
            if skipped_count > 0:
                self.logger.info(f"Skipping {skipped_count} winners that already exist in database")
            
            if not new_winners:
                self.logger.info("No new winners to write (all already exist)")
                return 0
            
            # Sanitize all data
            sanitized_winners = [self._sanitize_dict(w) for w in new_winners]
            
            # Insert new data only
            response = self.client.table(self.tables["winners"]).insert(
                sanitized_winners
            ).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} NEW winners to Supabase")
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing winners: {str(e)}", exc_info=True)
            raise
    
    def write_intraday_data(self, intraday_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """
        Write intraday indicator data to Supabase
        ONLY writes NEW symbols that don't already exist for this date
        
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
                detection_date = data[0].get('detection_date')
                
                # Get existing symbols for this date
                existing_symbols = self._get_existing_symbols(self.tables[table_key], detection_date)
                
                if existing_symbols:
                    self.logger.info(f"Found {len(existing_symbols)} existing {data_type} records for {detection_date}")
                
                # Filter out symbols that already exist
                new_data = [d for d in data if d.get('symbol') not in existing_symbols]
                
                skipped_count = len(data) - len(new_data)
                if skipped_count > 0:
                    self.logger.info(f"Skipping {skipped_count} {data_type} records that already exist")
                
                if not new_data:
                    self.logger.info(f"No new {data_type} data to write (all already exist)")
                    counts[data_type] = 0
                    continue
                
                # Sanitize all data
                sanitized_data = [self._sanitize_dict(d) for d in new_data]
                
                # Insert new data only
                response = self.client.table(self.tables[table_key]).insert(
                    sanitized_data
                ).execute()
                
                count = len(response.data) if response.data else 0
                counts[data_type] = count
                self.logger.info(f"Wrote {count} NEW rows to {self.tables[table_key]}")
                
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
    
    def check_date_exists(self, detection_date: str) -> bool:
        """
        Check if data already exists for a given date
        
        Args:
            detection_date: Date in ISO format (YYYY-MM-DD)
            
        Returns:
            True if data exists, False otherwise
        """
        try:
            response = self.client.table(self.tables["winners"]) \
                .select("detection_date") \
                .eq("detection_date", detection_date) \
                .limit(1) \
                .execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            self.logger.error(f"Error checking date exists: {str(e)}")
            return False
