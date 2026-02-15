"""
Supabase client for Daily Non-Winners tracking
Mirrors the structure of DailyWinnersSupabaseClient but for negative examples
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from supabase import create_client, Client


class DailyNonWinnersSupabaseClient:
    """
    Handler for writing and reading daily non-winners data to/from Supabase
    These are NEGATIVE examples for ML training
    """
    
    def __init__(self, config: dict):
        """
        Initialize Supabase client for daily non-winners
        
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
        
        # Table names for non-winners (parallel to winners tables)
        self.tables = {
            "non_winners": "daily_non_winners",
            "market_open": "non_winners_market_open",
            "market_close": "non_winners_market_close",
            "day_prior_open": "non_winners_day_prior_open",
            "day_prior_close": "non_winners_day_prior_close"
        }
        
        self.logger.info(f"Using non-winners tables: {', '.join(self.tables.values())}")
    
    def _sanitize_value(self, value: Any, field_name: str = None) -> Any:
        """
        Sanitize a value for Supabase/PostgreSQL
        Handles type conversions and invalid values
        
        Args:
            value: Value to sanitize
            field_name: Optional field name for context-aware conversion
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
            
            # Volume fields should be integers (BIGINT in database)
            if field_name and ('volume' in field_name.lower() or 'obv' in field_name.lower()):
                return int(value)
            
            return float(value)
        
        # Handle numpy bool
        if isinstance(value, np.bool_):
            return bool(value)
    
        # Handle regular Python floats
        if isinstance(value, float):
            if np.isinf(value) or np.isnan(value):
                return None
            
            # Volume fields should be integers (BIGINT in database)
            if field_name and ('volume' in field_name.lower() or 'obv' in field_name.lower()):
                return int(value)
            
            # FIX: convert float integers (0.0, 1.0) to int for flag fields
            if value.is_integer():
                return int(value)
        
            return value
        
        return value
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize all values in a dictionary AND remove auto-generated fields
        """
        # Remove fields that PostgreSQL auto-generates
        auto_fields = {'id', 'created_at', 'updated_at'}
        
        sanitized = {}
        for k, v in data.items():
            # Skip auto-generated fields
            if k in auto_fields:
                continue
            # Pass field name to sanitize_value for context-aware conversion
            sanitized[k] = self._sanitize_value(v, field_name=k)
        
        return sanitized
    
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
    
    def write_non_winners(self, non_winners: List[Dict[str, Any]]) -> int:
        """
        Write daily non-winners to Supabase
        ONLY writes NEW symbols that don't already exist for this date
        
        Args:
            non_winners: List of non-winner dictionaries
            
        Returns:
            Number of rows written
        """
        if not non_winners:
            self.logger.warning("No non-winners to write")
            return 0
        
        try:
            detection_date = non_winners[0].get('detection_date')
            
            # Get existing symbols for this date
            existing_symbols = self._get_existing_symbols(self.tables["non_winners"], detection_date)
            
            if existing_symbols:
                self.logger.info(f"Found {len(existing_symbols)} existing non-winners for {detection_date}")
            
            # Filter out symbols that already exist
            new_non_winners = [nw for nw in non_winners if nw.get('symbol') not in existing_symbols]
            
            skipped_count = len(non_winners) - len(new_non_winners)
            if skipped_count > 0:
                self.logger.info(f"Skipping {skipped_count} non-winners that already exist in database")
            
            if not new_non_winners:
                self.logger.info("No new non-winners to write (all already exist)")
                return 0
            
            # Sanitize all data
            sanitized_non_winners = [self._sanitize_dict(nw) for nw in new_non_winners]
            
            # Insert new data only
            response = self.client.table(self.tables["non_winners"]).insert(
                sanitized_non_winners
            ).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} NEW non-winners to Supabase")
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing non-winners: {str(e)}", exc_info=True)
            raise
    
    def write_intraday_data(self, intraday_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """
        Write intraday indicator data to Supabase
        ONLY writes NEW symbols that don't already exist for this date
        
        Args:
            intraday_data: Dictionary with 'market_open', 'market_close', 'day_prior_open', 'day_prior_close' keys
            
        Returns:
            Dictionary with counts for each table
        """
        counts = {}
        
        for data_type, table_key in [
            ('market_open', 'market_open'),
            ('market_close', 'market_close'),
            ('day_prior_open', 'day_prior_open'),
            ('day_prior_close', 'day_prior_close')
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
                
                # Sanitize all data (this removes 'id' and other auto-fields)
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
    
    def read_non_winners(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read daily non-winners from Supabase
        
        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame of non-winners
        """
        try:
            query = self.client.table(self.tables["non_winners"]).select("*")
            
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
            self.logger.error(f"Error reading non-winners: {str(e)}")
            return pd.DataFrame()
    
    def get_available_dates(self) -> List[str]:
        """
        Get list of all available detection dates
        
        Returns:
            List of date strings (ISO format)
        """
        try:
            response = self.client.table(self.tables["non_winners"]) \
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
            response = self.client.table(self.tables["non_winners"]) \
                .select("detection_date") \
                .eq("detection_date", detection_date) \
                .limit(1) \
                .execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            self.logger.error(f"Error checking date exists: {str(e)}")
            return False
    
    def get_day_prior_close_data(
        self,
        detection_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get T-1 close data for non-winners
        
        Args:
            detection_date: Optional date filter
            limit: Maximum rows to return
            
        Returns:
            DataFrame with T-1 close indicators
        """
        try:
            query = self.client.table(self.tables["day_prior_close"]).select("*")
            
            if detection_date:
                query = query.eq("detection_date", detection_date)
            
            query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading day prior close: {e}")
            return pd.DataFrame()
    
    def get_day_prior_open_data(
        self,
        detection_date: str = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get T-1 open data for non-winners
        
        Args:
            detection_date: Optional date filter
            limit: Maximum rows to return
            
        Returns:
            DataFrame with T-1 open indicators
        """
        try:
            query = self.client.table(self.tables["day_prior_open"]).select("*")
            
            if detection_date:
                query = query.eq("detection_date", detection_date)
            
            query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading day prior open: {e}")
            return pd.DataFrame()
