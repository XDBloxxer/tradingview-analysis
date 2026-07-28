"""
Supabase client for Daily Winners tracking
Completely separate tables from the spike/grinder analysis
ONLY writes NEW symbols that don't already exist for the date
FIXED: Removes auto-generated fields before insertion
ENHANCED: Supports day_prior_open table
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from supabase import create_client, Client
from postgrest.exceptions import APIError


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
            "day_prior_open": daily_winners_config.get("day_prior_open_table", "winners_day_prior_open"),
            "day_prior_close": daily_winners_config.get("day_prior_close_table", "winners_day_prior_close")
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
        
            # FIX: convert float integers (0.0, 1.0) to int
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
            sanitized[k] = self._sanitize_value(v)
        
        return sanitized
    
    # Matches the PostgREST "schema cache" error PostgREST/Supabase returns
    # (code PGRST204) when a payload contains a column that doesn't exist in
    # the target table, e.g.:
    #   "Could not find the 'ema20_slope' column of 'winners_market_open' in the schema cache"
    _MISSING_COLUMN_RE = re.compile(r"Could not find the '([^']+)' column")

    def _insert_with_schema_retry(self, table_name: str, rows: List[Dict[str, Any]],
                                   use_upsert: bool = False):
        """
        Insert (or upsert) rows into `table_name`, self-healing against
        PGRST204 "column not found in schema cache" errors.

        The DataFrame that produces these rows can grow new derived
        columns (e.g. new indicators) that haven't been added to the
        Supabase table yet. Rather than hard-failing the entire batch
        (which is what was happening for winners_market_open/close and
        winners_day_prior_open/close), drop any column PostgREST reports
        as unknown and retry, so the rest of the (valid) data still gets
        written. A warning is logged listing everything that was dropped
        so the schema drift is visible and can be fixed with a migration.

        When use_upsert=True, rows are upserted on the (symbol, detection_date)
        unique key with ignore_duplicates=True instead of inserted, so callers
        (e.g. backfills) can write without a pre-read existing-symbol check —
        conflicts are resolved server-side at zero egress cost.

        Returns the Supabase response from the (eventually) successful
        insert/upsert.
        """
        rows = [dict(r) for r in rows]  # don't mutate caller's dicts
        dropped_columns: set = set()

        max_attempts = 25  # generous ceiling; one retry per bad column found
        for attempt in range(max_attempts):
            try:
                if use_upsert:
                    response = self.client.table(table_name).upsert(
                        rows,
                        ignore_duplicates=True,
                        on_conflict="symbol,detection_date",
                    ).execute()
                else:
                    response = self.client.table(table_name).insert(rows).execute()
                if dropped_columns:
                    self.logger.warning(
                        f"Inserted into {table_name} after dropping column(s) not "
                        f"present in the DB schema: {sorted(dropped_columns)}. "
                        f"Add these columns to the table (or remove them from the "
                        f"data pipeline) to stop this warning."
                    )
                return response
            except APIError as e:
                # postgrest.exceptions.APIError stores structured fields as
                # attributes (.message/.code), NOT as a dict in .args[0] -
                # .args[0] is just the pre-formatted repr string.
                message = e.message or ""
                code = e.code
                match = self._MISSING_COLUMN_RE.search(message)
                if code == "PGRST204" and match:
                    bad_col = match.group(1)
                    dropped_columns.add(bad_col)
                    for r in rows:
                        r.pop(bad_col, None)
                    self.logger.debug(
                        f"{table_name}: column '{bad_col}' not in DB schema, "
                        f"dropping and retrying insert."
                    )
                    continue
                # Not a recoverable schema-cache error - re-raise as-is
                raise

        raise RuntimeError(
            f"Gave up inserting into {table_name} after {max_attempts} attempts "
            f"dropping unknown columns: {sorted(dropped_columns)}"
        )

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
    
    def write_winners(self, winners: List[Dict[str, Any]], allow_append: bool = False) -> int:
        """
        Write daily winners to Supabase.

        By default (allow_append=False) only NEW symbols that don't already
        exist for this date are inserted, protecting scheduled runs from
        accidentally duplicating data.

        Pass allow_append=True when backfilling/running manually to allow
        adding stocks to a date that already has records in the database —
        this skips the pre-read existing-symbol check and upserts instead,
        at zero extra egress cost.

        Args:
            winners: List of winner dictionaries
            allow_append: If True, skip the existing-symbol filter and write
                             all provided stocks (subject to DB unique constraints).
            
        Returns:
            Number of rows written
        """
        if not winners:
            self.logger.warning("No winners to write")
            return 0
        
        try:
            detection_date = winners[0].get('detection_date')

            if allow_append:
                self.logger.info(
                    f"allow_append=True: skipping duplicate check for {detection_date}, "
                    "writing all winners"
                )
                new_winners = winners
            else:
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

            # When appending/backfilling, use upsert with ignore_duplicates=True
            # so any symbols already in the DB are silently skipped server-side
            # rather than requiring a pre-read existing-symbol check (which costs
            # egress). Normal inserts don't need this because the pre-filter
            # above already removed them.
            response = self._insert_with_schema_retry(
                self.tables["winners"], sanitized_winners, use_upsert=allow_append
            )
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} NEW winners to Supabase")
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing winners: {str(e)}", exc_info=True)
            raise
    
    def write_intraday_data(self, intraday_data: Dict[str, List[Dict[str, Any]]],
                             allow_append: bool = False) -> Dict[str, int]:
        """
        Write intraday indicator data to Supabase.

        By default (allow_append=False) only NEW symbols that don't already
        exist for this date are inserted. Pass allow_append=True when
        backfilling/running manually to allow adding stocks to a date that
        already has records — this skips the pre-read existing-symbol check
        and upserts instead, at zero extra egress cost.
        
        Args:
            intraday_data: Dictionary with 'market_open', 'market_close', 'day_prior_open', 'day_prior_close' keys
            allow_append: If True, skip the existing-symbol filter.
            
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

                if allow_append:
                    self.logger.info(
                        f"allow_append=True: skipping duplicate check for {data_type} "
                        f"on {detection_date}, writing all records"
                    )
                    new_data = data
                else:
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
                
                # Sanitize all data (this now removes 'id' and other auto-fields)
                sanitized_data = [self._sanitize_dict(d) for d in new_data]
                
                # DEBUG: Log first record to verify structure
                if sanitized_data:
                    first_record = sanitized_data[0]
                    self.logger.info(f"DEBUG {data_type} - First record keys: {list(first_record.keys())[:10]}")
                    self.logger.info(f"DEBUG {data_type} - symbol: {first_record.get('symbol')}")
                    self.logger.info(f"DEBUG {data_type} - exchange: {first_record.get('exchange')}")
                    self.logger.info(f"DEBUG {data_type} - detection_date: {first_record.get('detection_date')}")
                
                # When appending/backfilling, use upsert with ignore_duplicates=True
                # so any symbols already in the DB are silently skipped server-side
                # rather than requiring a pre-read existing-symbol check.
                response = self._insert_with_schema_retry(
                    self.tables[table_key], sanitized_data, use_upsert=allow_append
                )
                
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
