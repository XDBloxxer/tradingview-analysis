"""
Supabase integration for storing analysis data
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from supabase import create_client, Client


class SupabaseClient:
    """
    Handler for writing and reading data to/from Supabase
    """
    
    def __init__(self, config: dict):
        """
        Initialize Supabase client
        
        Args:
            config: Configuration dictionary with supabase settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Get credentials from environment
        supabase_url: ${{ secrets.SUPABASE_URL }}
        supabase_key: ${{ secrets.SUPABASE_KEY }}
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
            )
        
        # Initialize Supabase client
        self.client: Client = create_client(supabase_url, supabase_key)
        
        # Table names
        supabase_config = config.get("supabase", {})
        self.tables = {
            "candidates": supabase_config.get("candidates_table", "candidates"),
            "raw_data": supabase_config.get("raw_data_table", "raw_data"),
            "analysis": supabase_config.get("analysis_table", "analysis"),
            "summary": supabase_config.get("summary_table", "summary_stats")
        }
        
        self.logger.info(f"Connected to Supabase")
    
    def write_candidates(self, candidates: List[Dict[str, Any]]) -> int:
        """
        Write candidate events to Supabase
        
        Args:
            candidates: List of event dictionaries
            
        Returns:
            Number of rows written
        """
        if not candidates:
            self.logger.warning("No candidates to write")
            return 0
        
        try:
            # Upsert data (insert or update if exists)
            response = self.client.table(self.tables["candidates"]).upsert(
                candidates,
                on_conflict="symbol,date"  # Composite unique key
            ).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} candidates to Supabase")
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing candidates: {str(e)}", exc_info=True)
            raise
    
    def write_raw_data(self, raw_data: List[Dict[str, Any]]) -> int:
        """
        Write raw indicator data to Supabase
        
        Args:
            raw_data: List of raw data dictionaries
            
        Returns:
            Number of rows written
        """
        if not raw_data:
            self.logger.warning("No raw data to write")
            return 0
        
        try:
            # Process in batches to avoid timeouts
            batch_size = 1000
            total_written = 0
            
            for i in range(0, len(raw_data), batch_size):
                batch = raw_data[i:i + batch_size]
                
                response = self.client.table(self.tables["raw_data"]).upsert(
                    batch,
                    on_conflict="symbol,event_date,time_lag"
                ).execute()
                
                count = len(response.data) if response.data else 0
                total_written += count
                self.logger.info(f"Wrote batch {i//batch_size + 1}: {count} rows")
            
            self.logger.info(f"Total raw data written: {total_written}")
            return total_written
            
        except Exception as e:
            self.logger.error(f"Error writing raw data: {str(e)}", exc_info=True)
            raise
    
    def write_analysis(self, analysis_data: List[Dict[str, Any]]) -> int:
        """
        Write analysis results to Supabase
        
        Args:
            analysis_data: List of analysis dictionaries
            
        Returns:
            Number of rows written
        """
        if not analysis_data:
            self.logger.warning("No analysis data to write")
            return 0
        
        try:
            response = self.client.table(self.tables["analysis"]).upsert(
                analysis_data,
                on_conflict="indicator,time_lag"
            ).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} analysis rows to Supabase")
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing analysis: {str(e)}", exc_info=True)
            raise
    
    def write_summary_stats(self, stats: Dict[str, Any]) -> int:
        """
        Write summary statistics to Supabase
        
        Args:
            stats: Summary statistics dictionary
            
        Returns:
            Number of rows written
        """
        if not stats:
            self.logger.warning("No summary stats to write")
            return 0
        
        try:
            # Convert to list of dicts for table format
            stats_list = [
                {"metric": k, "value": v}
                for k, v in stats.items()
            ]
            
            response = self.client.table(self.tables["summary"]).upsert(
                stats_list,
                on_conflict="metric"
            ).execute()
            
            count = len(response.data) if response.data else 0
            self.logger.info(f"Wrote {count} summary stats to Supabase")
            return count
            
        except Exception as e:
            self.logger.error(f"Error writing summary stats: {str(e)}", exc_info=True)
            raise
    
    def read_candidates(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Read candidates from Supabase
        
        Args:
            limit: Optional limit on number of rows to return
            
        Returns:
            DataFrame of candidates
        """
        try:
            query = self.client.table(self.tables["candidates"]).select("*")
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading candidates: {str(e)}")
            return pd.DataFrame()
    
    def read_raw_data(
        self,
        time_lag: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read raw data from Supabase
        
        Args:
            time_lag: Optional filter by time lag (e.g., "T-1")
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame of raw data
        """
        try:
            query = self.client.table(self.tables["raw_data"]).select("*")
            
            if time_lag:
                query = query.eq("time_lag", time_lag)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading raw data: {str(e)}")
            return pd.DataFrame()
    
    def read_all_raw_data_by_lag(self) -> Dict[str, pd.DataFrame]:
        """
        Read all raw data grouped by time lag
        
        Returns:
            Dictionary mapping time lag to DataFrame
        """
        try:
            # Get all unique time lags
            response = self.client.table(self.tables["raw_data"]) \
                .select("time_lag") \
                .execute()
            
            if not response.data:
                return {}
            
            time_lags = list(set(row["time_lag"] for row in response.data))
            
            # Read data for each time lag
            all_data = {}
            for lag in time_lags:
                df = self.read_raw_data(time_lag=lag)
                if not df.empty:
                    all_data[lag] = df
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error reading raw data by lag: {str(e)}")
            return {}
    
    def read_analysis(self, time_lag: Optional[str] = None) -> pd.DataFrame:
        """
        Read analysis results from Supabase
        
        Args:
            time_lag: Optional filter by time lag
            
        Returns:
            DataFrame of analysis results
        """
        try:
            query = self.client.table(self.tables["analysis"]).select("*")
            
            if time_lag:
                query = query.eq("time_lag", time_lag)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            return pd.DataFrame(response.data)
            
        except Exception as e:
            self.logger.error(f"Error reading analysis: {str(e)}")
            return pd.DataFrame()
    
    def read_summary_stats(self) -> Dict[str, Any]:
        """
        Read summary statistics from Supabase
        
        Returns:
            Dictionary of statistics
        """
        try:
            response = self.client.table(self.tables["summary"]).select("*").execute()
            
            if not response.data:
                return {}
            
            # Convert list of dicts to single dict
            return {row["metric"]: row["value"] for row in response.data}
            
        except Exception as e:
            self.logger.error(f"Error reading summary stats: {str(e)}")
            return {}
    
    def delete_all_data(self):
        """
        Delete all data from all tables (use with caution!)
        """
        self.logger.warning("Deleting all data from Supabase...")
        
        for table_name in self.tables.values():
            try:
                self.client.table(table_name).delete().neq("id", 0).execute()
                self.logger.info(f"Deleted data from {table_name}")
            except Exception as e:
                self.logger.error(f"Error deleting from {table_name}: {str(e)}")
