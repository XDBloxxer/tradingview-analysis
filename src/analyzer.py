"""
Analyzer for comparing Spikers vs Grinders across multiple time lags
Reads from Supabase and writes results to both Supabase and Google Sheets (sample only)
FIXED: Statistics now pulled from candidates table (actual event data)
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from src.sheets_writer import SheetsWriter
from src.supabase_client import SupabaseClient


class Analyzer:
    """
    Analyzes technical indicators to identify pre-move patterns that differentiate
    Spikers from Grinders across multiple time lags (T-1, T-3, T-5, T-10, T-30)
    """
    
    def __init__(self, config: dict):
        """
        Initialize analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize Supabase client (primary data store)
        self.supabase = SupabaseClient(config)
        
        # Initialize Google Sheets writer (sample data only)
        self.sheets_writer = SheetsWriter(config)
        
        # Analysis settings
        analysis_config = config.get("analysis", {})
        self.min_samples = analysis_config.get("min_samples", 10)
        self.confidence_level = analysis_config.get("confidence_level", 0.95)
        
        # Time lags to analyze
        self.time_lags = config.get("time_lags", [1, 3, 5, 10, 30])
        
        # Sample size for Google Sheets
        supabase_config = config.get("supabase", {})
        self.sheets_sample_size = supabase_config.get("sheets_sample_size", 50)
        
        self.logger.info(
            f"Analyzer initialized: "
            f"min_samples={self.min_samples}, "
            f"time_lags={self.time_lags}, "
            f"sheets_sample={self.sheets_sample_size}"
        )
    
    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis function that processes all time lags and generates
        comparative statistics for Spikers vs Grinders
        
        Analyzes ALL data from Supabase
        
        Returns:
            Dictionary containing:
            - summary: Summary comparison table (average differences)
            - stats: Overall statistics FROM CANDIDATES TABLE (actual event data)
            - sheets_summary: Summary for sample symbols only (for Google Sheets)
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING ANALYSIS (Reading from Supabase)")
        self.logger.info("=" * 60)
        
        # Read all raw data from Supabase
        all_data = self.supabase.read_all_raw_data_by_lag()
        
        if not all_data:
            self.logger.error("No raw data found in Supabase. Cannot proceed with analysis.")
            return {
                'summary': [],
                'stats': {},
                'sheets_summary': []
            }
        
        # Read candidates to get actual event statistics and sample symbols
        candidates_df = self.supabase.read_candidates()
        
        if candidates_df.empty:
            self.logger.error("No candidates found in Supabase.")
            return {
                'summary': [],
                'stats': {},
                'sheets_summary': []
            }
        
        # Determine sample symbols for sheets
        if len(candidates_df) > self.sheets_sample_size:
            sample_symbols = set(candidates_df.head(self.sheets_sample_size)['symbol'].tolist())
            self.logger.info(f"Will generate separate analysis for {len(sample_symbols)} sample symbols")
        else:
            sample_symbols = None
        
        # Generate summary comparison for each time lag (ALL DATA)
        summary_tables = []
        sheets_summary_tables = []
        
        for time_lag, lag_df in all_data.items():
            if lag_df.empty:
                continue
            
            self.logger.info(f"Analyzing {time_lag}: {len(lag_df)} rows")
            
            # Separate Spikers and Grinders (ALL DATA)
            spikers_df = lag_df[lag_df['event_type'] == 'Spiker'].copy()
            grinders_df = lag_df[lag_df['event_type'] == 'Grinder'].copy()
            
            self.logger.info(f"  {time_lag} - Spikers: {len(spikers_df)}, Grinders: {len(grinders_df)}")
            
            # Generate summary comparison for ALL data
            lag_summary = self._generate_summary_comparison(spikers_df, grinders_df, time_lag)
            summary_tables.extend(lag_summary)
            
            # Generate summary for SAMPLE data (for Google Sheets)
            if sample_symbols:
                sample_spikers = spikers_df[spikers_df['symbol'].isin(sample_symbols)].copy()
                sample_grinders = grinders_df[grinders_df['symbol'].isin(sample_symbols)].copy()
                
                if not sample_spikers.empty or not sample_grinders.empty:
                    self.logger.info(
                        f"  {time_lag} SAMPLE - Spikers: {len(sample_spikers)}, "
                        f"Grinders: {len(sample_grinders)}"
                    )
                    lag_sheets_summary = self._generate_summary_comparison(
                        sample_spikers, sample_grinders, time_lag
                    )
                    sheets_summary_tables.extend(lag_sheets_summary)
        
        # Generate overall statistics from CANDIDATES table (actual event data)
        self.logger.info("Generating statistics from CANDIDATES table (actual event data)...")
        stats = self._generate_statistics_from_candidates(candidates_df)
        
        results = {
            'summary': summary_tables,  # ALL data
            'stats': stats,  # FROM CANDIDATES (actual event metrics)
            'sheets_summary': sheets_summary_tables if sample_symbols else summary_tables  # SAMPLE data
        }
        
        self.logger.info("✓ Analysis completed successfully")
        self.logger.info(f"  - Full analysis: {len(summary_tables)} metrics")
        if sample_symbols:
            self.logger.info(f"  - Sheets sample: {len(sheets_summary_tables)} metrics")
        
        return results
    
    def _generate_summary_comparison(
        self,
        spikers_df: pd.DataFrame,
        grinders_df: pd.DataFrame,
        time_lag: str
    ) -> List[Dict[str, Any]]:
        """
        Generate summary comparison table showing average differences
        between Spikers and Grinders for each indicator
        
        Args:
            spikers_df: Spiker events DataFrame
            grinders_df: Grinder events DataFrame
            time_lag: Time lag identifier (e.g., "T-1")
            
        Returns:
            List of summary dictionaries
        """
        # Get all numeric columns (indicators)
        # Exclude metadata columns
        exclude_cols = {
            'id', 'symbol', 'event_date', 'event_type', 'exchange',
            'date', 'time_lag', 'created_at', 'updated_at'
        }
        
        # Find numeric indicator columns
        all_columns = set(spikers_df.columns) | set(grinders_df.columns)
        indicator_columns = [
            col for col in all_columns
            if col not in exclude_cols
        ]
        
        summary_table = []
        
        for col in sorted(indicator_columns):
            # Skip if column doesn't exist in either dataframe
            if col not in spikers_df.columns and col not in grinders_df.columns:
                continue
            
            # Skip if column is not numeric
            if col in spikers_df.columns and not pd.api.types.is_numeric_dtype(spikers_df[col]):
                continue
            if col in grinders_df.columns and not pd.api.types.is_numeric_dtype(grinders_df[col]):
                continue
            
            # Calculate means
            spiker_mean = spikers_df[col].mean() if col in spikers_df.columns else np.nan
            grinder_mean = grinders_df[col].mean() if col in grinders_df.columns else np.nan
            
            # Skip if both are NaN
            if pd.isna(spiker_mean) and pd.isna(grinder_mean):
                continue
            
            # Calculate difference
            if not pd.isna(spiker_mean) and not pd.isna(grinder_mean):
                difference = spiker_mean - grinder_mean
                ratio = spiker_mean / grinder_mean if grinder_mean != 0 else np.nan
            else:
                difference = np.nan
                ratio = np.nan
            
            summary_table.append({
                'time_lag': time_lag,
                'indicator': col,
                'avg_spikers': spiker_mean,
                'avg_grinders': grinder_mean,
                'difference': difference,
                'ratio': ratio
            })
        
        return summary_table
    
    def _generate_statistics_from_candidates(
        self,
        candidates_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate overall statistics from CANDIDATES table
        This contains the actual event day data (price, change_pct, volume)
        
        Args:
            candidates_df: Candidates DataFrame from Supabase
            
        Returns:
            Dictionary of statistics
        """
        self.logger.info("Generating statistics from candidates table...")
        
        # Separate by event type
        spikers_df = candidates_df[candidates_df['event_type'] == 'Spiker']
        grinders_df = candidates_df[candidates_df['event_type'] == 'Grinder']
        
        stats = {
            'total_events': len(candidates_df),
            'total_spikers': len(spikers_df),
            'total_grinders': len(grinders_df),
            'spiker_ratio': len(spikers_df) / len(candidates_df) if len(candidates_df) > 0 else 0,
            'grinder_ratio': len(grinders_df) / len(candidates_df) if len(candidates_df) > 0 else 0,
        }
        
        # CHANGE PERCENTAGE (from event day)
        if 'change_pct' in candidates_df.columns:
            if len(spikers_df) > 0:
                stats['avg_spiker_change_pct'] = float(spikers_df['change_pct'].mean())
                stats['median_spiker_change_pct'] = float(spikers_df['change_pct'].median())
                stats['min_spiker_change_pct'] = float(spikers_df['change_pct'].min())
                stats['max_spiker_change_pct'] = float(spikers_df['change_pct'].max())
            
            if len(grinders_df) > 0:
                stats['avg_grinder_change_pct'] = float(grinders_df['change_pct'].mean())
                stats['median_grinder_change_pct'] = float(grinders_df['change_pct'].median())
                stats['min_grinder_change_pct'] = float(grinders_df['change_pct'].min())
                stats['max_grinder_change_pct'] = float(grinders_df['change_pct'].max())
        
        # PRICE (from event day)
        if 'price' in candidates_df.columns:
            if len(spikers_df) > 0:
                stats['avg_spiker_price'] = float(spikers_df['price'].mean())
                stats['median_spiker_price'] = float(spikers_df['price'].median())
            
            if len(grinders_df) > 0:
                stats['avg_grinder_price'] = float(grinders_df['price'].mean())
                stats['median_grinder_price'] = float(grinders_df['price'].median())
        
        # VOLUME (from event day)
        if 'volume' in candidates_df.columns:
            if len(spikers_df) > 0:
                stats['avg_spiker_volume'] = float(spikers_df['volume'].mean())
                stats['median_spiker_volume'] = float(spikers_df['volume'].median())
            
            if len(grinders_df) > 0:
                stats['avg_grinder_volume'] = float(grinders_df['volume'].mean())
                stats['median_grinder_volume'] = float(grinders_df['volume'].median())
        
        # EXCHANGE BREAKDOWN
        if 'exchange' in candidates_df.columns:
            exchange_counts = candidates_df['exchange'].value_counts().to_dict()
            for exchange, count in exchange_counts.items():
                stats[f'count_{exchange}'] = int(count)
            
            # By event type and exchange
            for event_type in ['Spiker', 'Grinder']:
                event_df = candidates_df[candidates_df['event_type'] == event_type]
                if not event_df.empty and 'exchange' in event_df.columns:
                    exchange_counts = event_df['exchange'].value_counts().to_dict()
                    for exchange, count in exchange_counts.items():
                        stats[f'count_{event_type.lower()}_{exchange}'] = int(count)
        
        self.logger.info(f"✓ Generated {len(stats)} statistics from candidates")
        
        return stats
    
    def write_to_supabase(self, analysis_results: Dict[str, Any]):
        """
        Write ALL analysis results to Supabase
        
        Args:
            analysis_results: Analysis results dictionary
        """
        self.logger.info("Writing ALL analysis results to Supabase...")
        
        try:
            # Write summary to analysis table (ALL DATA)
            if analysis_results.get('summary'):
                count = self.supabase.write_analysis(analysis_results['summary'])
                self.logger.info(f"✓ Wrote {count} analysis rows to Supabase")
            
            # Write statistics to summary_stats table
            if analysis_results.get('stats'):
                count = self.supabase.write_summary_stats(analysis_results['stats'])
                self.logger.info(f"✓ Wrote {count} summary stats to Supabase")
            
            self.logger.info("✓ All analysis results written to Supabase")
            
        except Exception as e:
            self.logger.error(f"✗ Error writing to Supabase: {str(e)}", exc_info=True)
            raise
    
    def write_to_sheets(self, analysis_results: Dict[str, Any]):
        """
        Write SAMPLE analysis results to Google Sheets for validation
        Uses the sheets_summary (sample symbols only)
        
        Args:
            analysis_results: Analysis results dictionary
        """
        self.logger.info(f"Writing SAMPLE analysis to Google Sheets...")
        
        try:
            # Write sample summary to Analysis sheet
            if analysis_results.get('sheets_summary'):
                summary_df = pd.DataFrame(analysis_results['sheets_summary'])
                
                self.sheets_writer.write_analysis({'summary': summary_df})
                self.logger.info(f"✓ Wrote {len(summary_df)} sample analysis rows to Google Sheets")
            
            # Write all statistics (it's small)
            if analysis_results.get('stats'):
                self.sheets_writer.write_summary_stats(analysis_results['stats'])
                self.logger.info("✓ Wrote statistics to Google Sheets")
            
            self.logger.info("✓ Sample data written to Google Sheets for validation")
            
        except Exception as e:
            self.logger.error(f"✗ Error writing to Google Sheets: {str(e)}", exc_info=True)
            # Don't raise - sheets is just for validation
    
    def export_to_excel(self, analysis_results: Dict[str, Any], output_path: str = "ANALISIS_FINAL.xlsx"):
        """
        Export analysis results to Excel
        
        Args:
            analysis_results: Analysis results dictionary
            output_path: Path to output Excel file
        """
        self.logger.info(f"Exporting analysis to {output_path}...")
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Summary comparison
                if analysis_results.get('summary'):
                    summary_df = pd.DataFrame(analysis_results['summary'])
                    summary_df.to_excel(writer, sheet_name='SUMMARY_COMPARISON', index=False)
                    self.logger.info("✓ Wrote SUMMARY_COMPARISON sheet")
                
                # Sheet 2: Statistics
                if analysis_results.get('stats'):
                    stats_df = pd.DataFrame([
                        {'Metric': k, 'Value': v}
                        for k, v in analysis_results['stats'].items()
                    ])
                    stats_df.to_excel(writer, sheet_name='STATISTICS', index=False)
                    self.logger.info("✓ Wrote STATISTICS sheet")
            
            self.logger.info(f"✓ Excel export completed: {output_path}")
            
        except Exception as e:
            self.logger.error(f"✗ Error exporting to Excel: {str(e)}", exc_info=True)
            raise
