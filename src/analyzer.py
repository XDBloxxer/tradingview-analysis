"""
Analyzer for comparing Spikers vs Grinders across multiple time lags
Reads from Raw Data sheets (T-1, T-3, T-5, T-10, T-30) and generates comparative analysis
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from src.sheets_writer import SheetsWriter


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
        
        # Initialize Google Sheets writer
        self.sheets_writer = SheetsWriter(config)
        
        # Analysis settings
        analysis_config = config.get("analysis", {})
        self.min_samples = analysis_config.get("min_samples", 10)
        self.confidence_level = analysis_config.get("confidence_level", 0.95)
        
        # Time lags to analyze
        self.time_lags = config.get("time_lags", [1, 3, 5, 10, 30])
        
        # Sheet names for reading raw data
        sheets_config = config.get("google_sheets", {}).get("sheets", {})
        self.raw_data_sheet_template = sheets_config.get("raw_data", "Raw Data")
        
        self.logger.info(
            f"Analyzer initialized: "
            f"min_samples={self.min_samples}, "
            f"time_lags={self.time_lags}"
        )
    
    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis function that processes all time lags and generates
        comparative statistics for Spikers vs Grinders
        
        Returns:
            Dictionary containing:
            - summary: Summary comparison table (average differences)
            - stats: Overall statistics
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING ANALYSIS")
        self.logger.info("=" * 60)
        
        # Read all raw data sheets
        all_data = self._read_all_raw_data()
        
        if not all_data:
            self.logger.error("No raw data found. Cannot proceed with analysis.")
            return {
                'summary': [],
                'stats': {}
            }
        
        # Generate summary comparison for each time lag
        summary_tables = []
        
        for time_lag, lag_df in all_data.items():
            if lag_df.empty:
                continue
            
            self.logger.info(f"Analyzing {time_lag}: {len(lag_df)} rows")
            
            # Separate Spikers and Grinders
            spikers_df = lag_df[lag_df['Event_Type'] == 'Spiker'].copy()
            grinders_df = lag_df[lag_df['Event_Type'] == 'Grinder'].copy()
            
            self.logger.info(f"  {time_lag} - Spikers: {len(spikers_df)}, Grinders: {len(grinders_df)}")
            
            # Generate summary comparison for this time lag
            lag_summary = self._generate_summary_comparison(spikers_df, grinders_df, time_lag)
            summary_tables.extend(lag_summary)
        
        # Generate overall statistics using first available time lag data
        # (avoid concatenation issues)
        first_lag_df = list(all_data.values())[0] if all_data else pd.DataFrame()
        if not first_lag_df.empty:
            spikers_all = first_lag_df[first_lag_df['Event_Type'] == 'Spiker']
            grinders_all = first_lag_df[first_lag_df['Event_Type'] == 'Grinder']
            stats = self._generate_statistics(spikers_all, grinders_all)
        else:
            stats = {}
        
        results = {
            'summary': summary_tables,
            'stats': stats
        }
        
        self.logger.info("✓ Analysis completed successfully")
        
        return results
    
    def _read_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Read all raw data sheets (Raw Data_T-1, Raw Data_T-3, etc.)
        
        Returns:
            Dictionary mapping time lag to DataFrame
        """
        self.logger.info("Reading raw data sheets...")
        
        all_data = {}
        
        for lag in self.time_lags:
            sheet_name = f"{self.raw_data_sheet_template}_T-{lag}"
            
            try:
                df = self.sheets_writer.read_sheet(sheet_name)
                
                if df is not None and not df.empty:
                    all_data[f"T-{lag}"] = df
                    self.logger.info(f"✓ Read {len(df)} rows from {sheet_name}")
                else:
                    self.logger.warning(f"✗ No data in {sheet_name}")
                    
            except Exception as e:
                self.logger.warning(f"✗ Could not read {sheet_name}: {str(e)}")
                continue
        
        if not all_data:
            self.logger.error("No raw data sheets found!")
        else:
            self.logger.info(f"✓ Successfully read {len(all_data)} time lag datasets")
        
        return all_data
    
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
            'Symbol', 'Event_Date', 'Event_Type', 'Exchange',
            'Date', 'Time_Lag'
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
                'Time_Lag': time_lag,
                'Indicator': col,
                'AVG_Spikers': spiker_mean,
                'AVG_Grinders': grinder_mean,
                'Difference': difference,
                'Ratio': ratio,
                'Spiker_Count': spikers_df[col].count() if col in spikers_df.columns else 0,
                'Grinder_Count': grinders_df[col].count() if col in grinders_df.columns else 0
            })
        
        self.logger.info(f"  ✓ Generated summary for {len(summary_table)} indicators at {time_lag}")
        
        return summary_table
    
    def _generate_statistics(
        self,
        spikers_df: pd.DataFrame,
        grinders_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate overall statistics
        
        Args:
            spikers_df: Spiker events DataFrame
            grinders_df: Grinder events DataFrame
            
        Returns:
            Dictionary of statistics
        """
        self.logger.info("Generating statistics...")
        
        stats = {
            'total_events': len(spikers_df) + len(grinders_df),
            'total_spikers': len(spikers_df),
            'total_grinders': len(grinders_df),
            'spiker_ratio': len(spikers_df) / (len(spikers_df) + len(grinders_df)) if (len(spikers_df) + len(grinders_df)) > 0 else 0,
            'grinder_ratio': len(grinders_df) / (len(spikers_df) + len(grinders_df)) if (len(spikers_df) + len(grinders_df)) > 0 else 0,
        }
        
        # Find numeric columns for statistics (avoid the Exchange concatenation issue)
        numeric_cols = spikers_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to find change percentage columns
        change_cols = [col for col in numeric_cols if 'change' in col.lower() or 'cambio' in col.lower()]
        
        if change_cols and len(change_cols) > 0:
            change_col = change_cols[0]
            if change_col in spikers_df.columns:
                stats['avg_spiker_change_pct'] = float(spikers_df[change_col].mean())
            if change_col in grinders_df.columns:
                stats['avg_grinder_change_pct'] = float(grinders_df[change_col].mean())
        
        # Price statistics
        price_cols = [col for col in numeric_cols if col.lower() in ['price', 'close']]
        if price_cols and len(price_cols) > 0:
            price_col = price_cols[0]
            if price_col in spikers_df.columns:
                stats['avg_spiker_price'] = float(spikers_df[price_col].mean())
                stats['median_spiker_price'] = float(spikers_df[price_col].median())
            if price_col in grinders_df.columns:
                stats['avg_grinder_price'] = float(grinders_df[price_col].mean())
                stats['median_grinder_price'] = float(grinders_df[price_col].median())
        
        # Volume statistics
        volume_cols = [col for col in numeric_cols if col.lower() == 'volume']
        if volume_cols and len(volume_cols) > 0:
            volume_col = volume_cols[0]
            if volume_col in spikers_df.columns:
                stats['avg_spiker_volume'] = float(spikers_df[volume_col].mean())
            if volume_col in grinders_df.columns:
                stats['avg_grinder_volume'] = float(grinders_df[volume_col].mean())
        
        self.logger.info(f"✓ Generated {len(stats)} statistics")
        
        return stats
    
    def write_to_sheets(self, analysis_results: Dict[str, Any]):
        """
        Write analysis results to Google Sheets
        
        Args:
            analysis_results: Analysis results dictionary
        """
        self.logger.info("Writing analysis results to Google Sheets...")
        
        try:
            # Write summary comparison to Analysis sheet
            if analysis_results.get('summary'):
                summary_df = pd.DataFrame(analysis_results['summary'])
                self.sheets_writer.write_analysis({'summary': summary_df})
                self.logger.info(f"✓ Wrote {len(summary_df)} summary rows to Analysis sheet")
            
            # Write statistics to Summary Stats sheet
            if analysis_results.get('stats'):
                self.sheets_writer.write_summary_stats(analysis_results['stats'])
                self.logger.info("✓ Wrote statistics to Summary Stats sheet")
            
            self.logger.info("✓ All analysis results written to Google Sheets")
            
        except Exception as e:
            self.logger.error(f"✗ Error writing to Google Sheets: {str(e)}", exc_info=True)
            raise
    
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
