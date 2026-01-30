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
        self.compute_correlations = analysis_config.get("compute_correlations", True)
        
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
            - summary: Summary comparison table (like SUMMARY_PRE_MOVE)
            - spikers: Detailed spiker data
            - grinders: Detailed grinder data
            - stats: Overall statistics
            - correlations: Correlation analysis (if enabled)
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
                'spikers': [],
                'grinders': [],
                'stats': {},
                'correlations': {}
            }
        
        # Combine all time lag data
        combined_df = self._combine_time_lag_data(all_data)
        
        if combined_df.empty:
            self.logger.error("Combined data is empty. Cannot proceed with analysis.")
            return {
                'summary': [],
                'spikers': [],
                'grinders': [],
                'stats': {},
                'correlations': {}
            }
        
        self.logger.info(f"Combined dataset: {len(combined_df)} rows")
        
        # Separate Spikers and Grinders
        spikers_df = combined_df[combined_df['Event_Type'] == 'Spiker'].copy()
        grinders_df = combined_df[combined_df['Event_Type'] == 'Grinder'].copy()
        
        self.logger.info(f"Spikers: {len(spikers_df)}, Grinders: {len(grinders_df)}")
        
        # Generate summary comparison (like SUMMARY_PRE_MOVE)
        summary_table = self._generate_summary_comparison(spikers_df, grinders_df)
        
        # Generate detailed statistics
        stats = self._generate_statistics(spikers_df, grinders_df)
        
        # Generate correlations if enabled
        correlations = {}
        if self.compute_correlations:
            correlations = self._generate_correlations(spikers_df, grinders_df)
        
        # Prepare detailed data for export
        spikers_detail = self._prepare_detailed_export(spikers_df)
        grinders_detail = self._prepare_detailed_export(grinders_df)
        
        results = {
            'summary': summary_table,
            'spikers': spikers_detail,
            'grinders': grinders_detail,
            'stats': stats,
            'correlations': correlations
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
                    # Add time lag identifier
                    df['Time_Lag'] = f"T-{lag}"
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
    
    def _combine_time_lag_data(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine data from all time lags into a single DataFrame
        
        For analysis, we'll focus on T-1 (most recent) as the primary comparison,
        but keep all time lags for comprehensive analysis
        
        Args:
            all_data: Dictionary of time lag DataFrames
            
        Returns:
            Combined DataFrame
        """
        if not all_data:
            return pd.DataFrame()
        
        # Use T-1 as primary dataset (most predictive)
        if "T-1" in all_data:
            primary_df = all_data["T-1"].copy()
            self.logger.info(f"Using T-1 as primary dataset: {len(primary_df)} rows")
        else:
            # Fallback to first available
            primary_df = list(all_data.values())[0].copy()
            self.logger.warning("T-1 not found, using first available dataset")
        
        return primary_df
    
    def _generate_summary_comparison(
        self,
        spikers_df: pd.DataFrame,
        grinders_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Generate summary comparison table (like SUMMARY_PRE_MOVE)
        Shows average indicator values for Spikers vs Grinders
        
        Args:
            spikers_df: Spiker events DataFrame
            grinders_df: Grinder events DataFrame
            
        Returns:
            List of summary dictionaries
        """
        self.logger.info("Generating summary comparison table...")
        
        # Get all numeric columns (indicators)
        # Exclude metadata columns
        exclude_cols = {
            'Symbol', 'Event_Date', 'Event_Type', 'Exchange',
            'Indicator_Name', 'Time_Lag', 'Fecha_Detectada', 'Date',
            'Price', 'Cambio_Pct', 'Change_%', 'Volume'
        }
        
        # Find all T-1, T-3, etc. columns
        all_columns = set(spikers_df.columns) | set(grinders_df.columns)
        indicator_columns = [
            col for col in all_columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(spikers_df.get(col, grinders_df.get(col)))
        ]
        
        summary_table = []
        
        for col in sorted(indicator_columns):
            # Calculate means
            spiker_mean = spikers_df[col].mean() if col in spikers_df.columns else np.nan
            grinder_mean = grinders_df[col].mean() if col in grinders_df.columns else np.nan
            
            # Skip if both are NaN
            if pd.isna(spiker_mean) and pd.isna(grinder_mean):
                continue
            
            summary_table.append({
                'Indicator': col,
                'AVG_SPIKERS': spiker_mean,
                'AVG_GRINDERS': grinder_mean,
                'Difference': spiker_mean - grinder_mean if not pd.isna(spiker_mean) and not pd.isna(grinder_mean) else np.nan,
                'Ratio': spiker_mean / grinder_mean if not pd.isna(spiker_mean) and not pd.isna(grinder_mean) and grinder_mean != 0 else np.nan
            })
        
        self.logger.info(f"✓ Generated summary for {len(summary_table)} indicators")
        
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
        
        # Average change percentages
        if 'Cambio_Pct' in spikers_df.columns:
            stats['avg_spiker_change_pct'] = spikers_df['Cambio_Pct'].mean()
        elif 'Change_%' in spikers_df.columns:
            stats['avg_spiker_change_pct'] = spikers_df['Change_%'].mean()
        
        if 'Cambio_Pct' in grinders_df.columns:
            stats['avg_grinder_change_pct'] = grinders_df['Cambio_Pct'].mean()
        elif 'Change_%' in grinders_df.columns:
            stats['avg_grinder_change_pct'] = grinders_df['Change_%'].mean()
        
        # Price statistics
        if 'Price' in spikers_df.columns:
            stats['avg_spiker_price'] = spikers_df['Price'].mean()
            stats['median_spiker_price'] = spikers_df['Price'].median()
        
        if 'Price' in grinders_df.columns:
            stats['avg_grinder_price'] = grinders_df['Price'].mean()
            stats['median_grinder_price'] = grinders_df['Price'].median()
        
        # Volume statistics
        if 'Volume' in spikers_df.columns:
            stats['avg_spiker_volume'] = spikers_df['Volume'].mean()
        
        if 'Volume' in grinders_df.columns:
            stats['avg_grinder_volume'] = grinders_df['Volume'].mean()
        
        self.logger.info(f"✓ Generated {len(stats)} statistics")
        
        return stats
    
    def _generate_correlations(
        self,
        spikers_df: pd.DataFrame,
        grinders_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate correlation analysis between indicators and event outcomes
        
        Args:
            spikers_df: Spiker events DataFrame
            grinders_df: Grinder events DataFrame
            
        Returns:
            Dictionary of correlation data
        """
        self.logger.info("Generating correlation analysis...")
        
        correlations = {
            'spikers': {},
            'grinders': {}
        }
        
        # Get numeric columns
        exclude_cols = {
            'Symbol', 'Event_Date', 'Event_Type', 'Exchange',
            'Indicator_Name', 'Time_Lag', 'Fecha_Detectada', 'Date'
        }
        
        # Analyze Spiker correlations
        if 'Cambio_Pct' in spikers_df.columns or 'Change_%' in spikers_df.columns:
            change_col = 'Cambio_Pct' if 'Cambio_Pct' in spikers_df.columns else 'Change_%'
            numeric_cols = [
                col for col in spikers_df.columns
                if col not in exclude_cols
                and col != change_col
                and pd.api.types.is_numeric_dtype(spikers_df[col])
            ]
            
            for col in numeric_cols:
                try:
                    corr = spikers_df[[col, change_col]].corr().iloc[0, 1]
                    if not pd.isna(corr):
                        correlations['spikers'][col] = float(corr)
                except:
                    continue
        
        # Analyze Grinder correlations
        if 'Cambio_Pct' in grinders_df.columns or 'Change_%' in grinders_df.columns:
            change_col = 'Cambio_Pct' if 'Cambio_Pct' in grinders_df.columns else 'Change_%'
            numeric_cols = [
                col for col in grinders_df.columns
                if col not in exclude_cols
                and col != change_col
                and pd.api.types.is_numeric_dtype(grinders_df[col])
            ]
            
            for col in numeric_cols:
                try:
                    corr = grinders_df[[col, change_col]].corr().iloc[0, 1]
                    if not pd.isna(corr):
                        correlations['grinders'][col] = float(corr)
                except:
                    continue
        
        self.logger.info(
            f"✓ Generated correlations: "
            f"{len(correlations['spikers'])} for spikers, "
            f"{len(correlations['grinders'])} for grinders"
        )
        
        return correlations
    
    def _prepare_detailed_export(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare detailed data for export (for SPIKERS/GRINDERS sheets)
        
        Args:
            df: DataFrame to export
            
        Returns:
            List of dictionaries ready for export
        """
        if df.empty:
            return []
        
        # Convert to list of dicts, handling NaN and inf values
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, value in row.items():
                # Sanitize value
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    if np.isinf(value):
                        record[col] = None
                    else:
                        record[col] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    record[col] = value
            records.append(record)
        
        return records
    
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
            
            # Optionally write detailed Spiker/Grinder data
            if analysis_results.get('spikers') and analysis_results.get('grinders'):
                detailed_analysis = {
                    'summary': pd.DataFrame(analysis_results['summary']),
                    'spikers': analysis_results['spikers'],
                    'grinders': analysis_results['grinders']
                }
                self.sheets_writer.write_analysis(detailed_analysis)
                self.logger.info(
                    f"✓ Wrote detailed data: "
                    f"{len(analysis_results['spikers'])} spikers, "
                    f"{len(analysis_results['grinders'])} grinders"
                )
            
            self.logger.info("✓ All analysis results written to Google Sheets")
            
        except Exception as e:
            self.logger.error(f"✗ Error writing to Google Sheets: {str(e)}", exc_info=True)
            raise
    
    def export_to_excel(self, analysis_results: Dict[str, Any], output_path: str = "ANALISIS_FINAL.xlsx"):
        """
        Export analysis results to Excel (similar to phase_tree.py output)
        
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
                    summary_df.to_excel(writer, sheet_name='SUMMARY_PRE_MOVE', index=False)
                    self.logger.info("✓ Wrote SUMMARY_PRE_MOVE sheet")
                
                # Sheet 2: Spikers detail
                if analysis_results.get('spikers'):
                    spikers_df = pd.DataFrame(analysis_results['spikers'])
                    spikers_df.to_excel(writer, sheet_name='SPIKERS', index=False)
                    self.logger.info(f"✓ Wrote SPIKERS sheet ({len(spikers_df)} rows)")
                
                # Sheet 3: Grinders detail
                if analysis_results.get('grinders'):
                    grinders_df = pd.DataFrame(analysis_results['grinders'])
                    grinders_df.to_excel(writer, sheet_name='GRINDERS', index=False)
                    self.logger.info(f"✓ Wrote GRINDERS sheet ({len(grinders_df)} rows)")
                
                # Sheet 4: Statistics
                if analysis_results.get('stats'):
                    stats_df = pd.DataFrame([
                        {'Metric': k, 'Value': v}
                        for k, v in analysis_results['stats'].items()
                    ])
                    stats_df.to_excel(writer, sheet_name='STATISTICS', index=False)
                    self.logger.info("✓ Wrote STATISTICS sheet")
                
                # Sheet 5: Correlations (if available)
                if analysis_results.get('correlations'):
                    corr_data = []
                    for event_type in ['spikers', 'grinders']:
                        if event_type in analysis_results['correlations']:
                            for indicator, corr_value in analysis_results['correlations'][event_type].items():
                                corr_data.append({
                                    'Event_Type': event_type.capitalize(),
                                    'Indicator': indicator,
                                    'Correlation': corr_value
                                })
                    
                    if corr_data:
                        corr_df = pd.DataFrame(corr_data)
                        corr_df.to_excel(writer, sheet_name='CORRELATIONS', index=False)
                        self.logger.info("✓ Wrote CORRELATIONS sheet")
            
            self.logger.info(f"✓ Excel export completed: {output_path}")
            
        except Exception as e:
            self.logger.error(f"✗ Error exporting to Excel: {str(e)}", exc_info=True)
            raise
