"""
Complete analyzer.py with fixes

This version includes:
1. Proper error handling
2. Always returns valid dict with 'analysis' and 'summary' keys
3. Tries to read raw data from sheets
4. Better logging
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class Analyzer:
    """Analyzes stock event data to find patterns"""
    
    def __init__(self):
        """Initialize analyzer"""
        logger.info("Analyzer initialized")
    
    def analyze(self, sheets_writer=None, raw_data=None) -> Dict[str, pd.DataFrame]:
        """
        Analyze raw data to compare Spikers vs Grinders
        
        Args:
            sheets_writer: SheetsWriter instance to read data from sheets (optional)
            raw_data: Pre-loaded raw data DataFrame (optional)
            
        Returns:
            Dict with 'analysis' and 'summary' DataFrames
        """
        logger.info("=" * 70)
        logger.info("STARTING ANALYSIS")
        logger.info("=" * 70)
        
        # Get raw data from various sources
        data_df = None
        
        # Option 1: Use provided raw data
        if raw_data is not None and not raw_data.empty:
            logger.info(f"✓ Using provided raw data: {len(raw_data)} rows")
            data_df = raw_data
        
        # Option 2: Try to read from sheets
        elif sheets_writer is not None:
            logger.info("Attempting to read raw data from Google Sheets...")
            try:
                data_df = sheets_writer.read_raw_data()
                
                if data_df is not None and not data_df.empty:
                    logger.info(f"✓ Read {len(data_df)} rows from sheets")
                else:
                    logger.warning("Raw data from sheets is empty")
                    
            except Exception as e:
                logger.error(f"Error reading from sheets: {e}")
                data_df = None
        
        # Check if we got data
        if data_df is None or data_df.empty:
            logger.error("=" * 70)
            logger.error("✗ NO RAW DATA FOUND")
            logger.error("=" * 70)
            logger.error("Possible reasons:")
            logger.error("  1. Raw Data sheet in Google Sheets is empty")
            logger.error("  2. Raw data collection step was skipped")
            logger.error("  3. Sheet name mismatch (should be 'Raw Data')")
            logger.error("")
            logger.error("Solution: Run with --collect-raw or --all first")
            
            # Return empty results with error status
            return {
                'analysis': pd.DataFrame(),
                'summary': pd.DataFrame([
                    {'Metric': 'Status', 'Value': 'Failed'},
                    {'Metric': 'Error', 'Value': 'No raw data found'}
                ])
            }
        
        logger.info(f"Starting analysis with {len(data_df)} events")
        logger.info(f"  Columns: {len(data_df.columns)}")
        
        # Validate required columns
        if 'Event_Type' not in data_df.columns:
            logger.error("✗ Raw data missing 'Event_Type' column")
            logger.error(f"Available columns: {list(data_df.columns)[:10]}...")
            
            return {
                'analysis': pd.DataFrame(),
                'summary': pd.DataFrame([
                    {'Metric': 'Status', 'Value': 'Failed'},
                    {'Metric': 'Error', 'Value': 'Missing Event_Type column'}
                ])
            }
        
        # Get indicator columns (exclude metadata)
        metadata_cols = ['Symbol', 'Event_Date', 'Event_Type', 'Exchange']
        indicator_cols = [col for col in data_df.columns if col not in metadata_cols]
        
        if not indicator_cols:
            logger.error("✗ No indicator columns found in raw data")
            logger.error(f"All columns: {list(data_df.columns)}")
            
            return {
                'analysis': pd.DataFrame(),
                'summary': pd.DataFrame([
                    {'Metric': 'Status', 'Value': 'Failed'},
                    {'Metric': 'Error', 'Value': 'No indicators found'}
                ])
            }
        
        logger.info(f"Found {len(indicator_cols)} indicators to analyze")
        
        # Split by event type
        spikers = data_df[data_df['Event_Type'] == 'Spiker']
        grinders = data_df[data_df['Event_Type'] == 'Grinder']
        
        logger.info(f"  Spikers: {len(spikers)} events")
        logger.info(f"  Grinders: {len(grinders)} events")
        
        if len(spikers) == 0:
            logger.warning("⚠ No Spiker events found")
        
        if len(grinders) == 0:
            logger.warning("⚠ No Grinder events found")
        
        if len(spikers) == 0 or len(grinders) == 0:
            logger.error("✗ Need both Spikers and Grinders for comparison")
            
            return {
                'analysis': pd.DataFrame(),
                'summary': pd.DataFrame([
                    {'Metric': 'Status', 'Value': 'Failed'},
                    {'Metric': 'Error', 'Value': 'Missing Spikers or Grinders'}
                ])
            }
        
        # Calculate analysis for each indicator
        logger.info("Calculating averages for each indicator...")
        analysis_data = []
        
        for i, indicator in enumerate(indicator_cols):
            try:
                # Calculate averages
                avg_spiker = spikers[indicator].mean()
                avg_grinder = grinders[indicator].mean()
                
                # Only include if both are valid numbers
                if pd.notna(avg_spiker) and pd.notna(avg_grinder):
                    difference = avg_spiker - avg_grinder
                    
                    analysis_data.append({
                        'Indicator': indicator,
                        'AVG_SPIKERS': round(avg_spiker, 4),
                        'AVG_GRINDERS': round(avg_grinder, 4),
                        'Difference': round(difference, 4)
                    })
                    
            except Exception as e:
                logger.debug(f"Could not analyze {indicator}: {e}")
                continue
            
            # Progress logging
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(indicator_cols)} indicators...")
        
        if not analysis_data:
            logger.error("✗ No analysis data generated")
            logger.error("All indicators may have invalid values")
            
            return {
                'analysis': pd.DataFrame(),
                'summary': pd.DataFrame([
                    {'Metric': 'Status', 'Value': 'Failed'},
                    {'Metric': 'Error', 'Value': 'No valid analysis results'}
                ])
            }
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame(analysis_data)
        
        # Sort by absolute difference (most predictive indicators first)
        analysis_df['Abs_Difference'] = analysis_df['Difference'].abs()
        analysis_df = analysis_df.sort_values('Abs_Difference', ascending=False)
        analysis_df = analysis_df.drop('Abs_Difference', axis=1)
        
        logger.info(f"✓ Generated analysis for {len(analysis_df)} indicators")
        
        # Generate summary statistics
        summary_data = [
            {'Metric': 'Status', 'Value': 'Success'},
            {'Metric': 'Total_Indicators_Analyzed', 'Value': len(analysis_df)},
            {'Metric': 'Total_Events', 'Value': len(data_df)},
            {'Metric': 'Total_Spikers', 'Value': len(spikers)},
            {'Metric': 'Total_Grinders', 'Value': len(grinders)},
        ]
        
        # Top predictive indicators
        if len(analysis_df) > 0:
            top_indicator = analysis_df.iloc[0]
            summary_data.extend([
                {'Metric': 'Top_Indicator', 'Value': top_indicator['Indicator']},
                {'Metric': 'Top_Indicator_Difference', 'Value': round(top_indicator['Difference'], 4)},
            ])
        
        summary_df = pd.DataFrame(summary_data)
        
        logger.info("=" * 70)
        logger.info("✓ ANALYSIS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Analyzed: {len(analysis_df)} indicators")
        logger.info(f"  Events: {len(spikers)} Spikers, {len(grinders)} Grinders")
        
        if len(analysis_df) > 0:
            logger.info(f"  Top indicator: {top_indicator['Indicator']}")
            logger.info(f"    Difference: {top_indicator['Difference']:.4f}")
        
        return {
            'analysis': analysis_df,
            'summary': summary_df
        }
