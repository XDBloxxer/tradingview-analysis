"""
Google Sheets integration for writing analysis data
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np


class SheetsWriter:
    """
    Handler for writing and reading data to/from Google Sheets
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self, config: dict):
        """
        Initialize Google Sheets writer
        
        Args:
            config: Configuration dictionary with google_sheets settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        sheets_config = config.get("google_sheets", {})
        self.spreadsheet_id = sheets_config.get("spreadsheet_id")
        credentials_path = sheets_config.get("credentials_path")
        
        if not self.spreadsheet_id:
            raise ValueError("spreadsheet_id not configured in config.yaml")
        
        if not Path(credentials_path).exists():
            raise FileNotFoundError(
                f"Google Sheets credentials not found: {credentials_path}\n"
                f"Please download service account JSON and place it there."
            )
        
        # Initialize Google Sheets client
        self.client = self._authenticate(credentials_path)
        self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
        
        # Get sheet names from config
        sheet_names = sheets_config.get("sheets", {})
        self.sheet_names = {
            "candidates": sheet_names.get("candidates", "Candidates"),
            "raw_data": sheet_names.get("raw_data", "Raw Data"),
            "analysis": sheet_names.get("analysis", "Analysis"),
            "summary": sheet_names.get("summary", "Summary Stats")
        }
        
        self.logger.info(f"Connected to Google Sheets: {self.spreadsheet_id}")
    
    def _authenticate(self, credentials_path: str) -> gspread.Client:
        """
        Authenticate with Google Sheets API
        
        Args:
            credentials_path: Path to service account JSON
            
        Returns:
            Authenticated gspread client
        """
        credentials = Credentials.from_service_account_file(
            credentials_path,
            scopes=self.SCOPES
        )
        return gspread.authorize(credentials)
    
    def _sanitize_value(self, value):
        """
        Sanitize a value for Google Sheets JSON serialization
        Replaces inf, -inf, and NaN with None
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized value safe for JSON
        """
        if value is None:
            return None
        
        # Handle pandas NA types
        if pd.isna(value):
            return None
        
        # Handle numeric types
        if isinstance(value, (int, float, np.number)):
            # Check for inf or nan
            if np.isinf(value) or np.isnan(value):
                return None
            # Convert numpy types to Python types
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
        
        return value
    
    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize entire DataFrame for Google Sheets
        
        Args:
            df: DataFrame to sanitize
            
        Returns:
            Sanitized DataFrame
        """
        df = df.copy()
        
        # Replace inf and -inf with None
        df = df.replace([np.inf, -np.inf], None)
        
        # Replace NaN with None
        df = df.where(pd.notna(df), None)
        
        return df
    
    def write_candidates(self, candidates: List[Dict[str, Any]]):
        """
        Write candidate events to Candidates sheet
        
        Args:
            candidates: List of event dictionaries
        """
        if not candidates:
            self.logger.warning("No candidates to write")
            return
        
        # Sanitize data before creating DataFrame
        sanitized_candidates = []
        for candidate in candidates:
            sanitized = {k: self._sanitize_value(v) for k, v in candidate.items()}
            sanitized_candidates.append(sanitized)
        
        df = pd.DataFrame(sanitized_candidates)
        df = self._sanitize_dataframe(df)
        
        self._write_dataframe(df, self.sheet_names["candidates"])
        
        self.logger.info(f"Wrote {len(candidates)} candidates to sheet")
    
    def write_raw_data(self, raw_data: List[Dict[str, Any]]):
        """
        Write raw indicator data to Raw Data sheets (one per time lag)
        
        Format: Each row = one stock event, columns = indicators at different time lags
        Example row: Symbol, Event_Date, Event_Type, T-1_RSI, T-1_Williams, T-3_RSI, T-3_Williams, ...
        
        Args:
            raw_data: List of raw data dictionaries
        """
        if not raw_data:
            self.logger.warning("No raw data to write")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        df = self._sanitize_dataframe(df)
        
        # Get time lag columns (T-1, T-3, T-5, etc.)
        time_lag_cols = [col for col in df.columns if col.startswith('T-')]
        
        if not time_lag_cols:
            # No time lag columns, write to single sheet
            self._write_dataframe(df, self.sheet_names["raw_data"])
            self.logger.info(f"Wrote {len(raw_data)} raw data rows to sheet")
            return
        
        # Extract unique time lags
        time_lags = sorted(set(col.split('_')[0] for col in time_lag_cols))
        
        # For each time lag, create a separate sheet with just that time lag's data
        for time_lag in time_lags:
            # Get columns for this time lag
            lag_cols = [col for col in time_lag_cols if col.startswith(f"{time_lag}_")]
            
            # Metadata columns
            metadata_cols = ['Symbol', 'Event_Date', 'Event_Type', 'Exchange']
            available_metadata = [col for col in metadata_cols if col in df.columns]
            
            # Create time-lag specific DataFrame (wide format - one row per symbol)
            lag_df = df[available_metadata + lag_cols].copy()
            
            # Rename columns to remove time lag prefix (T-1_RSI -> RSI)
            rename_dict = {col: col.replace(f"{time_lag}_", "") for col in lag_cols}
            lag_df = lag_df.rename(columns=rename_dict)
            
            # Remove duplicate rows (same symbol/event with multiple indicator rows)
            # Group by metadata and take first occurrence
            lag_df = lag_df.groupby(available_metadata, as_index=False).first()
            
            # Remove rows with all None indicator values
            indicator_cols = [col for col in lag_df.columns if col not in available_metadata]
            lag_df = lag_df.dropna(how='all', subset=indicator_cols)
            
            if not lag_df.empty:
                sheet_name = f"{self.sheet_names['raw_data']}_{time_lag}"
                self._write_dataframe(lag_df, sheet_name)
                self.logger.info(f"Wrote {len(lag_df)} rows to {sheet_name}")
    
    def write_analysis(self, analysis: Dict[str, Any]):
        """
        Write analysis results to Analysis sheet
        
        Args:
            analysis: Analysis results dictionary with 'summary', 'spikers', 'grinders'
        """
        # Write summary table (like SUMMARY_PRE_MOVE)
        if "summary" in analysis:
            if isinstance(analysis["summary"], pd.DataFrame):
                df_summary = analysis["summary"]
            else:
                df_summary = pd.DataFrame(analysis["summary"])
            
            df_summary = self._sanitize_dataframe(df_summary)
            self._write_dataframe(df_summary, self.sheet_names["analysis"])
            self.logger.info("Wrote analysis summary to sheet")
        
        # Write detailed results if present
        if "spikers" in analysis and "grinders" in analysis:
            # Append spikers and grinders as separate sections
            if isinstance(analysis["spikers"], pd.DataFrame):
                df_spikers = analysis["spikers"]
            else:
                df_spikers = pd.DataFrame(analysis["spikers"])
            
            if isinstance(analysis["grinders"], pd.DataFrame):
                df_grinders = analysis["grinders"]
            else:
                df_grinders = pd.DataFrame(analysis["grinders"])
            
            df_spikers = self._sanitize_dataframe(df_spikers)
            df_grinders = self._sanitize_dataframe(df_grinders)
            
            # Get current data to append below summary
            worksheet = self.spreadsheet.worksheet(self.sheet_names["analysis"])
            current_rows = len(worksheet.get_all_values())
            
            # Write spikers
            if not df_spikers.empty:
                self._append_dataframe(
                    df_spikers,
                    self.sheet_names["analysis"],
                    start_row=current_rows + 3,
                    header="SPIKERS DETAIL"
                )
            
            # Write grinders
            if not df_grinders.empty:
                current_rows = len(worksheet.get_all_values())
                self._append_dataframe(
                    df_grinders,
                    self.sheet_names["analysis"],
                    start_row=current_rows + 3,
                    header="GRINDERS DETAIL"
                )
    
    def write_summary_stats(self, stats: Dict[str, Any]):
        """
        Write summary statistics to Summary Stats sheet
        
        Args:
            stats: Summary statistics dictionary
        """
        if not stats:
            self.logger.warning("No summary stats to write")
            return
        
        # Sanitize stats
        sanitized_stats = {k: self._sanitize_value(v) for k, v in stats.items()}
        
        # Convert stats dict to DataFrame format
        df = pd.DataFrame([
            {"Metric": k, "Value": v}
            for k, v in sanitized_stats.items()
        ])
        
        df = self._sanitize_dataframe(df)
        
        self._write_dataframe(df, self.sheet_names["summary"])
        self.logger.info("Wrote summary statistics to sheet")
    
    def _write_dataframe(self, df: pd.DataFrame, sheet_name: str):
        """
        Write DataFrame to sheet (overwrites existing content)
        
        Args:
            df: DataFrame to write
            sheet_name: Name of sheet to write to
        """
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            # Create sheet if it doesn't exist
            worksheet = self.spreadsheet.add_worksheet(
                title=sheet_name,
                rows=1000,
                cols=26
            )
        
        # Clear existing content
        worksheet.clear()
        
        # Convert DataFrame to list of lists with sanitized values
        data = []
        
        # Add headers
        data.append(df.columns.tolist())
        
        # Add rows with sanitized values
        for _, row in df.iterrows():
            sanitized_row = [self._sanitize_value(v) for v in row.values]
            data.append(sanitized_row)
        
        # Write data
        worksheet.update('A1', data, value_input_option='RAW')
        
        # Format header row
        worksheet.format('A1:Z1', {
            "backgroundColor": {"red": 0.2, "green": 0.2, "blue": 0.2},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}
        })
    
    def _append_dataframe(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        start_row: int,
        header: str = None
    ):
        """
        Append DataFrame to sheet at specific row
        
        Args:
            df: DataFrame to append
            sheet_name: Name of sheet
            start_row: Row to start appending (1-indexed)
            header: Optional header text to add before data
        """
        worksheet = self.spreadsheet.worksheet(sheet_name)
        
        # Add header if provided
        if header:
            worksheet.update(f'A{start_row}', [[header]], value_input_option='RAW')
            start_row += 1
        
        # Convert DataFrame to list of lists with sanitized values
        data = []
        data.append(df.columns.tolist())
        
        for _, row in df.iterrows():
            sanitized_row = [self._sanitize_value(v) for v in row.values]
            data.append(sanitized_row)
        
        # Append data
        worksheet.update(f'A{start_row}', data, value_input_option='RAW')
    
    def read_candidates(self) -> pd.DataFrame:
        """
        Read candidates from Google Sheets
        
        Returns:
            DataFrame of candidates
        """
        return self.read_sheet(self.sheet_names["candidates"])
    
    def read_raw_data(self) -> pd.DataFrame:
        """
        Read raw data from Google Sheets
        
        Returns:
            DataFrame of raw data
        """
        return self.read_sheet(self.sheet_names["raw_data"])
    
    def read_sheet(self, sheet_name: str) -> Optional[pd.DataFrame]:
        """
        Read any sheet by name from Google Sheets
        
        Args:
            sheet_name: Name of sheet to read
            
        Returns:
            DataFrame of sheet data, or None if sheet not found
        """
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
            data = worksheet.get_all_records()
            
            if not data:
                self.logger.warning(f"No data found in sheet: {sheet_name}")
                return pd.DataFrame()
            
            return pd.DataFrame(data)
            
        except gspread.exceptions.WorksheetNotFound:
            self.logger.warning(f"Sheet not found: {sheet_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading sheet {sheet_name}: {str(e)}")
            return None
    
    def list_sheets(self) -> List[str]:
        """
        List all sheet names in the spreadsheet
        
        Returns:
            List of sheet names
        """
        try:
            worksheets = self.spreadsheet.worksheets()
            return [ws.title for ws in worksheets]
        except Exception as e:
            self.logger.error(f"Error listing sheets: {str(e)}")
            return []
