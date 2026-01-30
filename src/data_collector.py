"""
Data collector for fetching technical indicators with PROPER TIME LAGS
Uses yfinance for historical data (proven approach from enhanced_explosive_analyzer.py)
Falls back to TradingView for real-time/current data when needed
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, UltimateOscillator, AwesomeOscillatorIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Data sources
import yfinance as yf

try:
    from tradingview_scraper.symbols.technicals import Indicators
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False
    print("Warning: tradingview-scraper not installed. TradingView data unavailable.")

from .rate_limiter import RateLimiter
from .sheets_writer import SheetsWriter
from .utils import get_indicator_mapping


class DataCollector:
    """
    Collects technical indicator data for candidate events with PROPER historical time lags
    
    Primary method: yfinance (historical OHLCV data)
    Fallback method: TradingView (current values only)
    
    This fixes the issue where all time lags (T-1, T-3, T-5, etc.) were identical
    """
    
    # Parallel processing settings
    MAX_WORKERS = 5
    
    # Lookback period for historical data (days)
    LOOKBACK_DAYS = 90  # 3 months should cover T-30 and more
    
    def __init__(self, config: dict):
        """
        Initialize data collector
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Time lags to collect (T-1, T-3, T-5, etc.)
        self.time_lags = config.get("time_lags", [1, 3, 5, 10, 30])
        
        # Maximum lag determines minimum data needed
        self.max_lag = max(self.time_lags)
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        self.sheets_writer = SheetsWriter(config)
        
        # Indicator mapping
        self.indicator_mapping = get_indicator_mapping(config)
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'cached': 0,
            'yfinance': 0,
            'tradingview': 0
        }
        
        # Cache for already-fetched historical data
        self.cache = {}
        
        self.logger.info(
            f"Data collector initialized: "
            f"{len(self.indicator_mapping)} indicators, "
            f"{self.MAX_WORKERS} parallel workers, "
            f"time lags: {self.time_lags}"
        )
    
    def collect_indicator_data(self) -> List[Dict[str, Any]]:
        """
        Collect indicator data for all candidates with PROPER historical time lags
        Uses parallel processing for speed
        
        Returns:
            List of raw data dictionaries
        """
        self.logger.info("Starting parallel historical indicator collection...")
        
        # Read candidates from Google Sheets
        candidates_df = self.sheets_writer.read_candidates()
        
        if candidates_df.empty:
            self.logger.warning("No candidates found in Google Sheets")
            return []
        
        self.logger.info(f"Found {len(candidates_df)} candidates to process")
        self.stats['total'] = len(candidates_df)
        
        # Convert to list of dicts for processing
        candidates = candidates_df.to_dict('records')
        
        # Process candidates in parallel
        all_raw_data = []
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_candidate = {
                executor.submit(self._process_candidate, candidate): candidate
                for candidate in candidates
            }
            
            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_candidate),
                total=len(candidates),
                desc="Collecting indicators"
            ):
                try:
                    result = future.result()
                    if result:
                        all_raw_data.extend(result)
                except Exception as e:
                    candidate = future_to_candidate[future]
                    self.logger.debug(f"Error processing {candidate.get('Symbol', 'unknown')}: {str(e)}")
                    self.stats['failed'] += 1
        
        self.logger.info(f"✓ Collected {len(all_raw_data)} raw data rows")
        self.logger.info(
            f"  Stats - Success: {self.stats['success']}, "
            f"Failed: {self.stats['failed']}, "
            f"Cached: {self.stats['cached']}, "
            f"yfinance: {self.stats['yfinance']}, "
            f"TradingView: {self.stats['tradingview']}"
        )
        
        return all_raw_data
    
    def _process_candidate(self, candidate: Dict) -> List[Dict[str, Any]]:
        """
        Process a single candidate and collect indicators with proper time lags
        
        Args:
            candidate: Candidate dictionary with Symbol, Date, Event_Type, Exchange
            
        Returns:
            List of indicator data rows
        """
        symbol = candidate['Symbol']
        event_date = pd.to_datetime(candidate['Date']).date()
        event_type = candidate['Event_Type']
        exchange = candidate.get('Exchange', 'NASDAQ')
        
        # Check cache
        cache_key = f"{symbol}:{event_date}"
        if cache_key in self.cache:
            self.stats['cached'] += 1
            indicators_df = self.cache[cache_key]
        else:
            # Fetch historical data with indicators
            indicators_df = self._fetch_historical_indicators(symbol, event_date)
            
            if indicators_df is not None and not indicators_df.empty:
                self.cache[cache_key] = indicators_df
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
                return []
        
        # Build data rows for each indicator with proper time lags
        raw_data = []
        
        # Get all available indicators from the dataframe
        available_indicators = [col for col in indicators_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert index to date-only for comparison (remove timezone and time)
        # This fixes the timezone comparison error
        indicators_df_copy = indicators_df.copy()
        indicators_df_copy.index = pd.to_datetime(indicators_df_copy.index).date
        
        for indicator_name in available_indicators:
            try:
                # Create row with metadata
                data_row = {
                    'Symbol': symbol,
                    'Event_Date': event_date.isoformat(),
                    'Event_Type': event_type,
                    'Exchange': exchange,
                    'Indicator_Name': indicator_name,
                }
                
                # Add time lag columns with ACTUAL historical values
                for lag in self.time_lags:
                    lag_date = event_date - timedelta(days=lag)
                    
                    # Find the closest available date
                    if lag_date in indicators_df_copy.index:
                        value = indicators_df_copy.loc[lag_date, indicator_name]
                    else:
                        # Get closest prior date
                        prior_dates = [d for d in indicators_df_copy.index if d <= lag_date]
                        if len(prior_dates) > 0:
                            value = indicators_df_copy.loc[prior_dates[-1], indicator_name]
                        else:
                            value = np.nan
                    
                    # Validate the value - must be finite and not NaN
                    # This prevents inf, -inf, and NaN from being added
                    if pd.notna(value):
                        try:
                            float_value = float(value)
                            if np.isfinite(float_value):
                                data_row[f"T-{lag}"] = float_value
                            else:
                                # Value is inf or -inf, set to None
                                data_row[f"T-{lag}"] = None
                        except (ValueError, TypeError):
                            # Can't convert to float, set to None
                            data_row[f"T-{lag}"] = None
                    else:
                        data_row[f"T-{lag}"] = None
                
                # Only add row if we have at least one valid (non-None) lag value
                if any(data_row.get(f"T-{lag}") is not None for lag in self.time_lags):
                    raw_data.append(data_row)
                    
            except Exception as e:
                self.logger.debug(f"Error processing indicator {indicator_name} for {symbol}: {e}")
                continue
        
        return raw_data
    
    def _fetch_historical_indicators(self, symbol: str, event_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        Fetch historical data and calculate indicators
        Uses yfinance for reliable historical OHLCV data
        
        Args:
            symbol: Stock symbol
            event_date: Event date
            
        Returns:
            DataFrame with date index and indicator columns
        """
        try:
            # Calculate date range
            end_date = event_date
            start_date = event_date - timedelta(days=self.LOOKBACK_DAYS)
            
            # Fetch historical data from yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 50:
                self.logger.debug(f"Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            self.stats['yfinance'] += 1
            
            # Calculate all technical indicators
            indicators_df = self._calculate_all_indicators(df)
            
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators from OHLCV data
        Based on the proven approach from enhanced_explosive_analyzer.py
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all calculated indicators
        """
        result = pd.DataFrame(index=df.index)
        
        # ===== BASIC PRICE DATA =====
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        # ===== RSI & MOMENTUM =====
        try:
            rsi = RSIIndicator(close=df['Close'], window=14)
            result['RSI'] = rsi.rsi()
            result['RSI[1]'] = result['RSI'].shift(1)
        except Exception as e:
            self.logger.debug(f"Error calculating RSI: {e}")
        
        try:
            result['Mom'] = df['Close'].diff(10)
            result['Mom[1]'] = result['Mom'].shift(1)
        except Exception as e:
            self.logger.debug(f"Error calculating Momentum: {e}")
        
        # ===== MACD =====
        try:
            macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            result['MACD.macd'] = macd.macd()
            result['MACD.signal'] = macd.macd_signal()
            result['MACD_diff'] = macd.macd_diff()
        except Exception as e:
            self.logger.debug(f"Error calculating MACD: {e}")
        
        # ===== STOCHASTIC =====
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            result['Stoch.K'] = stoch.stoch()
            result['Stoch.D'] = stoch.stoch_signal()
            result['Stoch.K[1]'] = result['Stoch.K'].shift(1)
            result['Stoch.D[1]'] = result['Stoch.D'].shift(1)
        except Exception as e:
            self.logger.debug(f"Error calculating Stochastic: {e}")
        
        # ===== ADX (TREND STRENGTH) =====
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['ADX'] = adx.adx()
            result['ADX+DI'] = adx.adx_pos()
            result['ADX-DI'] = adx.adx_neg()
        except Exception as e:
            self.logger.debug(f"Error calculating ADX: {e}")
        
        # ===== CCI (COMMODITY CHANNEL INDEX) =====
        try:
            cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['CCI20'] = cci.cci()
        except Exception as e:
            self.logger.debug(f"Error calculating CCI: {e}")
        
        # ===== AWESOME OSCILLATOR =====
        try:
            ao = AwesomeOscillatorIndicator(high=df['High'], low=df['Low'], window1=5, window2=34)
            result['AO'] = ao.awesome_oscillator()
        except Exception as e:
            self.logger.debug(f"Error calculating AO: {e}")
        
        # ===== WILLIAMS %R =====
        try:
            wr = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14)
            result['W.R'] = wr.williams_r()
        except Exception as e:
            self.logger.debug(f"Error calculating Williams %R: {e}")
        
        # ===== ULTIMATE OSCILLATOR =====
        try:
            uo = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], 
                                   window1=7, window2=14, window3=28)
            result['UO'] = uo.ultimate_oscillator()
        except Exception as e:
            self.logger.debug(f"Error calculating UO: {e}")
        
        # ===== BOLLINGER BANDS =====
        try:
            bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
            result['BB.upper'] = bb.bollinger_hband()
            result['BB.lower'] = bb.bollinger_lband()
            result['BB.middle'] = bb.bollinger_mavg()
            result['BB_width'] = (result['BB.upper'] - result['BB.lower']) / result['BB.middle'] * 100
            result['BBPower'] = (df['Close'] - result['BB.lower']) / (result['BB.upper'] - result['BB.lower'])
        except Exception as e:
            self.logger.debug(f"Error calculating Bollinger Bands: {e}")
        
        # ===== ATR (VOLATILITY) =====
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['ATR'] = atr.average_true_range()
        except Exception as e:
            self.logger.debug(f"Error calculating ATR: {e}")
        
        # ===== MOVING AVERAGES =====
        for period in [5, 10, 20, 50, 100, 200]:
            try:
                result[f'EMA{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            except Exception as e:
                self.logger.debug(f"Error calculating EMA{period}: {e}")
            
            try:
                result[f'SMA{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
            except Exception as e:
                self.logger.debug(f"Error calculating SMA{period}: {e}")
        
        # ===== VOLUME INDICATORS =====
        try:
            result['volume_sma5'] = result['volume'].rolling(window=5).mean()
            result['volume_sma20'] = result['volume'].rolling(window=20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma20']
        except Exception as e:
            self.logger.debug(f"Error calculating volume indicators: {e}")
        
        # ===== PRICE CHANGES =====
        for days in [1, 3, 5, 10, 20]:
            try:
                result[f'price_change_{days}d'] = df['Close'].pct_change(days) * 100
            except Exception as e:
                self.logger.debug(f"Error calculating price_change_{days}d: {e}")
        
        # ===== VOLATILITY =====
        try:
            result['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * 100 * np.sqrt(252)
        except Exception as e:
            self.logger.debug(f"Error calculating volatility: {e}")
        
        # ===== TREND INDICATORS (BOOLEAN) =====
        try:
            result['EMA20_above_EMA50'] = (result['EMA20'] > result['EMA50']).astype(int)
            result['EMA50_above_EMA200'] = (result['EMA50'] > result['EMA200']).astype(int)
            result['price_above_EMA20'] = (df['Close'] > result['EMA20']).astype(int)
            result['EMA10_above_EMA20'] = (result['EMA10'] > result['EMA20']).astype(int)
        except Exception as e:
            self.logger.debug(f"Error calculating trend indicators: {e}")
        
        # ===== 52-WEEK HIGH/LOW =====
        try:
            result['high_52w'] = df['High'].rolling(window=252, min_periods=1).max()
            result['low_52w'] = df['Low'].rolling(window=252, min_periods=1).min()
            result['price_vs_high_52w'] = (df['Close'] / result['high_52w'] - 1) * 100
            result['price_vs_low_52w'] = (df['Close'] / result['low_52w'] - 1) * 100
        except Exception as e:
            self.logger.debug(f"Error calculating 52w high/low: {e}")
        
        # ===== GAPS =====
        try:
            result['gap_%'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
            result['gap_up'] = (result['gap_%'] > 2).astype(int)
            result['gap_down'] = (result['gap_%'] < -2).astype(int)
        except Exception as e:
            self.logger.debug(f"Error calculating gaps: {e}")
        
        # ===== VWAP (VOLUME WEIGHTED AVERAGE PRICE) =====
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            result['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        except Exception as e:
            self.logger.debug(f"Error calculating VWAP: {e}")
        
        return result
    
    def _fetch_tradingview_current(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """
        Fetch CURRENT indicators from TradingView (fallback method)
        Only returns current values, not historical
        
        Args:
            exchange: Exchange name
            symbol: Stock symbol
            
        Returns:
            Dictionary of current indicator values
        """
        if not TRADINGVIEW_AVAILABLE:
            return {}
        
        try:
            time.sleep(0.1)  # Rate limiting
            
            scanner = Indicators()
            results = scanner.scrape(
                exchange=exchange,
                symbol=symbol,
                timeframe='1d',
                allIndicators=True
            )
            
            if results and results.get('status') == 'success':
                self.stats['tradingview'] += 1
                return results.get('data', {})
            
            return {}
            
        except Exception as e:
            self.logger.debug(f"Error fetching TradingView data for {symbol}: {str(e)}")
            return {}
    
    def write_to_sheets(self, raw_data: List[Dict[str, Any]]):
        """
        Write raw data to Google Sheets
        
        Args:
            raw_data: List of raw data dictionaries
        """
        self.sheets_writer.write_raw_data(raw_data)
