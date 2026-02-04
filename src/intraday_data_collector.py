"""
Intraday Data Collector - Captures indicators at specific times of day
- Market open (9:30am NYC)
- Market close (4pm NYC)
- Previous day T-1
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Data sources
import yfinance as yf

from .rate_limiter import RateLimiter
from .utils import get_indicator_mapping

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, UltimateOscillator, AwesomeOscillatorIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class IntradayDataCollector:
    """
    Collects technical indicator data at specific times:
    - Market open (9:30am NYC)
    - Market close (4pm NYC)  
    - Previous day (T-1)
    """
    
    # Parallel processing settings
    MAX_WORKERS = 5
    
    # Lookback period for historical data (days)
    LOOKBACK_DAYS = 90
    
    def __init__(self, config: dict):
        """
        Initialize intraday data collector
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        
        # Indicator mapping
        self.indicator_mapping = get_indicator_mapping(config)
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'cached': 0,
            'yfinance': 0
        }
        
        # Cache for already-fetched historical data
        self.cache = {}
        
        self.logger.info(
            f"Intraday data collector initialized: "
            f"{len(self.indicator_mapping)} indicators, "
            f"{self.MAX_WORKERS} parallel workers"
        )
    
    def collect_intraday_data(
        self,
        winners: List[Dict[str, Any]],
        target_date: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect indicator data at market open, close, and T-1 for the SAME winning stocks
        
        Args:
            winners: List of winner dictionaries with symbol info (the same stocks at all timepoints)
            target_date: Date to collect data for
            
        Returns:
            Dictionary with keys: 'market_open', 'market_close', 'day_prior'
            Each contains list of data dictionaries for the SAME symbols
        """
        self.logger.info(f"Collecting intraday data for {len(winners)} SAME winner stocks on {target_date.date()}...")
        
        self.stats['total'] = len(winners)
        
        # Process winners in parallel
        # NOTE: We collect data for the SAME stocks at three different timepoints:
        # 1. market_open - indicators at 9:30am on target_date
        # 2. market_close - indicators at 4pm on target_date (these are the end-of-day winners)
        # 3. day_prior - indicators at 4pm on target_date - 1 day
        all_data = {
            'market_open': [],
            'market_close': [],
            'day_prior': []
        }
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_winner = {
                executor.submit(self._process_winner, winner, target_date): winner
                for winner in winners
            }
            
            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_winner),
                total=len(winners),
                desc="Collecting intraday indicators"
            ):
                try:
                    result = future.result()
                    if result:
                        all_data['market_open'].append(result['market_open'])
                        all_data['market_close'].append(result['market_close'])
                        all_data['day_prior'].append(result['day_prior'])
                except Exception as e:
                    winner = future_to_winner[future]
                    self.logger.debug(f"Error processing {winner.get('symbol', 'unknown')}: {str(e)}")
                    self.stats['failed'] += 1
        
        self.logger.info(
            f"✓ Collected data - "
            f"Market Open: {len(all_data['market_open'])}, "
            f"Market Close: {len(all_data['market_close'])}, "
            f"Day Prior: {len(all_data['day_prior'])}"
        )
        self.logger.info(
            f"  Stats - Success: {self.stats['success']}, "
            f"Failed: {self.stats['failed']}, "
            f"Cached: {self.stats['cached']}"
        )
        
        return all_data
    
    def _process_winner(
        self,
        winner: Dict,
        target_date: datetime
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Process a single winner and collect indicators at three time points
        
        Args:
            winner: Winner dictionary with symbol info
            target_date: Target date for data collection
            
        Returns:
            Dictionary with 'market_open', 'market_close', 'day_prior' data
        """
        symbol = winner.get('symbol')
        exchange = winner.get('exchange', 'NASDAQ')
        
        # Check cache
        cache_key = f"{symbol}:{target_date.date()}"
        if cache_key in self.cache:
            self.stats['cached'] += 1
            indicators_df = self.cache[cache_key]
        else:
            # Fetch historical data with indicators
            indicators_df = self._fetch_historical_indicators(symbol, target_date.date())
            
            if indicators_df is not None and not indicators_df.empty:
                self.cache[cache_key] = indicators_df
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
                return None
        
        # Convert index to date-only for comparison
        indicators_df_copy = indicators_df.copy()
        indicators_df_copy.index = pd.to_datetime(indicators_df_copy.index).date
        
        target_date_only = target_date.date()
        prior_date = target_date_only - timedelta(days=1)
        
        # Get data for target date (market open and close are same day, different conceptual times)
        if target_date_only in indicators_df_copy.index:
            target_day_data = indicators_df_copy.loc[target_date_only]
        else:
            # Get closest available date
            available_dates = [d for d in indicators_df_copy.index if d <= target_date_only]
            if not available_dates:
                return None
            target_day_data = indicators_df_copy.loc[available_dates[-1]]
        
        # Get data for prior day
        if prior_date in indicators_df_copy.index:
            prior_day_data = indicators_df_copy.loc[prior_date]
        else:
            # Get closest available date
            available_dates = [d for d in indicators_df_copy.index if d <= prior_date]
            if not available_dates:
                return None
            prior_day_data = indicators_df_copy.loc[available_dates[-1]]
        
        # Build result - three snapshots
        result = {
            'market_open': self._build_snapshot(
                symbol, exchange, target_date, 'market_open', target_day_data
            ),
            'market_close': self._build_snapshot(
                symbol, exchange, target_date, 'market_close', target_day_data
            ),
            'day_prior': self._build_snapshot(
                symbol, exchange, target_date, 'day_prior', prior_day_data
            )
        }
        
        return result
    
    def _build_snapshot(
        self,
        symbol: str,
        exchange: str,
        target_date: datetime,
        snapshot_type: str,
        data_series: pd.Series
    ) -> Dict[str, Any]:
        """
        Build a data snapshot from indicator series
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name
            target_date: Target date
            snapshot_type: 'market_open', 'market_close', or 'day_prior'
            data_series: Pandas Series with indicator values
            
        Returns:
            Dictionary with metadata and indicator values
        """
        snapshot = {
            'symbol': symbol,
            'exchange': exchange,
            'detection_date': target_date.date().isoformat(),
            'snapshot_type': snapshot_type
        }
        
        # Add time stamps
        if snapshot_type == 'market_open':
            snapshot['snapshot_time'] = '09:30:00'
        elif snapshot_type == 'market_close':
            snapshot['snapshot_time'] = '16:00:00'
        else:  # day_prior
            snapshot['snapshot_time'] = '16:00:00'
            snapshot['snapshot_date'] = (target_date.date() - timedelta(days=1)).isoformat()
        
        # Add all indicators as columns
        valid_values = False
        for indicator_name in data_series.index:
            value = data_series.get(indicator_name)
            
            # Validate the value
            if pd.notna(value):
                try:
                    float_value = float(value)
                    if np.isfinite(float_value):
                        # Convert to lowercase for consistency
                        snapshot[indicator_name.lower()] = float_value
                        valid_values = True
                    else:
                        snapshot[indicator_name.lower()] = None
                except (ValueError, TypeError):
                    snapshot[indicator_name.lower()] = None
            else:
                snapshot[indicator_name.lower()] = None
        
        return snapshot if valid_values else None
    
    def _fetch_historical_indicators(
        self,
        symbol: str,
        target_date: datetime.date
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data and calculate indicators
        Uses yfinance for reliable historical OHLCV data
        
        Args:
            symbol: Stock symbol
            target_date: Target date
            
        Returns:
            DataFrame with date index and indicator columns
        """
        try:
            # Calculate date range
            end_date = target_date + timedelta(days=1)  # Include target date
            start_date = target_date - timedelta(days=self.LOOKBACK_DAYS)
            
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
