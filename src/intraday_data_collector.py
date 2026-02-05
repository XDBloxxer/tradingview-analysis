"""
Intraday Data Collector - FIXED VERSION
Captures indicators at ACTUAL market open (9:30am) and close (4pm) using intraday data
Only includes indicators that exist in database schema
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

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class IntradayDataCollector:
    """
    Collects technical indicator data at specific times using INTRADAY data:
    - Market open (9:30am NYC) - from 5-minute bars
    - Market close (4pm NYC) - from 5-minute bars
    - Previous day (T-1) - from daily bars
    
    Only includes indicators that exist in database schema
    """
    
    # Parallel processing settings
    MAX_WORKERS = 5
    
    # Historical data lookback
    LOOKBACK_DAYS = 90
    
    def __init__(self, config: dict):
        """Initialize intraday data collector"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0
        }
        
        # Cache for already-fetched data
        self.cache = {}
        
        self.logger.info("Intraday data collector initialized (minimal indicators for database schema)")
    
    def collect_intraday_data(
        self,
        winners: List[Dict[str, Any]],
        target_date: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect indicator data at market open, close, and T-1
        
        Args:
            winners: List of winner dictionaries
            target_date: Date to collect data for
            
        Returns:
            Dictionary with 'market_open', 'market_close', 'day_prior'
        """
        self.logger.info(f"Collecting intraday data for {len(winners)} winners on {target_date.date()}...")
        
        self.stats['total'] = len(winners)
        
        all_data = {
            'market_open': [],
            'market_close': [],
            'day_prior': []
        }
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_winner = {
                executor.submit(self._process_winner, winner, target_date): winner
                for winner in winners
            }
            
            for future in tqdm(
                as_completed(future_to_winner),
                total=len(winners),
                desc="Collecting intraday indicators"
            ):
                try:
                    result = future.result()
                    if result:
                        if result['market_open']:
                            all_data['market_open'].append(result['market_open'])
                        if result['market_close']:
                            all_data['market_close'].append(result['market_close'])
                        if result['day_prior']:
                            all_data['day_prior'].append(result['day_prior'])
                except Exception as e:
                    winner = future_to_winner[future]
                    self.logger.debug(f"Error processing {winner.get('symbol', 'unknown')}: {e}")
                    self.stats['failed'] += 1
        
        self.logger.info(
            f"✓ Collected - "
            f"Market Open: {len(all_data['market_open'])}, "
            f"Market Close: {len(all_data['market_close'])}, "
            f"Day Prior: {len(all_data['day_prior'])}"
        )
        
        return all_data
    
    def _process_winner(
        self,
        winner: Dict,
        target_date: datetime
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Process a single winner and collect indicators at three time points"""
        symbol = winner.get('symbol')
        exchange = winner.get('exchange', 'NASDAQ')

        # Skip invalid winners
        if not symbol:
            self.logger.debug(f"Skipping invalid winner (missing symbol): {winner}")
            return None
        
        try:
            # Check if target date is today - affects data availability
            is_today = target_date.date() == datetime.now().date()
            
            # Fetch INTRADAY data for market open/close
            intraday_df = self._fetch_intraday_data(symbol, target_date.date(), is_today)
            
            # Fetch DAILY data for day prior
            daily_df = self._fetch_daily_data(symbol, target_date.date())
            
            if intraday_df is None and daily_df is None:
                self.stats['failed'] += 1
                return None
            
            self.stats['success'] += 1
            
            # Extract market open snapshot (9:30am from intraday data)
            market_open_snapshot = self._extract_market_open(
                intraday_df, symbol, exchange, target_date
            ) if intraday_df is not None else None
            
            # Extract market close snapshot (4pm from intraday data)
            market_close_snapshot = self._extract_market_close(
                intraday_df, symbol, exchange, target_date
            ) if intraday_df is not None else None
            
            # Extract day prior snapshot (from daily data)
            day_prior_snapshot = self._extract_day_prior(
                daily_df, symbol, exchange, target_date
            ) if daily_df is not None else None
            
            return {
                'market_open': market_open_snapshot,
                'market_close': market_close_snapshot,
                'day_prior': day_prior_snapshot
            }
            
        except Exception as e:
            self.logger.debug(f"Error processing {symbol}: {e}")
            return None
    
    def _fetch_intraday_data(
        self,
        symbol: str,
        target_date: datetime.date,
        is_today: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 5-minute intraday data for target date
        """
        cache_key = f"{symbol}:intraday:{target_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            if is_today:
                df = ticker.history(period='1d', interval='5m')
            else:
                start_dt = datetime.combine(target_date, time(9, 0))
                end_dt = datetime.combine(target_date + timedelta(days=1), time(0, 0))
                
                days_ago = (datetime.now().date() - target_date).days
                if days_ago > 60:
                    self.logger.debug(f"Date {target_date} is beyond 60-day intraday limit for {symbol}")
                    return None
                
                df = ticker.history(start=start_dt, end=end_dt, interval='5m')
            
            if df.empty or len(df) < 5:
                self.logger.debug(f"Insufficient intraday data for {symbol}")
                return None
            
            # Calculate MINIMAL indicators
            indicators_df = self._calculate_minimal_indicators(df)
            
            self.cache[cache_key] = indicators_df
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching intraday for {symbol}: {e}")
            return None
    
    def _fetch_daily_data(
        self,
        symbol: str,
        target_date: datetime.date
    ) -> Optional[pd.DataFrame]:
        """Fetch daily data for day prior calculation"""
        cache_key = f"{symbol}:daily:{target_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            end_date = target_date + timedelta(days=1)
            start_date = target_date - timedelta(days=90)
            
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 20:
                return None
            
            # Calculate MINIMAL indicators
            indicators_df = self._calculate_minimal_indicators(df)
            indicators_df.index = pd.to_datetime(indicators_df.index).date
            
            self.cache[cache_key] = indicators_df
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching daily for {symbol}: {e}")
            return None
            
    def _serialize_value(self, v):
        """
        Convert pandas/numpy values to Python types
        while preserving integers for DB insertion.
        Only fixes integer/float casting issue.
        """
        if pd.isna(v) or np.isinf(v):
            return None
    
        # Preserve integer types
        if isinstance(v, (np.integer, int)):
            return int(v)
    
        # Preserve float types
        if isinstance(v, (np.floating, float)):
            return float(v)
    
        return v

    
    
    def _extract_market_open(
        self,
        intraday_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract indicators at market open (9:30-9:45am)"""
        try:
            # Find bars around 9:30am
            morning_bars = intraday_df[
                (intraday_df.index.time >= time(9, 30)) &
                (intraday_df.index.time <= time(9, 45))
            ]
            
            if morning_bars.empty:
                morning_bars = intraday_df[intraday_df.index.time >= time(9, 30)]
                if morning_bars.empty:
                    self.logger.debug(f"No morning bars found for {symbol}")
                    return None
            
            # Use first available bar
            open_data = morning_bars.iloc[0]
            
            snapshot = {
                'symbol': symbol,
                'exchange': exchange,
                'detection_date': target_date.date().isoformat(),
                'snapshot_type': 'market_open',
                'snapshot_time': '09:30:00'
            }
            
            # Add indicators (only basic OHLCV)
            for key, value in open_data.items():
                if pd.notna(value) and not np.isinf(value):
                    try:
                        snapshot[key.lower()] = self._serialize_value(value)
                    except:
                        snapshot[key.lower()] = None
            
            self.logger.debug(f"Extracted market_open for {symbol}: {list(snapshot.keys())}")
            return snapshot
            
        except Exception as e:
            self.logger.debug(f"Error extracting market open for {symbol}: {e}")
            return None
    
    def _extract_market_close(
        self,
        intraday_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract indicators at market close (3:55-4:00pm)"""
        try:
            # Find bars around 4pm
            close_bars = intraday_df[
                (intraday_df.index.time >= time(15, 55)) &
                (intraday_df.index.time <= time(16, 0))
            ]
            
            if close_bars.empty:
                close_bars = intraday_df[intraday_df.index.time <= time(16, 0)]
                if close_bars.empty:
                    self.logger.debug(f"No close bars found for {symbol}")
                    return None
            
            # Use last available bar
            close_data = close_bars.iloc[-1]
            
            snapshot = {
                'symbol': symbol,
                'exchange': exchange,
                'detection_date': target_date.date().isoformat(),
                'snapshot_type': 'market_close',
                'snapshot_time': '16:00:00'
            }
            
            # Add indicators (only basic OHLCV)
            for key, value in close_data.items():
                if pd.notna(value) and not np.isinf(value):
                    try:
                        snapshot[key.lower()] = self._serialize_value(value)
                    except:
                        snapshot[key.lower()] = None
            
            self.logger.debug(f"Extracted market_close for {symbol}: {list(snapshot.keys())}")
            return snapshot
            
        except Exception as e:
            self.logger.debug(f"Error extracting market close for {symbol}: {e}")
            return None
    
    def _extract_day_prior(
        self,
        daily_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract indicators for previous day"""
        try:
            prior_date = target_date.date() - timedelta(days=1)
            
            # Find closest prior date (handles weekends/holidays)
            available_dates = [d for d in daily_df.index if d <= prior_date]
            
            if not available_dates:
                self.logger.debug(f"No prior dates found for {symbol}")
                return None
            
            actual_date = available_dates[-1]
            prior_data = daily_df.loc[actual_date]
            
            snapshot = {
                'symbol': symbol,
                'exchange': exchange,
                'detection_date': target_date.date().isoformat(),
                'snapshot_type': 'day_prior',
                'snapshot_time': '16:00:00',
                'snapshot_date': actual_date.isoformat()
            }
            
            # Add indicators (only basic OHLCV)
            for key, value in prior_data.items():
                if pd.notna(value) and not np.isinf(value):
                    try:
                        snapshot[key.lower()] = self._serialize_value(value)
                    except:
                        snapshot[key.lower()] = None
            
            self.logger.debug(f"Extracted day_prior for {symbol}: {list(snapshot.keys())}")
            return snapshot
            
        except Exception as e:
            self.logger.debug(f"Error extracting day prior for {symbol}: {e}")
            return None
    
    def _calculate_minimal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators matching the EXACT database schema
        Column names must match exactly including dots, brackets, spaces
        """
        result = pd.DataFrame(index=df.index)
        
        # Basic OHLCV
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        # RSI
        try:
            rsi_ind = RSIIndicator(close=df['Close'], window=14)
            result['rsi'] = rsi_ind.rsi()
            result['rsi[1]'] = result['rsi'].shift(1)
        except:
            pass
        
        # Momentum
        try:
            result['mom'] = df['Close'].diff(10)
            result['mom[1]'] = result['mom'].shift(1)
        except:
            pass
        
        # MACD (use dots not underscores)
        try:
            macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            result['macd.macd'] = macd.macd()
            result['macd.signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
        except:
            pass
        
        # Stochastic (use dots)
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            result['stoch.k'] = stoch.stoch()
            result['stoch.d'] = stoch.stoch_signal()
            result['stoch.k[1]'] = result['stoch.k'].shift(1)
            result['stoch.d[1]'] = result['stoch.d'].shift(1)
        except:
            pass
        
        # ADX (with spaces in column names!)
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['adx'] = adx.adx()
            result['adx+di'] = adx.adx_pos()  # Note the space!
            result['adx-di'] = adx.adx_neg()  # Note the space!
        except:
            pass
        
        # Bollinger Bands (use dots)
        try:
            bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
            result['bb.upper'] = bb.bollinger_hband()
            result['bb.lower'] = bb.bollinger_lband()
            result['bb.middle'] = bb.bollinger_mavg()
            result['bb_width'] = (result['bb.upper'] - result['bb.lower']) / result['bb.middle'] * 100
            result['bbpower'] = (df['Close'] - result['bb.lower']) / (result['bb.upper'] - result['bb.lower'])
        except:
            pass
        
        # ATR
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['atr'] = atr.average_true_range()
        except:
            pass
        
        # Williams %R
        try:
            wr = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14)
            result['w.r'] = wr.williams_r()
        except:
            pass
        
        # CCI
        try:
            cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['cci20'] = cci.cci()
        except:
            pass
        
        # Awesome Oscillator
        try:
            ao = AwesomeOscillatorIndicator(high=df['High'], low=df['Low'], window1=5, window2=34)
            result['ao'] = ao.awesome_oscillator()
        except:
            pass
        
        # Ultimate Oscillator
        try:
            uo = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], 
                                   window1=7, window2=14, window3=28)
            result['uo'] = uo.ultimate_oscillator()
        except:
            pass
        
        # EMAs
        for period in [5, 10, 20, 50, 100, 200]:
            try:
                result[f'ema{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            except:
                pass
        
        # SMAs
        for period in [5, 10, 20, 50, 100, 200]:
            try:
                result[f'sma{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
            except:
                pass
        
        # Volume indicators
        try:
            result['volume_sma5'] = result['volume'].rolling(window=5).mean()
            result['volume_sma20'] = result['volume'].rolling(window=20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma20']
        except:
            pass
        
        # Price changes
        for days in [1, 3, 5, 10, 20]:
            try:
                result[f'price_change_{days}d'] = df['Close'].pct_change(days) * 100
            except:
                pass
        
        # Volatility
        try:
            result['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * 100 * np.sqrt(252)
        except:
            pass
        
        # VWAP
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            result['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        except:
            pass
        
        # 52-week high/low
        try:
            result['high_52w'] = df['High'].rolling(window=252, min_periods=1).max()
            result['low_52w'] = df['Low'].rolling(window=252, min_periods=1).min()
            result['price_vs_high_52w'] = (df['Close'] / result['high_52w'] - 1) * 100
            result['price_vs_low_52w'] = (df['Close'] / result['low_52w'] - 1) * 100
        except:
            pass
        
        # Gaps (note the space in column name!)
        try:
            result['gap_%'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
            result['gap_up'] = (result['gap_ %'] > 2).astype(int)
            result['gap_down'] = (result['gap_ %'] < -2).astype(int)
        except:
            pass
        
        # Trend indicators (boolean as integers)
        try:
            result['ema20_above_ema50'] = (result['ema20'] > result['ema50']).astype(int)
            result['ema50_above_ema200'] = (result['ema50'] > result['ema200']).astype(int)
            result['price_above_ema20'] = (df['Close'] > result['ema20']).astype(int)
            result['ema10_above_ema20'] = (result['ema10'] > result['ema20']).astype(int)
        except:
            pass
        
        return result
