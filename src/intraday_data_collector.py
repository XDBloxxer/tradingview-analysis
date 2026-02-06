"""
Intraday Data Collector - FIXED VERSION
Captures indicators at ACTUAL market open (9:30am) and close (4pm) using intraday data
FIXED: Fetches sufficient historical data for indicator calculations
FIXED: Handles the 60-day intraday limit by falling back to daily data for longer periods
FIXED: Always creates market_close rows even when market hasn't closed (with nulls)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from zoneinfo import ZoneInfo


# Data sources
import yfinance as yf

from .rate_limiter import RateLimiter

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, AwesomeOscillatorIndicator, UltimateOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
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
        
        self.logger.info("Intraday data collector initialized")
    
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
            # Fetch historical DAILY data for indicator calculation (need 200+ days)
            daily_df = self._fetch_daily_data_extended(symbol, target_date.date())
            
            if daily_df is None:
                self.stats['failed'] += 1
                return None
            
            # Fetch TODAY's intraday data to get specific timestamps
            intraday_df = self._fetch_intraday_data_today(symbol, target_date)
            
            self.stats['success'] += 1
            
            # Extract market open snapshot
            market_open_snapshot = self._extract_market_open(
                intraday_df, daily_df, symbol, exchange, target_date
            )
            
            # Extract market close snapshot (ALWAYS returns a dict, even if market not closed)
            market_close_snapshot = self._extract_market_close(
                intraday_df, daily_df, symbol, exchange, target_date
            )
            
            # Extract day prior snapshot (from daily data)
            day_prior_snapshot = self._extract_day_prior(
                daily_df, symbol, exchange, target_date
            )
            
            return {
                'market_open': market_open_snapshot,
                'market_close': market_close_snapshot,
                'day_prior': day_prior_snapshot
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            return None
    
    def _fetch_intraday_data_today(
        self,
        symbol: str,
        target_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch TODAY's 5-minute intraday data just to get current price at specific times
        """
        target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
        cache_key = f"{symbol}:intraday:{target_date_obj}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            market_tz = ZoneInfo("America/New_York")
            now_et = datetime.now(market_tz)
            is_today = target_date_obj == now_et.date()
            
            ticker = yf.Ticker(symbol)
            
            if is_today:
                self.logger.debug(f"Fetching today's intraday data for {symbol}")
                df = ticker.history(period='1d', interval='5m')
            else:
                # Historical - within 60 days
                days_ago = (datetime.now().date() - target_date_obj).days
                if days_ago > 60:
                    self.logger.debug(f"Date {target_date_obj} is beyond 60-day intraday limit for {symbol}")
                    return None
                
                start_dt = datetime.combine(target_date_obj, time(9, 0))
                end_dt = datetime.combine(target_date_obj + timedelta(days=1), time(17, 0))
                
                self.logger.debug(f"Fetching historical intraday data for {symbol} on {target_date_obj}")
                df = ticker.history(start=start_dt, end=end_dt, interval='5m')
            
            if df.empty:
                self.logger.debug(f"No intraday data for {symbol}")
                return None
            
            # Normalize timezone
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')
            
            # Keep only bars from target date
            df = df[df.index.date == target_date_obj]
            
            if df.empty:
                self.logger.debug(f"No bars for target date {target_date_obj} for {symbol}")
                return None
            
            self.logger.debug(f"Fetched {len(df)} intraday bars for {symbol}")
            
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            self.logger.debug(f"Error fetching intraday for {symbol}: {e}")
            return None
    
    def _fetch_daily_data_extended(
        self,
        symbol: str,
        target_date: datetime.date
    ) -> Optional[pd.DataFrame]:
        """Fetch 1+ year of daily data for indicator calculation"""
        cache_key = f"{symbol}:daily_extended:{target_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch enough data for 200-day indicators
            end_date = target_date + timedelta(days=1)
            start_date = target_date - timedelta(days=400)  # Extra buffer
            
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 50:
                self.logger.debug(f"Insufficient daily data for {symbol}")
                return None
            
            # Calculate indicators on daily data
            indicators_df = self._calculate_minimal_indicators(df)
            indicators_df.index = pd.to_datetime(indicators_df.index).date
            
            self.logger.debug(f"Fetched {len(indicators_df)} days of data for {symbol}")
            
            self.cache[cache_key] = indicators_df
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching daily extended for {symbol}: {e}")
            return None
            
    def _serialize_value(self, v):
        """Convert pandas/numpy values to Python types"""
        if pd.isna(v) or (isinstance(v, (float, np.floating)) and np.isinf(v)):
            return None
    
        if isinstance(v, (np.integer, int)):
            return int(v)
    
        if isinstance(v, (np.floating, float)):
            return float(v)
    
        return v
    
    def _extract_market_open(
        self,
        intraday_df: Optional[pd.DataFrame],
        daily_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract indicators at market open using daily data for indicators + intraday for current price"""
        try:
            target_date_obj = target_date.date()
            
            # Get today's indicators from daily data (will be most recent available)
            if target_date_obj not in daily_df.index:
                # Use most recent date
                available_dates = [d for d in daily_df.index if d <= target_date_obj]
                if not available_dates:
                    self.logger.debug(f"No daily data available for market open {symbol}")
                    return None
                indicator_date = available_dates[-1]
            else:
                indicator_date = target_date_obj
            
            # Get the indicator values
            indicator_data = daily_df.loc[indicator_date]
            
            # Get current OHLCV from intraday if available
            if intraday_df is not None and len(intraday_df) > 0:
                # Find morning bars
                morning_bars = intraday_df[
                    (intraday_df.index.time >= time(9, 30)) &
                    (intraday_df.index.time <= time(10, 0))
                ]
                
                if morning_bars.empty:
                    morning_bars = intraday_df[intraday_df.index.time >= time(9, 30)]
                    if morning_bars.empty:
                        morning_bars = intraday_df.head(1)
                
                if not morning_bars.empty:
                    current_bar = morning_bars.iloc[0]
                    actual_time = morning_bars.index[0].strftime('%H:%M:%S')
                    self.logger.debug(f"Using {actual_time} bar for market_open {symbol}")
                    
                    # Override OHLCV with current values
                    indicator_data = indicator_data.copy()
                    indicator_data['open'] = current_bar['Open']
                    indicator_data['high'] = current_bar['High']
                    indicator_data['low'] = current_bar['Low']
                    indicator_data['close'] = current_bar['Close']
                    indicator_data['volume'] = current_bar['Volume']
            
            snapshot = {
                'symbol': symbol,
                'exchange': exchange,
                'detection_date': target_date_obj.isoformat(),
                'snapshot_type': 'market_open',
                'snapshot_time': '09:30:00'
            }
            
            reserved_fields = {'symbol', 'exchange', 'detection_date', 'snapshot_type', 'snapshot_time'}
            
            # Add all indicator values
            for key, value in indicator_data.items():
                key_lower = key.lower()
                if key_lower in reserved_fields:
                    continue
                    
                snapshot[key_lower] = self._serialize_value(value)
            
            self.logger.debug(f"Extracted market_open for {symbol} with {len(snapshot)} fields")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error extracting market open for {symbol}: {e}", exc_info=True)
            return None
    
    def _extract_market_close(
        self,
        intraday_df: Optional[pd.DataFrame],
        daily_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime
    ) -> Dict[str, Any]:
        """
        Extract indicators at market close
        ALWAYS returns a dict (even if market hasn't closed) with metadata fields
        Indicator fields will be null if market hasn't closed yet
        """
        target_date_obj = target_date.date()
        
        # Start with base snapshot (always created)
        snapshot = {
            'symbol': symbol,
            'exchange': exchange,
            'detection_date': target_date_obj.isoformat(),
            'snapshot_type': 'market_close',
            'snapshot_time': '16:00:00'
        }
        
        try:
            # Check if we have close bars in intraday data
            has_close_data = False
            
            if intraday_df is not None and not intraday_df.empty:
                close_bars = intraday_df[
                    (intraday_df.index.time >= time(15, 55)) &
                    (intraday_df.index.time <= time(16, 0))
                ]
                
                if not close_bars.empty:
                    has_close_data = True
                    
                    # Get indicator values from daily data
                    if target_date_obj in daily_df.index:
                        indicator_date = target_date_obj
                    else:
                        available_dates = [d for d in daily_df.index if d <= target_date_obj]
                        if available_dates:
                            indicator_date = available_dates[-1]
                        else:
                            self.logger.debug(f"No daily data for market close {symbol}")
                            return snapshot  # Return with just metadata, no indicators
                    
                    indicator_data = daily_df.loc[indicator_date].copy()
                    
                    # Use actual close values
                    current_bar = close_bars.iloc[-1]
                    actual_time = close_bars.index[-1].strftime('%H:%M:%S')
                    self.logger.debug(f"Using {actual_time} bar for market_close {symbol}")
                    
                    indicator_data['open'] = current_bar['Open']
                    indicator_data['high'] = current_bar['High']
                    indicator_data['low'] = current_bar['Low']
                    indicator_data['close'] = current_bar['Close']
                    indicator_data['volume'] = current_bar['Volume']
                    
                    # Add all indicator values
                    reserved_fields = {'symbol', 'exchange', 'detection_date', 'snapshot_type', 'snapshot_time'}
                    
                    for key, value in indicator_data.items():
                        key_lower = key.lower()
                        if key_lower in reserved_fields:
                            continue
                            
                        snapshot[key_lower] = self._serialize_value(value)
                    
                    self.logger.debug(f"Extracted market_close for {symbol} with {len(snapshot)} fields")
                else:
                    self.logger.debug(f"Market hasn't closed yet for {symbol} - creating row with nulls")
            else:
                self.logger.debug(f"No intraday data for market_close {symbol} - creating row with nulls")
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error extracting market close for {symbol}: {e}", exc_info=True)
            # Still return the base snapshot with metadata even on error
            return snapshot
    
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
            
            available_dates = [d for d in daily_df.index if d <= prior_date]
            
            if not available_dates:
                self.logger.debug(f"No prior dates for {symbol}")
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
            
            reserved_fields = {'symbol', 'exchange', 'detection_date', 'snapshot_type', 
                              'snapshot_time', 'snapshot_date'}
            
            for key, value in prior_data.items():
                key_lower = key.lower()
                if key_lower in reserved_fields:
                    continue
                    
                snapshot[key_lower] = self._serialize_value(value)
            
            self.logger.debug(f"Extracted day_prior for {symbol} with {len(snapshot)} fields")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error extracting day prior for {symbol}: {e}", exc_info=True)
            return None
    
    def _calculate_minimal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators matching database schema"""
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
        
        # MACD
        try:
            macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            result['macd.macd'] = macd.macd()
            result['macd.signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
        except:
            pass
        
        # Stochastic
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            result['stoch.k'] = stoch.stoch()
            result['stoch.d'] = stoch.stoch_signal()
            result['stoch.k[1]'] = result['stoch.k'].shift(1)
            result['stoch.d[1]'] = result['stoch.d'].shift(1)
        except:
            pass
        
        # ADX
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['adx'] = adx.adx()
            result['adx+di'] = adx.adx_pos()
            result['adx-di'] = adx.adx_neg()
        except:
            pass
        
        # Bollinger Bands
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
        
        # Gaps
        try:
            result['gap_%'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
            result['gap_up'] = (result['gap_%'] > 2).astype(int)
            result['gap_down'] = (result['gap_%'] < -2).astype(int)
        except:
            pass
        
        # Trend indicators
        try:
            result['ema20_above_ema50'] = (result['ema20'] > result['ema50']).astype(int)
            result['ema50_above_ema200'] = (result['ema50'] > result['ema200']).astype(int)
            result['price_above_ema20'] = (df['Close'] > result['ema20']).astype(int)
            result['ema10_above_ema20'] = (result['ema10'] > result['ema20']).astype(int)
        except:
            pass
        
        return result
