"""
Intraday Data Collector - ENHANCED VERSION
Captures indicators at ACTUAL market open (9:30am) and close (4pm) using intraday data
ENHANCED: Added T-1 open data collection and many more technical indicators
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
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, AwesomeOscillatorIndicator, UltimateOscillator, ROCIndicator, KAMAIndicator, TSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator, AroonIndicator, PSARIndicator, VortexIndicator, MassIndex, DPOIndicator, KSTIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator, NegativeVolumeIndexIndicator, VolumeWeightedAveragePrice


class IntradayDataCollector:
    """
    Collects technical indicator data at specific times using INTRADAY data:
    - Market open (9:30am NYC) - from 5-minute bars
    - Market close (4pm NYC) - from 5-minute bars
    - Previous day OPEN (T-1 9:30am) - from 5-minute bars
    - Previous day CLOSE (T-1 4pm) - from daily bars
    
    Includes comprehensive set of technical indicators
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
        
        self.logger.info("Intraday data collector initialized (ENHANCED with T-1 open)")
    
    def collect_intraday_data(
        self,
        winners: List[Dict[str, Any]],
        target_date: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect indicator data at market open, close, T-1 open, and T-1 close
        
        Args:
            winners: List of winner dictionaries
            target_date: Date to collect data for
            
        Returns:
            Dictionary with 'market_open', 'market_close', 'day_prior_open', 'day_prior_close'
        """
        self.logger.info(f"Collecting intraday data for {len(winners)} winners on {target_date.date()}...")
        
        self.stats['total'] = len(winners)
        
        all_data = {
            'market_open': [],
            'market_close': [],
            'day_prior_open': [],
            'day_prior_close': []
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
                        if result['day_prior_open']:
                            all_data['day_prior_open'].append(result['day_prior_open'])
                        if result['day_prior_close']:
                            all_data['day_prior_close'].append(result['day_prior_close'])
                except Exception as e:
                    winner = future_to_winner[future]
                    self.logger.debug(f"Error processing {winner.get('symbol', 'unknown')}: {e}")
                    self.stats['failed'] += 1
        
        self.logger.info(
            f"✓ Collected - "
            f"Market Open: {len(all_data['market_open'])}, "
            f"Market Close: {len(all_data['market_close'])}, "
            f"Day Prior Open: {len(all_data['day_prior_open'])}, "
            f"Day Prior Close: {len(all_data['day_prior_close'])}"
        )
        
        return all_data
    
    def _process_winner(
        self,
        winner: Dict,
        target_date: datetime
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Process a single winner and collect indicators at four time points"""
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
            
            # Fetch PREVIOUS DAY's intraday data for T-1 open
            prior_date = target_date - timedelta(days=1)
            prior_intraday_df = self._fetch_intraday_data_today(symbol, prior_date)
            
            self.stats['success'] += 1
            
            # Extract market open snapshot (current day 9:30am)
            market_open_snapshot = self._extract_market_open(
                intraday_df, daily_df, symbol, exchange, target_date
            )
            
            # Extract market close snapshot (current day 4pm)
            market_close_snapshot = self._extract_market_close(
                intraday_df, daily_df, symbol, exchange, target_date
            )
            
            # Extract day prior OPEN snapshot (T-1 9:30am)
            day_prior_open_snapshot = self._extract_day_prior_open(
                prior_intraday_df, daily_df, symbol, exchange, target_date
            )
            
            # Extract day prior CLOSE snapshot (T-1 4pm, from daily data)
            day_prior_close_snapshot = self._extract_day_prior_close(
                daily_df, symbol, exchange, target_date
            )
            
            return {
                'market_open': market_open_snapshot,
                'market_close': market_close_snapshot,
                'day_prior_open': day_prior_open_snapshot,
                'day_prior_close': day_prior_close_snapshot
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
        Fetch a specific day's 5-minute intraday data
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
            indicators_df = self._calculate_enhanced_indicators(df)
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
    
    def _extract_day_prior_open(
        self,
        prior_intraday_df: Optional[pd.DataFrame],
        daily_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract indicators for previous day's market open (T-1 9:30am)"""
        try:
            prior_date = target_date.date() - timedelta(days=1)
            
            # Get prior day's indicators from daily data
            available_dates = [d for d in daily_df.index if d <= prior_date]
            
            if not available_dates:
                self.logger.debug(f"No prior dates for {symbol}")
                return None
            
            actual_date = available_dates[-1]
            prior_data = daily_df.loc[actual_date].copy()
            
            # Try to get actual open price from intraday data
            if prior_intraday_df is not None and len(prior_intraday_df) > 0:
                # Find morning bars from previous day
                morning_bars = prior_intraday_df[
                    (prior_intraday_df.index.time >= time(9, 30)) &
                    (prior_intraday_df.index.time <= time(10, 0))
                ]
                
                if morning_bars.empty:
                    morning_bars = prior_intraday_df[prior_intraday_df.index.time >= time(9, 30)]
                    if morning_bars.empty:
                        morning_bars = prior_intraday_df.head(1)
                
                if not morning_bars.empty:
                    current_bar = morning_bars.iloc[0]
                    actual_time = morning_bars.index[0].strftime('%H:%M:%S')
                    self.logger.debug(f"Using {actual_time} bar for day_prior_open {symbol}")
                    
                    # Override OHLCV with actual values from that morning
                    prior_data['open'] = current_bar['Open']
                    prior_data['high'] = current_bar['High']
                    prior_data['low'] = current_bar['Low']
                    prior_data['close'] = current_bar['Close']
                    prior_data['volume'] = current_bar['Volume']
            
            snapshot = {
                'symbol': symbol,
                'exchange': exchange,
                'detection_date': target_date.date().isoformat(),
                'snapshot_type': 'day_prior_open',
                'snapshot_time': '09:30:00',
                'snapshot_date': actual_date.isoformat()
            }
            
            reserved_fields = {'symbol', 'exchange', 'detection_date', 'snapshot_type', 
                              'snapshot_time', 'snapshot_date'}
            
            for key, value in prior_data.items():
                key_lower = key.lower()
                if key_lower in reserved_fields:
                    continue
                    
                snapshot[key_lower] = self._serialize_value(value)
            
            self.logger.debug(f"Extracted day_prior_open for {symbol} with {len(snapshot)} fields")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error extracting day prior open for {symbol}: {e}", exc_info=True)
            return None
    
    def _extract_day_prior_close(
        self,
        daily_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Extract indicators for previous day's market close (T-1 4pm)"""
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
                'snapshot_type': 'day_prior_close',
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
            
            self.logger.debug(f"Extracted day_prior_close for {symbol} with {len(snapshot)} fields")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error extracting day prior close for {symbol}: {e}", exc_info=True)
            return None
    
    def _calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators matching database schema"""
        result = pd.DataFrame(index=df.index)
        
        # Basic OHLCV
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        # === MOMENTUM INDICATORS ===
        
        # RSI (14-period)
        try:
            rsi_ind = RSIIndicator(close=df['Close'], window=14)
            result['rsi'] = rsi_ind.rsi()
            result['rsi[1]'] = result['rsi'].shift(1)
            result['rsi[2]'] = result['rsi'].shift(2)
        except:
            pass
        
        # Momentum (10-period)
        try:
            result['mom'] = df['Close'].diff(10)
            result['mom[1]'] = result['mom'].shift(1)
        except:
            pass
        
        # Rate of Change (ROC)
        try:
            roc = ROCIndicator(close=df['Close'], window=12)
            result['roc'] = roc.roc()
        except:
            pass
        
        # Stochastic Oscillator
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            result['stoch.k'] = stoch.stoch()
            result['stoch.d'] = stoch.stoch_signal()
            result['stoch.k[1]'] = result['stoch.k'].shift(1)
            result['stoch.d[1]'] = result['stoch.d'].shift(1)
        except:
            pass
        
        # Williams %R
        try:
            wr = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14)
            result['w.r'] = wr.williams_r()
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
        
        # KAMA (Kaufman's Adaptive Moving Average)
        try:
            kama = KAMAIndicator(close=df['Close'], window=10, pow1=2, pow2=30)
            result['kama'] = kama.kama()
        except:
            pass
        
        # TSI (True Strength Index)
        try:
            tsi = TSIIndicator(close=df['Close'], window_slow=25, window_fast=13)
            result['tsi'] = tsi.tsi()
        except:
            pass
        
        # === TREND INDICATORS ===
        
        # MACD
        try:
            macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            result['macd.macd'] = macd.macd()
            result['macd.signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
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
        
        # CCI (Commodity Channel Index)
        try:
            cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['cci20'] = cci.cci()
        except:
            pass
        
        # Aroon Indicator
        try:
            aroon = AroonIndicator(close=df['Close'], window=25)
            result['aroon_up'] = aroon.aroon_up()
            result['aroon_down'] = aroon.aroon_down()
            result['aroon_indicator'] = aroon.aroon_indicator()
        except:
            pass
        
        # Parabolic SAR
        try:
            psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'])
            result['psar'] = psar.psar()
            result['psar_up'] = psar.psar_up()
            result['psar_down'] = psar.psar_down()
        except:
            pass
        
        # Vortex Indicator
        try:
            vortex = VortexIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['vortex_pos'] = vortex.vortex_indicator_pos()
            result['vortex_neg'] = vortex.vortex_indicator_neg()
        except:
            pass
        
        # Mass Index
        try:
            mass = MassIndex(high=df['High'], low=df['Low'], window_fast=9, window_slow=25)
            result['mass_index'] = mass.mass_index()
        except:
            pass
        
        # DPO (Detrended Price Oscillator)
        try:
            dpo = DPOIndicator(close=df['Close'], window=20)
            result['dpo'] = dpo.dpo()
        except:
            pass
        
        # KST (Know Sure Thing)
        try:
            kst = KSTIndicator(close=df['Close'])
            result['kst'] = kst.kst()
            result['kst_signal'] = kst.kst_sig()
        except:
            pass
        
        # === VOLATILITY INDICATORS ===
        
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
        
        # ATR (Average True Range)
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['atr'] = atr.average_true_range()
            result['atr_pct'] = (result['atr'] / df['Close']) * 100
        except:
            pass
        
        # Keltner Channel
        try:
            keltner = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['keltner_upper'] = keltner.keltner_channel_hband()
            result['keltner_lower'] = keltner.keltner_channel_lband()
            result['keltner_middle'] = keltner.keltner_channel_mband()
        except:
            pass
        
        # Donchian Channel
        try:
            donchian = DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['donchian_upper'] = donchian.donchian_channel_hband()
            result['donchian_lower'] = donchian.donchian_channel_lband()
            result['donchian_middle'] = donchian.donchian_channel_mband()
        except:
            pass
        
        # === VOLUME INDICATORS ===
        
        # On-Balance Volume (OBV)
        try:
            obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
            result['obv'] = obv.on_balance_volume()
        except:
            pass
        
        # Chaikin Money Flow
        try:
            cmf = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], 
                                           volume=df['Volume'], window=20)
            result['cmf'] = cmf.chaikin_money_flow()
        except:
            pass
        
        # Force Index
        try:
            fi = ForceIndexIndicator(close=df['Close'], volume=df['Volume'], window=13)
            result['force_index'] = fi.force_index()
        except:
            pass
        
        # Ease of Movement
        try:
            eom = EaseOfMovementIndicator(high=df['High'], low=df['Low'], volume=df['Volume'], window=14)
            result['eom'] = eom.ease_of_movement()
            result['eom_signal'] = eom.sma_ease_of_movement()
        except:
            pass
        
        # Volume Price Trend
        try:
            vpt = VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'])
            result['vpt'] = vpt.volume_price_trend()
        except:
            pass
        
        # Negative Volume Index
        try:
            nvi = NegativeVolumeIndexIndicator(close=df['Close'], volume=df['Volume'])
            result['nvi'] = nvi.negative_volume_index()
        except:
            pass
        
        # === MOVING AVERAGES ===
        
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
        
        # Volume SMAs
        try:
            result['volume_sma5'] = result['volume'].rolling(window=5).mean()
            result['volume_sma10'] = result['volume'].rolling(window=10).mean()
            result['volume_sma20'] = result['volume'].rolling(window=20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma20']
        except:
            pass
        
        # === PRICE CHANGES ===
        
        for days in [1, 2, 3, 5, 10, 20, 30]:
            try:
                result[f'price_change_{days}d'] = df['Close'].pct_change(days) * 100
            except:
                pass
        
        # === VOLATILITY MEASURES ===
        
        try:
            result['volatility_10d'] = df['Close'].pct_change().rolling(window=10).std() * 100 * np.sqrt(252)
            result['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * 100 * np.sqrt(252)
            result['volatility_30d'] = df['Close'].pct_change().rolling(window=30).std() * 100 * np.sqrt(252)
        except:
            pass
        
        # === VWAP ===
        
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            result['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        except:
            pass
        
        # === 52-WEEK HIGH/LOW ===
        
        try:
            result['high_52w'] = df['High'].rolling(window=252, min_periods=1).max()
            result['low_52w'] = df['Low'].rolling(window=252, min_periods=1).min()
            result['price_vs_high_52w'] = (df['Close'] / result['high_52w'] - 1) * 100
            result['price_vs_low_52w'] = (df['Close'] / result['low_52w'] - 1) * 100
        except:
            pass
        
        # === GAPS ===
        
        try:
            result['gap_%'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
            result['gap_up'] = (result['gap_%'] > 2).astype(int)
            result['gap_down'] = (result['gap_%'] < -2).astype(int)
        except:
            pass
        
        # === TREND FLAGS ===
        
        try:
            result['ema20_above_ema50'] = (result['ema20'] > result['ema50']).astype(int)
            result['ema50_above_ema200'] = (result['ema50'] > result['ema200']).astype(int)
            result['price_above_ema20'] = (df['Close'] > result['ema20']).astype(int)
            result['ema10_above_ema20'] = (result['ema10'] > result['ema20']).astype(int)
            result['sma50_above_sma200'] = (result['sma50'] > result['sma200']).astype(int)
        except:
            pass
        
        # === CANDLESTICK PATTERNS (simple) ===
        
        try:
            # Doji
            body = abs(df['Close'] - df['Open'])
            range_hl = df['High'] - df['Low']
            result['doji'] = (body / range_hl < 0.1).astype(int)
            
            # Hammer
            lower_shadow = df['Open'].combine(df['Close'], min) - df['Low']
            upper_shadow = df['High'] - df['Open'].combine(df['Close'], max)
            result['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
            
            # Engulfing
            prev_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
            result['bullish_engulfing'] = ((df['Close'] > df['Open']) & 
                                          (df['Close'].shift(1) < df['Open'].shift(1)) &
                                          (body > prev_body)).astype(int)
        except:
            pass
        
        return result
