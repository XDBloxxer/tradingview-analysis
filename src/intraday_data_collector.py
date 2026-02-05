"""
Intraday Data Collector - FIXED VERSION
Captures indicators at ACTUAL market open (9:30am) and close (4pm) using intraday data
Can fetch data for current day or historical dates
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
    Collects technical indicator data at specific times using INTRADAY data:
    - Market open (9:30am NYC) - from 5-minute bars
    - Market close (4pm NYC) - from 5-minute bars
    - Previous day (T-1) - from daily bars
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
        
        # Cache for already-fetched data
        self.cache = {}
        
        self.logger.info("Intraday data collector initialized (using 5-min bars for market open/close)")
    
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
        
        try:
            # Check if target date is today - affects data availability
            is_today = target_date.date() == datetime.now().date()
            
            # Fetch INTRADAY data for market open/close
            # For today: we can get current intraday bars
            # For historical: yfinance limits to last 60 days
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
        
        Args:
            symbol: Stock symbol
            target_date: Target date
            is_today: Whether target date is today (affects data fetching strategy)
        """
        cache_key = f"{symbol}:intraday:{target_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            if is_today:
                # For today, fetch last available intraday data
                df = ticker.history(period='1d', interval='5m')
            else:
                # For historical dates - yfinance only keeps 60 days of intraday
                # Create datetime range for target date
                start_dt = datetime.combine(target_date, time(9, 0))  # 9am
                end_dt = datetime.combine(target_date + timedelta(days=1), time(0, 0))  # Next day midnight
                
                # Check if date is within 60-day window
                days_ago = (datetime.now().date() - target_date).days
                if days_ago > 60:
                    self.logger.debug(f"Date {target_date} is beyond 60-day intraday limit for {symbol}")
                    return None
                
                # Fetch 5-minute data
                df = ticker.history(
                    start=start_dt,
                    end=end_dt,
                    interval='5m'
                )
            
            if df.empty or len(df) < 5:
                self.logger.debug(f"Insufficient intraday data for {symbol}")
                return None
            
            # Calculate indicators on 5-min data
            indicators_df = self._calculate_indicators_lightweight(df)
            
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
            
            indicators_df = self._calculate_all_indicators(df)
            indicators_df.index = pd.to_datetime(indicators_df.index).date
            
            self.cache[cache_key] = indicators_df
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching daily for {symbol}: {e}")
            return None
    
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
                # Try to get first bar of the day
                morning_bars = intraday_df[intraday_df.index.time >= time(9, 30)]
                if morning_bars.empty:
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
            
            # Add indicators
            for key, value in open_data.items():
                if pd.notna(value):
                    try:
                        snapshot[key.lower()] = float(value)
                    except:
                        snapshot[key.lower()] = None
            
            return snapshot
            
        except Exception as e:
            self.logger.debug(f"Error extracting market open: {e}")
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
                # Try to get last bar of the day
                close_bars = intraday_df[intraday_df.index.time <= time(16, 0)]
                if close_bars.empty:
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
            
            # Add indicators
            for key, value in close_data.items():
                if pd.notna(value):
                    try:
                        snapshot[key.lower()] = float(value)
                    except:
                        snapshot[key.lower()] = None
            
            return snapshot
            
        except Exception as e:
            self.logger.debug(f"Error extracting market close: {e}")
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
            
            # Add indicators
            for key, value in prior_data.items():
                if pd.notna(value):
                    try:
                        snapshot[key.lower()] = float(value)
                    except:
                        snapshot[key.lower()] = None
            
            return snapshot
            
        except Exception as e:
            self.logger.debug(f"Error extracting day prior: {e}")
            return None
    
    def _calculate_indicators_lightweight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators on intraday data (lighter weight)"""
        result = pd.DataFrame(index=df.index)
        
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        # Only calculate fast indicators for intraday
        try:
            rsi = RSIIndicator(close=df['Close'], window=14)
            result['rsi'] = rsi.rsi()
        except:
            pass
        
        try:
            macd = MACD(close=df['Close'])
            result['macd.macd'] = macd.macd()
            result['macd.signal'] = macd.macd_signal()
        except:
            pass
        
        try:
            result['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = df['Volume'] / result['volume_sma_20']
        except:
            pass
        
        # EMAs
        for period in [10, 20, 50]:
            try:
                result[f'ema{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            except:
                pass
        
        return result
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate full indicators on daily data"""
        result = pd.DataFrame(index=df.index)
        
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        # RSI
        try:
            rsi = RSIIndicator(close=df['Close'], window=14)
            result['rsi'] = rsi.rsi()
        except:
            pass
        
        # MACD
        try:
            macd = MACD(close=df['Close'])
            result['macd.macd'] = macd.macd()
            result['macd.signal'] = macd.macd_signal()
        except:
            pass
        
        # Stochastic
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
            result['stoch.k'] = stoch.stoch()
            result['stoch.d'] = stoch.stoch_signal()
        except:
            pass
        
        # ADX
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
            result['adx'] = adx.adx()
        except:
            pass
        
        # Bollinger Bands
        try:
            bb = BollingerBands(close=df['Close'])
            result['bb.upper'] = bb.bollinger_hband()
            result['bb.lower'] = bb.bollinger_lband()
            result['bb.middle'] = bb.bollinger_mavg()
        except:
            pass
        
        # ATR
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
            result['atr'] = atr.average_true_range()
        except:
            pass
        
        # EMAs
        for period in [10, 20, 50, 200]:
            try:
                result[f'ema{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            except:
                pass
        
        # SMAs
        for period in [10, 20, 50, 200]:
            try:
                result[f'sma{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
            except:
                pass
        
        # Volume indicators
        try:
            result['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = df['Volume'] / result['volume_sma_20']
        except:
            pass
        
        return result
