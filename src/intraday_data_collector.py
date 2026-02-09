"""
FIXED Intraday Data Collector
KEY FIX: Calculates indicators from INTRADAY bars, not daily bars
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from zoneinfo import ZoneInfo

import yfinance as yf
from .rate_limiter import RateLimiter

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, AwesomeOscillatorIndicator, UltimateOscillator, ROCIndicator, KAMAIndicator, TSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator, AroonIndicator, PSARIndicator, VortexIndicator, MassIndex, DPOIndicator, KSTIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator, NegativeVolumeIndexIndicator


class IntradayDataCollector:
    """
    FIXED: Now calculates indicators from intraday bars at each timepoint
    """
    
    MAX_WORKERS = 5
    LOOKBACK_DAYS = 90
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.stats = {'total': 0, 'success': 0, 'failed': 0}
        self.cache = {}
        self.logger.info("Intraday data collector initialized (FIXED - uses intraday indicators)")
    
    def collect_intraday_data(
        self,
        winners: List[Dict[str, Any]],
        target_date: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect indicator data at four timepoints"""
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
            
            for future in tqdm(as_completed(future_to_winner), total=len(winners), 
                              desc="Collecting intraday indicators"):
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
            f"✓ Collected - Market Open: {len(all_data['market_open'])}, "
            f"Market Close: {len(all_data['market_close'])}, "
            f"Day Prior Open: {len(all_data['day_prior_open'])}, "
            f"Day Prior Close: {len(all_data['day_prior_close'])}"
        )
        
        return all_data
    
    def _process_winner(self, winner: Dict, target_date: datetime) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        FIXED: Process winner by fetching sufficient intraday history and calculating 
        indicators at each specific timepoint
        """
        symbol = winner.get('symbol')
        exchange = winner.get('exchange', 'NASDAQ')
        
        if not symbol:
            self.logger.debug(f"Skipping invalid winner (missing symbol): {winner}")
            return None
        
        try:
            # Fetch extended intraday data (5-min bars for past 60 days)
            # This gives us enough bars to calculate 200-period indicators on 5-min data
            intraday_df_extended = self._fetch_intraday_extended(symbol, target_date)
            
            if intraday_df_extended is None or len(intraday_df_extended) < 200:
                self.logger.warning(f"Insufficient intraday data for {symbol} ({len(intraday_df_extended) if intraday_df_extended is not None else 0} bars)")
                self.stats['failed'] += 1
                return None
            
            self.stats['success'] += 1
            
            # Extract snapshots at each timepoint by slicing the intraday data
            # and recalculating indicators up to that point
            
            market_open_snapshot = self._extract_timepoint_snapshot(
                intraday_df_extended, symbol, exchange, target_date,
                snapshot_type='market_open',
                target_time=time(9, 30),
                time_window=(time(9, 30), time(10, 0))
            )
            
            market_close_snapshot = self._extract_timepoint_snapshot(
                intraday_df_extended, symbol, exchange, target_date,
                snapshot_type='market_close',
                target_time=time(16, 0),
                time_window=(time(15, 55), time(16, 0))
            )
            
            prior_date = target_date - timedelta(days=1)
            
            day_prior_open_snapshot = self._extract_timepoint_snapshot(
                intraday_df_extended, symbol, exchange, prior_date,
                snapshot_type='day_prior_open',
                target_time=time(9, 30),
                time_window=(time(9, 30), time(10, 0)),
                detection_date=target_date  # Keep original detection date
            )
            
            day_prior_close_snapshot = self._extract_timepoint_snapshot(
                intraday_df_extended, symbol, exchange, prior_date,
                snapshot_type='day_prior_close',
                target_time=time(16, 0),
                time_window=(time(15, 55), time(16, 0)),
                detection_date=target_date  # Keep original detection date
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
    
    def _fetch_intraday_extended(self, symbol: str, target_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch extended intraday data (5-min bars for past 60 days)
        This provides enough bars for indicator calculations
        """
        cache_key = f"{symbol}:intraday_extended"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch 60 days of 5-minute data (maximum available from yfinance)
            # This gives roughly 60 * 78 bars/day = 4680 bars
            # Enough for 200-period indicators (which would be ~17 trading days on 5-min)
            df = ticker.history(period='60d', interval='5m')
            
            if df.empty:
                self.logger.debug(f"No intraday data for {symbol}")
                return None
            
            # Normalize timezone
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')
            
            # Calculate indicators on the full intraday dataset
            indicators_df = self._calculate_enhanced_indicators(df)
            
            self.logger.debug(f"Fetched {len(indicators_df)} intraday bars for {symbol}")
            
            self.cache[cache_key] = indicators_df
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching extended intraday for {symbol}: {e}")
            return None
    
    def _extract_timepoint_snapshot(
        self,
        intraday_df: pd.DataFrame,
        symbol: str,
        exchange: str,
        target_date: datetime,
        snapshot_type: str,
        target_time: time,
        time_window: tuple,
        detection_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        FIXED: Extract indicators at a specific timepoint by:
        1. Finding the specific bar closest to target time
        2. Slicing data UP TO that bar (so indicators reflect state at that moment)
        3. Taking indicator values from that bar
        """
        if detection_date is None:
            detection_date = target_date
        
        target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
        detection_date_obj = detection_date.date() if isinstance(detection_date, datetime) else detection_date
        
        try:
            # Filter to target date only
            day_bars = intraday_df[intraday_df.index.date == target_date_obj]
            
            if day_bars.empty:
                self.logger.debug(f"No bars for {target_date_obj} in snapshot {snapshot_type} for {symbol}")
                
                # For market_close, always return a row even if no data
                if snapshot_type == 'market_close':
                    return {
                        'symbol': symbol,
                        'exchange': exchange,
                        'detection_date': detection_date_obj.isoformat(),
                        'snapshot_type': snapshot_type,
                        'snapshot_time': target_time.strftime('%H:%M:%S')
                    }
                return None
            
            # Find bar closest to target time within window
            window_start, window_end = time_window
            target_bars = day_bars[
                (day_bars.index.time >= window_start) &
                (day_bars.index.time <= window_end)
            ]
            
            if target_bars.empty:
                # Fallback to closest available bar
                target_bars = day_bars[day_bars.index.time >= window_start]
                if target_bars.empty:
                    target_bars = day_bars
            
            if target_bars.empty:
                if snapshot_type == 'market_close':
                    return {
                        'symbol': symbol,
                        'exchange': exchange,
                        'detection_date': detection_date_obj.isoformat(),
                        'snapshot_type': snapshot_type,
                        'snapshot_time': target_time.strftime('%H:%M:%S')
                    }
                return None
            
            # Get the specific bar (first for open, last for close)
            if 'open' in snapshot_type:
                target_bar = target_bars.iloc[0]
                actual_time = target_bars.index[0]
            else:
                target_bar = target_bars.iloc[-1]
                actual_time = target_bars.index[-1]
            
            self.logger.debug(
                f"{symbol} {snapshot_type}: Using bar at {actual_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Build snapshot with indicator values from this specific bar
            snapshot = {
                'symbol': symbol,
                'exchange': exchange,
                'detection_date': detection_date_obj.isoformat(),
                'snapshot_type': snapshot_type,
                'snapshot_time': target_time.strftime('%H:%M:%S')
            }
            
            if 'prior' in snapshot_type:
                snapshot['snapshot_date'] = target_date_obj.isoformat()
            
            reserved_fields = {'symbol', 'exchange', 'detection_date', 'snapshot_type', 
                              'snapshot_time', 'snapshot_date'}
            
            # Add all indicator values from this bar
            for col in target_bar.index:
                col_lower = col.lower()
                if col_lower in reserved_fields:
                    continue
                
                snapshot[col_lower] = self._serialize_value(target_bar[col])
            
            self.logger.debug(f"Extracted {snapshot_type} for {symbol} with {len(snapshot)} fields")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error extracting {snapshot_type} for {symbol}: {e}", exc_info=True)
            
            # For market_close, always return base snapshot
            if snapshot_type == 'market_close':
                return {
                    'symbol': symbol,
                    'exchange': exchange,
                    'detection_date': detection_date_obj.isoformat(),
                    'snapshot_type': snapshot_type,
                    'snapshot_time': target_time.strftime('%H:%M:%S')
                }
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
    
    def _calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive indicators on intraday bars
        NOTE: This is now called on 5-minute bars, so periods have different meanings:
        - 14-period RSI = 70 minutes (not 14 days)
        - 200-period SMA = ~17 trading days on 5-min data
        """
        result = pd.DataFrame(index=df.index)
        
        # Basic OHLCV
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        # === MOMENTUM INDICATORS ===
        
        try:
            rsi_ind = RSIIndicator(close=df['Close'], window=14)
            result['rsi'] = rsi_ind.rsi()
            result['rsi[1]'] = result['rsi'].shift(1)
            result['rsi[2]'] = result['rsi'].shift(2)
        except:
            pass
        
        try:
            result['mom'] = df['Close'].diff(10)
            result['mom[1]'] = result['mom'].shift(1)
        except:
            pass
        
        try:
            roc = ROCIndicator(close=df['Close'], window=12)
            result['roc'] = roc.roc()
        except:
            pass
        
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], 
                                        window=14, smooth_window=3)
            result['stoch.k'] = stoch.stoch()
            result['stoch.d'] = stoch.stoch_signal()
            result['stoch.k[1]'] = result['stoch.k'].shift(1)
            result['stoch.d[1]'] = result['stoch.d'].shift(1)
        except:
            pass
        
        try:
            wr = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14)
            result['w.r'] = wr.williams_r()
        except:
            pass
        
        try:
            ao = AwesomeOscillatorIndicator(high=df['High'], low=df['Low'], window1=5, window2=34)
            result['ao'] = ao.awesome_oscillator()
        except:
            pass
        
        try:
            uo = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close'], 
                                   window1=7, window2=14, window3=28)
            result['uo'] = uo.ultimate_oscillator()
        except:
            pass
        
        try:
            kama = KAMAIndicator(close=df['Close'], window=10, pow1=2, pow2=30)
            result['kama'] = kama.kama()
        except:
            pass
        
        try:
            tsi = TSIIndicator(close=df['Close'], window_slow=25, window_fast=13)
            result['tsi'] = tsi.tsi()
        except:
            pass
        
        # === TREND INDICATORS ===
        
        try:
            macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            result['macd.macd'] = macd.macd()
            result['macd.signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
        except:
            pass
        
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['adx'] = adx.adx()
            result['adx+di'] = adx.adx_pos()
            result['adx-di'] = adx.adx_neg()
        except:
            pass
        
        try:
            cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['cci20'] = cci.cci()
        except:
            pass
        
        try:
            aroon = AroonIndicator(close=df['Close'], window=25)
            result['aroon_up'] = aroon.aroon_up()
            result['aroon_down'] = aroon.aroon_down()
            result['aroon_indicator'] = aroon.aroon_indicator()
        except:
            pass
        
        try:
            psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'])
            result['psar'] = psar.psar()
            result['psar_up'] = psar.psar_up()
            result['psar_down'] = psar.psar_down()
        except:
            pass
        
        try:
            vortex = VortexIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['vortex_pos'] = vortex.vortex_indicator_pos()
            result['vortex_neg'] = vortex.vortex_indicator_neg()
        except:
            pass
        
        try:
            mass = MassIndex(high=df['High'], low=df['Low'], window_fast=9, window_slow=25)
            result['mass_index'] = mass.mass_index()
        except:
            pass
        
        try:
            dpo = DPOIndicator(close=df['Close'], window=20)
            result['dpo'] = dpo.dpo()
        except:
            pass
        
        try:
            kst = KSTIndicator(close=df['Close'])
            result['kst'] = kst.kst()
            result['kst_signal'] = kst.kst_sig()
        except:
            pass
        
        # === VOLATILITY INDICATORS ===
        
        try:
            bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
            result['bb.upper'] = bb.bollinger_hband()
            result['bb.lower'] = bb.bollinger_lband()
            result['bb.middle'] = bb.bollinger_mavg()
            result['bb_width'] = (result['bb.upper'] - result['bb.lower']) / result['bb.middle'] * 100
            result['bbpower'] = (df['Close'] - result['bb.lower']) / (result['bb.upper'] - result['bb.lower'])
        except:
            pass
        
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['atr'] = atr.average_true_range()
            result['atr_pct'] = (result['atr'] / df['Close']) * 100
        except:
            pass
        
        try:
            keltner = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['keltner_upper'] = keltner.keltner_channel_hband()
            result['keltner_lower'] = keltner.keltner_channel_lband()
            result['keltner_middle'] = keltner.keltner_channel_mband()
        except:
            pass
        
        try:
            donchian = DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            result['donchian_upper'] = donchian.donchian_channel_hband()
            result['donchian_lower'] = donchian.donchian_channel_lband()
            result['donchian_middle'] = donchian.donchian_channel_mband()
        except:
            pass
        
        # === VOLUME INDICATORS ===
        
        try:
            obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
            result['obv'] = obv.on_balance_volume()
        except:
            pass
        
        try:
            cmf = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], 
                                           volume=df['Volume'], window=20)
            result['cmf'] = cmf.chaikin_money_flow()
        except:
            pass
        
        try:
            fi = ForceIndexIndicator(close=df['Close'], volume=df['Volume'], window=13)
            result['force_index'] = fi.force_index()
        except:
            pass
        
        try:
            eom = EaseOfMovementIndicator(high=df['High'], low=df['Low'], volume=df['Volume'], window=14)
            result['eom'] = eom.ease_of_movement()
            result['eom_signal'] = eom.sma_ease_of_movement()
        except:
            pass
        
        try:
            vpt = VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'])
            result['vpt'] = vpt.volume_price_trend()
        except:
            pass
        
        try:
            nvi = NegativeVolumeIndexIndicator(close=df['Close'], volume=df['Volume'])
            result['nvi'] = nvi.negative_volume_index()
        except:
            pass
        
        # === MOVING AVERAGES (adjusted for intraday) ===
        
        # On 5-min bars, these periods represent:
        # 20 periods = ~1.5 hours
        # 50 periods = ~4 hours  
        # 200 periods = ~17 trading days
        for period in [5, 10, 20, 50, 100, 200]:
            try:
                result[f'ema{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            except:
                pass
        
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
        
        # === PRICE CHANGES (match your existing schema) ===
        
        # Use the same field names as your original daily-based code
        # These represent changes over N bars (5-min bars, not days)
        for days in [1, 2, 3, 5, 10, 20, 30]:
            try:
                result[f'price_change_{days}d'] = df['Close'].pct_change(days) * 100
            except:
                pass
        
        # === 52-WEEK HIGH/LOW ===
        # Note: On 5-min data, 252 periods = ~21 trading days, not 52 weeks
        # If you want true 52-week data, this should use daily bars
        # For now, keeping calculation consistent with original schema
        
        try:
            result['high_52w'] = df['High'].rolling(window=252, min_periods=1).max()
            result['low_52w'] = df['Low'].rolling(window=252, min_periods=1).min()
            result['price_vs_high_52w'] = (df['Close'] / result['high_52w'] - 1) * 100
            result['price_vs_low_52w'] = (df['Close'] / result['low_52w'] - 1) * 100
        except:
            pass
        
        # === VWAP ===
        
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            result['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
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
        
        # === CANDLESTICK PATTERNS ===
        
        try:
            body = abs(df['Close'] - df['Open'])
            range_hl = df['High'] - df['Low']
            result['doji'] = (body / range_hl < 0.1).astype(int)
            
            lower_shadow = df['Open'].combine(df['Close'], min) - df['Low']
            upper_shadow = df['High'] - df['Open'].combine(df['Close'], max)
            result['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
            
            prev_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
            result['bullish_engulfing'] = ((df['Close'] > df['Open']) & 
                                          (df['Close'].shift(1) < df['Open'].shift(1)) &
                                          (body > prev_body)).astype(int)
        except:
            pass
        
        return result
