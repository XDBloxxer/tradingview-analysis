#!/usr/bin/env python3
"""
ML Stock Screener & Predictor - MULTI-TIMEPOINT VERSION

This version:
1. Fetches ALL timepoints (T-0 open/close, T-1 open/close, T-3, T-5, T-10)
2. Preserves timepoint prefixes (t0_open_rsi, t1_close_macd, etc.)
3. Feeds multi-timepoint features to model
4. Model learns temporal patterns across all timepoints
"""

import argparse
import logging
import pytz
from datetime import time as dt_time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.explosion_predictor import ExplosionPredictor
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient

try:
    from tradingview_scraper.symbols.screener import Screener
    SCREENER_AVAILABLE = True
except ImportError:
    SCREENER_AVAILABLE = False


def setup_logging(verbose: bool = False):
    """Setup basic logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class SmartScreener:
    """Intelligent screener that learns optimal filters over time"""
    
    def __init__(self, config: dict = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.filters = self._load_learned_filters()
        
        if SCREENER_AVAILABLE:
            self.screener = Screener()
        else:
            self.screener = None
    
    def _load_learned_filters(self) -> dict:
        """Load learned filters from previous model training"""
        defaults = {
            'min_price': 0.50,
            'max_price': 500.0,
            'min_volume': 100000,
        }
        
        try:
            filter_path = Path('ml_models/learned_filters.json')
            if filter_path.exists():
                with open(filter_path, 'r') as f:
                    learned = json.load(f)
                for key, value in learned.items():
                    if value is not None:
                        defaults[key] = value
                self.logger.info("✓ Loaded learned screening filters")
        except Exception as e:
            self.logger.debug(f"Using default filters: {e}")
        
        return defaults
    
    def screen_with_tradingview(self, max_results: int = 500) -> pd.DataFrame:
        """Screen stocks using TradingView with learned filters"""
        if not SCREENER_AVAILABLE or self.screener is None:
            self.logger.error("TradingView screener not available!")
            return pd.DataFrame()
        
        self.logger.info("Screening stocks with learned filters...")
        
        try:
            filters = [
                {'left': 'close', 'operation': 'greater', 'right': self.filters['min_price']},
                {'left': 'close', 'operation': 'less', 'right': self.filters['max_price']},
                {'left': 'volume', 'operation': 'greater', 'right': self.filters['min_volume']},
            ]
            
            result = self.screener.screen(
                market='america',
                filters=filters,
                sort_by='volume',
                sort_order='desc',
                limit=max_results
            )
            
            if result['status'] == 'success' and result.get('data'):
                df = pd.DataFrame(result['data'])
                self.logger.info(f"✓ Screened {len(df)} stocks")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return pd.DataFrame()


def fetch_complete_timepoint_data(symbol: str, logger: logging.Logger) -> dict:
    """
    Fetch ALL timepoint data for multi-timepoint model
    
    Returns dict with ALL timepoints preserved:
    {
        'symbol': 'AAPL',
        'exchange': 'NASDAQ',
        't0_open': {indicators},
        't0_close': {indicators},
        't1_open': {indicators},
        't1_close': {indicators},
        't3': {indicators},
        't5': {indicators},
        't10': {indicators}
    }
    """
    import yfinance as yf
    import ta
    
    try:
        ticker = yf.Ticker(symbol)
        nyc_tz = pytz.timezone('America/New_York')
        
        # Fetch 5-minute data for T-0 and T-1
        df_5min = ticker.history(period='60d', interval='5m')
        
        if df_5min.empty or len(df_5min) < 200:
            logger.debug(f"{symbol}: Insufficient 5-minute data")
            return None
        
        # Normalize timezone
        if df_5min.index.tz is None:
            df_5min.index = df_5min.index.tz_localize('America/New_York')
        else:
            df_5min.index = df_5min.index.tz_convert('America/New_York')
        
        # Calculate indicators on 5-minute data
        df_5min_indicators = calculate_indicators_5min(df_5min)
        
        # Get available trading days
        available_dates_5min = sorted(list(set(df_5min_indicators.index.date)), reverse=True)
        
        if len(available_dates_5min) < 2:
            return None
        
        t0_date = available_dates_5min[0]
        t1_date = available_dates_5min[1]
        
        # Extract T-0 and T-1 snapshots
        t0_open = extract_timepoint_5min(df_5min_indicators, t0_date, 'open', logger, symbol)
        t0_close = extract_timepoint_5min(df_5min_indicators, t0_date, 'close', logger, symbol)
        t1_open = extract_timepoint_5min(df_5min_indicators, t1_date, 'open', logger, symbol)
        t1_close = extract_timepoint_5min(df_5min_indicators, t1_date, 'close', logger, symbol)
        
        if not all([t0_open, t0_close, t1_open, t1_close]):
            return None
        
        # Fetch daily data for T-3, T-5, T-10
        df_daily = ticker.history(period='90d', interval='1d')
        
        if df_daily.empty or len(df_daily) < 20:
            return None
        
        df_daily_indicators = calculate_indicators_daily(df_daily)
        available_dates_daily = sorted(df_daily_indicators.index.date, reverse=True)
        
        if t0_date not in available_dates_daily:
            return None
        
        t0_idx = available_dates_daily.index(t0_date)
        
        snapshots = {}
        
        # T-3, T-5, T-10
        for lag_name, lag_days in [('t3', 3), ('t5', 5), ('t10', 10)]:
            if t0_idx + lag_days < len(available_dates_daily):
                lag_date = available_dates_daily[t0_idx + lag_days]
                lag_data = extract_timepoint_daily(df_daily_indicators, lag_date, logger, symbol)
                if lag_data:
                    snapshots[lag_name] = lag_data
        
        # Return ALL timepoints
        result = {
            'symbol': symbol,
            'exchange': 'NASDAQ',
            't0_open': t0_open,
            't0_close': t0_close,
            't1_open': t1_open,
            't1_close': t1_close,
            **snapshots
        }
        
        logger.debug(f"{symbol}: Fetched {len(result)-2} timepoints")
        
        return result
        
    except Exception as e:
        logger.debug(f"{symbol}: Error - {e}")
        return None


def extract_timepoint_5min(df: pd.DataFrame, date, timepoint: str, logger, symbol: str) -> dict:
    """Extract indicators from 5-min data at specific timepoint"""
    
    day_bars = df[df.index.date == date]
    
    if day_bars.empty:
        return None
    
    if timepoint == 'open':
        target_bars = day_bars[(day_bars.index.time >= dt_time(9, 30)) & 
                               (day_bars.index.time <= dt_time(10, 0))]
        if target_bars.empty:
            target_bars = day_bars[day_bars.index.time >= dt_time(9, 30)]
        if not target_bars.empty:
            bar = target_bars.iloc[0]
        else:
            return None
    else:  # close
        target_bars = day_bars[(day_bars.index.time >= dt_time(15, 55)) & 
                               (day_bars.index.time <= dt_time(16, 0))]
        if target_bars.empty:
            target_bars = day_bars[day_bars.index.time >= dt_time(15, 0)]
        if not target_bars.empty:
            bar = target_bars.iloc[-1]
        else:
            return None
    
    return {k.lower(): (v if pd.notna(v) and not np.isinf(v) else None) 
            for k, v in bar.to_dict().items()}


def extract_timepoint_daily(df: pd.DataFrame, date, logger, symbol: str) -> dict:
    """Extract indicators from daily data"""
    
    day_bars = df[df.index.date == date]
    
    if day_bars.empty:
        return None
    
    bar = day_bars.iloc[-1]
    
    return {k.lower(): (v if pd.notna(v) and not np.isinf(v) else None) 
            for k, v in bar.to_dict().items()}


def calculate_indicators_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators on 5-minute bars"""
    import ta
    
    result = pd.DataFrame(index=df.index)
    
    # Basic OHLCV
    result['close'] = df['Close']
    result['open'] = df['Open']
    result['high'] = df['High']
    result['low'] = df['Low']
    result['volume'] = df['Volume']
    
    # Momentum
    try:
        result['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        result['stoch.k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        result['stoch.d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        result['ao'] = ta.momentum.awesome_oscillator(df['High'], df['Low'], window1=5, window2=34)
        result['uo'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
    except: pass
    
    # Trend
    try:
        result['macd.macd'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        result['macd.signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        result['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        result['cci20'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    except: pass
    
    # Volatility
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        result['bb.upper'] = bb.bollinger_hband()
        result['bb.lower'] = bb.bollinger_lband()
        result['bb.middle'] = bb.bollinger_mavg()
        result['bb_width'] = (result['bb.upper'] - result['bb.lower']) / result['bb.middle'] * 100
        
        result['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        result['keltner_upper'] = keltner.keltner_channel_hband()
        result['keltner_lower'] = keltner.keltner_channel_lband()
        
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
        result['donchian_upper'] = donchian.donchian_channel_hband()
        result['donchian_lower'] = donchian.donchian_channel_lband()
        result['donchian_middle'] = donchian.donchian_channel_mband()
    except: pass
    
    # Volume
    try:
        result['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        result['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    except: pass
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        try:
            result[f'ema{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
            result[f'sma{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
        except: pass
    
    # VWAP
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        result['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    except: pass
    
    # Volatility measures
    try:
        result['volatility_10d'] = df['Close'].pct_change().rolling(window=10).std() * 100
        result['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * 100
        result['volatility_30d'] = df['Close'].pct_change().rolling(window=30).std() * 100
    except: pass
    
    return result


def calculate_indicators_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators on daily bars (for T-3, T-5, T-10)"""
    import ta
    
    result = pd.DataFrame(index=df.index)
    
    # Basic OHLCV
    result['close'] = df['Close']
    result['open'] = df['Open']
    result['high'] = df['High']
    result['low'] = df['Low']
    result['volume'] = df['Volume']
    
    # Momentum
    try:
        result['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        result['stoch.k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        result['stoch.d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        result['ao'] = ta.momentum.awesome_oscillator(df['High'], df['Low'], window1=5, window2=34)
        result['uo'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
    except: pass
    
    # Trend
    try:
        result['macd.macd'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        result['macd.signal'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        result['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        result['cci20'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    except: pass
    
    # Volatility
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        result['bb.upper'] = bb.bollinger_hband()
        result['bb.lower'] = bb.bollinger_lband()
        result['bb.middle'] = bb.bollinger_mavg()
        result['bb_width'] = (result['bb.upper'] - result['bb.lower']) / result['bb.middle'] * 100
        
        result['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        result['keltner_upper'] = keltner.keltner_channel_hband()
        result['keltner_lower'] = keltner.keltner_channel_lband()
        
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
        result['donchian_upper'] = donchian.donchian_channel_hband()
        result['donchian_lower'] = donchian.donchian_channel_lband()
        result['donchian_middle'] = donchian.donchian_channel_mband()
    except: pass
    
    # Volume
    try:
        result['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        result['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    except: pass
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        try:
            result[f'ema{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
            result[f'sma{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
        except: pass
    
    # VWAP
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        result['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    except: pass
    
    # Volatility measures
    try:
        result['volatility_10d'] = df['Close'].pct_change().rolling(window=10).std() * 100
        result['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * 100
        result['volatility_30d'] = df['Close'].pct_change().rolling(window=30).std() * 100
    except: pass
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    logger.info("="*80)
    logger.info("ML SCREENING & PREDICTION - MULTI-TIMEPOINT SYSTEM")
    logger.info("="*80)
    logger.info("Fetching ALL timepoints: T-0 open/close, T-1 open/close, T-3, T-5, T-10")
    logger.info("Model will learn temporal patterns across all timepoints")
    logger.info("="*80)
    
    # Initialize
    screener = SmartScreener(logger=logger)
    
    try:
        predictor = ExplosionPredictor()
        supabase = MLPredictionSupabaseClient({})
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # STEP 1: SCREENING
    logger.info("\n" + "="*80)
    logger.info("STEP 1: INTELLIGENT SCREENING")
    logger.info("="*80)
    
    screened_df = screener.screen_with_tradingview(max_results=args.max_results)
    
    if screened_df.empty:
        logger.error("No stocks passed screening")
        return 1
    
    logger.info(f"✓ Screened {len(screened_df)} stocks")
    
    # Extract symbols
    symbols = []
    if 'symbol' in screened_df.columns:
        symbols = screened_df['symbol'].str.split(':').str[-1].tolist()
    else:
        logger.error("No symbol column")
        return 1
    
    # STEP 2: FETCH COMPLETE TIMEPOINT DATA
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FETCH ALL TIMEPOINT DATA")
    logger.info("="*80)
    logger.info(f"Fetching data for {len(symbols)} stocks...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    enriched_stocks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_complete_timepoint_data, sym, logger): sym 
                  for sym in symbols}
        
        for i, future in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(symbols)}")
            
            result = future.result()
            if result:
                enriched_stocks.append(result)
    
    if not enriched_stocks:
        logger.error("Failed to fetch data")
        return 1
    
    logger.info(f"✓ Fetched complete data for {len(enriched_stocks)} stocks")
    
    # STEP 3: PREPARE MULTI-TIMEPOINT FEATURES
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PREPARE MULTI-TIMEPOINT FEATURES")
    logger.info("="*80)
    
    features_list = []
    for stock in enriched_stocks:
        feature_row = {
            'symbol': stock['symbol'], 
            'exchange': stock['exchange']
        }
        
        # Add ALL timepoint features WITH PREFIXES
        for timepoint in ['t0_open', 't0_close', 't1_open', 't1_close', 't3', 't5', 't10']:
            if timepoint in stock:
                for k, v in stock[timepoint].items():
                    feature_row[f'{timepoint}_{k}'] = v
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"✓ Prepared {len(features_df)} stocks with {len(features_df.columns)} multi-timepoint features")
    
    # STEP 4: ML PREDICTION
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MULTI-TIMEPOINT ML PREDICTION")
    logger.info("="*80)
    
    historical_gains = supabase.get_historical_prediction_accuracy(days_back=30)
    
    try:
        predictions_df = predictor.predict_with_targets(features_df, historical_gains)
        logger.info(f"✓ Generated {len(predictions_df)} predictions")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # STEP 5: TOP PREDICTIONS
    logger.info("\n" + "="*80)
    logger.info(f"STEP 5: TOP {args.top_n} PREDICTIONS")
    logger.info("="*80)
    
    top_predictions = predictions_df.head(args.top_n)
    
    logger.info(f"\nTop {min(20, len(top_predictions))} Predictions:")
    logger.info("-" * 100)
    logger.info(f"{'#':<4} {'Symbol':<8} {'Signal':<13} {'Prob':<8} {'Price':<10} {'Target':<10} {'Gain':<8}")
    logger.info("-" * 100)
    
    for idx, row in top_predictions.head(20).iterrows():
        current_price = row.get('current_price', 0)
        if current_price == 0:
            # Try to get from any close feature
            for col in row.index:
                if 'close' in col and row[col] > 0:
                    current_price = row[col]
                    break
        
        logger.info(
            f"{idx+1:<4} {row['symbol']:<8} {row['signal']:<13} "
            f"{row['explosion_probability']*100:>6.2f}%  "
            f"${current_price:>8.2f}  "
            f"${row.get('target_price', 0):>8.2f}  "
            f"+{row.get('target_gain_pct', 0):>5.1f}%"
        )
    
    # STEP 6: STORE PREDICTIONS
    logger.info("\n" + "="*80)
    logger.info("STEP 6: STORE PREDICTIONS")
    logger.info("="*80)
    
    prediction_date = datetime.now().date().isoformat()
    predictions_list = []
    
    for _, row in top_predictions.iterrows():
        current_price = row.get('current_price', 0)
        if current_price == 0:
            for col in row.index:
                if 'close' in col and row[col] > 0:
                    current_price = row[col]
                    break
        
        prediction_record = {
            'symbol': row['symbol'],
            'exchange': row.get('exchange', 'NASDAQ'),
            'prediction_date': prediction_date,
            'explosion_probability': float(row['explosion_probability']),
            'prediction': int(row['prediction']),
            'signal': row['signal'],
            'target_gain_pct': float(row.get('target_gain_pct', 0)),
            'target_gain_low': float(row.get('target_gain_low', 0)),
            'target_gain_high': float(row.get('target_gain_high', 0)),
            'current_price': float(current_price),
            'target_price': float(row.get('target_price', 0)),
            'target_price_low': float(row.get('target_price_low', 0)),
            'target_price_high': float(row.get('target_price_high', 0)),
        }
        
        predictions_list.append(prediction_record)
    
    if predictions_list:
        count = supabase.write_predictions(predictions_list)
        logger.info(f"✓ Wrote {count} predictions")
    
    # STEP 7: LOG STATISTICS
    screening_log = {
        'screening_date': prediction_date,
        'total_symbols_attempted': args.max_results,
        'symbols_fetched_successfully': len(screened_df),
        'symbols_after_all_filters': len(features_df),
        'total_predictions': len(predictions_df),
        'strong_buy_count': len(predictions_df[predictions_df['signal'] == 'STRONG BUY']),
        'buy_count': len(predictions_df[predictions_df['signal'] == 'BUY']),
        'hold_count': len(predictions_df[predictions_df['signal'] == 'HOLD']),
        'avoid_count': len(predictions_df[predictions_df['signal'] == 'AVOID']),
        'avg_probability': float(predictions_df['explosion_probability'].mean()),
        'model_version': 'xgboost_multi_timepoint_v1'
    }
    
    supabase.write_screening_log(screening_log)
    
    # Export CSV
    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PREDICTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Predictions for: {prediction_date}")
    logger.info(f"Model type: MULTI-TIMEPOINT (learns from T-0, T-1, T-3, T-5, T-10)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
