#!/usr/bin/env python3
"""
ML Stock Screener & Predictor - FIXED VERSION

PROPERLY fetches T-3, T-5, T-10 indicators with prefixes to match CSV-trained model
Uses DAILY charts (not 5-minute)
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


def log_probability_distribution(predictions_df: pd.DataFrame, logger: logging.Logger, label: str = ""):
    """
    Log a detailed probability distribution histogram.
    This is the primary diagnostic tool for understanding why predictions
    are clustering at low values.

    Args:
        predictions_df: DataFrame with 'explosion_probability' and 'signal' columns
        logger: Logger instance
        label: Optional label prefix for the log section
    """
    if predictions_df.empty:
        logger.warning("Cannot log probability distribution — predictions DataFrame is empty")
        return

    probs = predictions_df['explosion_probability']
    title = f"PROBABILITY DISTRIBUTION{' — ' + label if label else ''}"

    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)

    # Bucket breakdown with ASCII bar chart
    buckets = [
        ("0–10%  (AVOID)",   probs < 0.10),
        ("10–20% (AVOID)",   (probs >= 0.10) & (probs < 0.20)),
        ("20–30% (AVOID)",   (probs >= 0.20) & (probs < 0.30)),
        ("30–40% (AVOID)",   (probs >= 0.30) & (probs < 0.40)),
        ("40–50% (AVOID)",   (probs >= 0.40) & (probs < 0.50)),
        ("50–60% (HOLD)",    (probs >= 0.50) & (probs < 0.60)),
        ("60–70% (HOLD)",    (probs >= 0.60) & (probs < 0.70)),
        ("70–80% (BUY)",     (probs >= 0.70) & (probs < 0.80)),
        ("80–90% (BUY)",     (probs >= 0.80) & (probs < 0.90)),
        ("90–100% (STRONG)", probs >= 0.90),
    ]

    total = len(probs)
    bar_width = 30

    for bucket_label, mask in buckets:
        count = mask.sum()
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 100 * bar_width)
        logger.info(f"  {bucket_label:<22} {count:>4}  ({pct:>5.1f}%)  {bar}")

    logger.info("-" * 70)

    # Summary statistics
    logger.info(f"  Total stocks evaluated:  {total}")
    logger.info(f"  Min probability:         {probs.min():.4f} ({probs.min()*100:.2f}%)")
    logger.info(f"  Max probability:         {probs.max():.4f} ({probs.max()*100:.2f}%)")
    logger.info(f"  Mean probability:        {probs.mean():.4f} ({probs.mean()*100:.2f}%)")
    logger.info(f"  Median probability:      {probs.median():.4f} ({probs.median()*100:.2f}%)")
    logger.info(f"  Std deviation:           {probs.std():.4f}")

    # Signal breakdown
    logger.info("")
    logger.info("  Signal breakdown:")
    if 'signal' in predictions_df.columns:
        signal_counts = predictions_df['signal'].value_counts()
        for signal in ['STRONG BUY', 'BUY', 'HOLD', 'AVOID']:
            count = signal_counts.get(signal, 0)
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"    {signal:<12} {count:>4}  ({pct:>5.1f}%)")

    # Automatic diagnosis based on the distribution shape
    logger.info("")
    logger.info("  ── Diagnosis ──────────────────────────────────────────────────")

    avoid_pct = (probs < 0.50).sum() / total * 100 if total > 0 else 0
    high_pct  = (probs >= 0.70).sum() / total * 100 if total > 0 else 0

    if probs.max() < 0.20:
        logger.warning("  ⚠️  MAX PROB < 20% — likely broken/corrupted model or severe feature mismatch.")
        logger.warning("     Check: (1) is the correct model loaded from archive?")
        logger.warning("            (2) what is the feature coverage % reported above?")

    elif probs.max() < 0.50:
        logger.warning("  ⚠️  MAX PROB < 50% — model is not finding any plausible candidates.")
        logger.warning("     Check: (1) screening population (are winners-type stocks in the 500?)")
        logger.warning("            (2) missing 30% features — are they high-importance ones?")
        logger.warning("            (3) fine-tuning corruption (confirm model restored from archive)")

    elif avoid_pct > 95:
        logger.warning("  ⚠️  >95% AVOID — screening population likely mismatched to training distribution.")
        logger.warning("     The model was trained on stocks that moved 15%+. Consider adding")
        logger.warning("     relative_volume_10d_calc > 1.5 and price < 100 to screener filters.")

    elif high_pct == 0:
        logger.info("  ℹ️  No BUY/STRONG BUY signals today — either a quiet market day or")
        logger.info("     the screening population needs tighter relative-volume / price-range filters.")

    else:
        logger.info(f"  ✅ Distribution looks healthy — {high_pct:.1f}% of stocks scored BUY or higher.")

    logger.info("=" * 70)
    logger.info("")

    # Top 10 by probability for quick review
    logger.info("  Top 10 stocks by probability:")
    logger.info(f"  {'#':<4} {'Symbol':<8} {'Prob':>8}  {'Signal'}")
    logger.info("  " + "-" * 40)
    for rank, (_, row) in enumerate(predictions_df.head(10).iterrows(), 1):
        prob_pct = row['explosion_probability'] * 100
        signal   = row.get('signal', 'N/A')
        symbol   = row.get('symbol', 'N/A')
        logger.info(f"  {rank:<4} {symbol:<8} {prob_pct:>7.2f}%  {signal}")

    logger.info("")


def get_next_trading_day() -> str:
    """
    Get the next NYSE trading day that this prediction is FOR.
    The workflow runs at 3 AM UTC = 10 PM EST (previous calendar day).
    """
    est = pytz.timezone('America/New_York')
    now_est = datetime.now(est)
    prediction_day = now_est + timedelta(days=1)
    while prediction_day.weekday() >= 5:
        prediction_day += timedelta(days=1)
    return prediction_day.date().isoformat()


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
            'min_rsi': None,
            'max_rsi': None,
            'min_volume_ratio': None,
            'trend_filter': None,
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
                {'left': 'close',  'operation': 'greater', 'right': self.filters['min_price']},
                {'left': 'close',  'operation': 'less',    'right': self.filters['max_price']},
                {'left': 'volume', 'operation': 'greater', 'right': self.filters['min_volume']},
            ]
            
            if self.filters.get('min_rsi') is not None:
                filters.append({'left': 'RSI', 'operation': 'greater', 'right': self.filters['min_rsi']})
            
            if self.filters.get('max_rsi') is not None:
                filters.append({'left': 'RSI', 'operation': 'less', 'right': self.filters['max_rsi']})
            
            if self.filters.get('min_volume_ratio') is not None:
                filters.append({
                    'left': 'relative_volume_10d_calc',
                    'operation': 'greater',
                    'right': self.filters['min_volume_ratio']
                })
            
            self.logger.info(f"Applying {len(filters)} filters")
            for f in filters:
                self.logger.debug(f"  Filter: {f}")
            
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


def fetch_stock_data_for_prediction(symbol: str, logger: logging.Logger) -> dict:
    """
    Fetch stock data for prediction.
    - T-3, T-5, T-10: DAILY charts
    - T-1 open/close: 5-MINUTE intraday charts (matches training data)
    """
    import yfinance as yf
    import ta
    
    try:
        ticker = yf.Ticker(symbol)
        
        df_daily = ticker.history(period='90d', interval='1d')
        
        if df_daily.empty or len(df_daily) < 20:
            logger.debug(f"{symbol}: Insufficient daily data")
            return None
        
        df_indicators_daily = calculate_comprehensive_indicators_daily(df_daily)
        
        if df_indicators_daily.empty:
            return None
        
        available_dates = sorted(df_indicators_daily.index.date, reverse=True)
        
        if len(available_dates) < 10:
            logger.debug(f"{symbol}: Need at least 10 trading days, have {len(available_dates)}")
            return None
        
        t3_date  = available_dates[3]  if len(available_dates) > 3  else available_dates[-1]
        t5_date  = available_dates[5]  if len(available_dates) > 5  else available_dates[-1]
        t10_date = available_dates[10] if len(available_dates) > 10 else available_dates[-1]
        
        t3_data  = extract_features_with_prefix(df_indicators_daily, t3_date,  't3',  logger, symbol)
        t5_data  = extract_features_with_prefix(df_indicators_daily, t5_date,  't5',  logger, symbol)
        t10_data = extract_features_with_prefix(df_indicators_daily, t10_date, 't10', logger, symbol)
        
        if not t3_data:
            logger.debug(f"{symbol}: Failed to extract T-3 data")
            return None
        
        df_intraday = ticker.history(period='60d', interval='5m')
        
        if df_intraday.empty or len(df_intraday) < 200:
            logger.debug(f"{symbol}: Insufficient intraday data for T-1 - SKIPPING STOCK")
            return None
        
        df_indicators_intraday = calculate_comprehensive_indicators_intraday(df_intraday)
        
        if df_indicators_intraday.empty:
            logger.debug(f"{symbol}: Failed to calculate intraday indicators")
            return None
        
        if df_indicators_intraday.index.tz is None:
            df_indicators_intraday.index = df_indicators_intraday.index.tz_localize('America/New_York')
        else:
            df_indicators_intraday.index = df_indicators_intraday.index.tz_convert('America/New_York')
        
        yesterday = available_dates[1] if len(available_dates) > 1 else available_dates[-1]
        
        t1_close_data = extract_intraday_snapshot(
            df_indicators_intraday, yesterday, dt_time(16, 0), 't1_close', logger, symbol
        )
        t1_open_data = extract_intraday_snapshot(
            df_indicators_intraday, yesterday, dt_time(9, 30), 't1_open', logger, symbol
        )
        
        if not t1_close_data or not t1_open_data:
            logger.debug(f"{symbol}: Failed to extract T-1 intraday snapshots, falling back to daily")
            t1_date = available_dates[1] if len(available_dates) > 1 else available_dates[-1]
            t1_close_data = extract_features_with_prefix(df_indicators_daily, t1_date, 't1_close', logger, symbol)
            t1_open_data  = extract_features_with_prefix(df_indicators_daily, t1_date, 't1_open',  logger, symbol)
        
        result = {
            'symbol': symbol,
            'exchange': 'NASDAQ',
            **t3_data,
            **t5_data,
            **t10_data,
            **t1_close_data,
            **t1_open_data
        }
        
        logger.debug(f"{symbol}: {len(result)} total features assembled")
        return result
        
    except Exception as e:
        logger.debug(f"{symbol}: Error - {e}")
        return None


def extract_features_with_prefix(df: pd.DataFrame, date, prefix: str, logger, symbol: str) -> dict:
    """Extract indicators with prefix (e.g., t3_, t5_, t10_) from DAILY data"""
    day_bars = df[df.index.date == date]
    if day_bars.empty:
        logger.debug(f"{symbol}: No data for {date} (prefix {prefix})")
        return {}
    bar = day_bars.iloc[-1]
    return {
        f"{prefix}_{k}": (v if (pd.notna(v) and not np.isinf(v)) else None)
        for k, v in bar.to_dict().items()
    }


def extract_intraday_snapshot(
    df_intraday: pd.DataFrame,
    target_date,
    target_time,
    prefix: str,
    logger,
    symbol: str
) -> dict:
    """Extract indicators from 5-MINUTE intraday data at specific time"""
    day_bars = df_intraday[df_intraday.index.date == target_date]
    if day_bars.empty:
        logger.debug(f"{symbol}: No intraday data for {target_date}")
        return {}
    
    window_start = (datetime.combine(target_date, target_time) - timedelta(minutes=5)).time()
    window_end   = (datetime.combine(target_date, target_time) + timedelta(minutes=30)).time()
    
    target_bars = day_bars[
        (day_bars.index.time >= window_start) &
        (day_bars.index.time <= window_end)
    ]
    if target_bars.empty:
        target_bars = day_bars
    
    bar = target_bars.iloc[0] if target_time.hour < 12 else target_bars.iloc[-1]
    
    return {
        f"{prefix}_{k}": (v if (pd.notna(v) and not np.isinf(v)) else None)
        for k, v in bar.to_dict().items()
    }


def calculate_comprehensive_indicators_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators on DAILY bars"""
    import ta
    
    result = pd.DataFrame(index=df.index)
    result['Close']  = df['Close']
    result['Open']   = df['Open']
    result['High']   = df['High']
    result['Low']    = df['Low']
    result['Volume'] = df['Volume']
    
    for period in [5, 10, 20, 50]:
        try: result[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
        except: pass
    
    for period in [5, 10, 12, 20, 26, 50]:
        try: result[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        except: pass
    
    try:
        result['WMA_10'] = ta.trend.wma_indicator(df['Close'], window=10)
        result['WMA_20'] = ta.trend.wma_indicator(df['Close'], window=20)
    except: pass
    
    try:
        result['HMA_9']  = ta.trend.wma_indicator(df['Close'], window=9)
        result['HMA_20'] = ta.trend.wma_indicator(df['Close'], window=20)
    except: pass
    
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        result['VWMA_20'] = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    except: pass
    
    try:
        result['Price_vs_SMA20'] = (df['Close'] / result['SMA_20'] - 1) * 100
        result['Price_vs_SMA50'] = (df['Close'] / result['SMA_50'] - 1) * 100
        result['Price_vs_EMA20'] = (df['Close'] / result['EMA_20'] - 1) * 100
    except: pass
    
    try:
        result['SMA_20_50_Diff'] = result['SMA_20'] - result['SMA_50']
        result['EMA_12_26_Diff'] = result['EMA_12'] - result['EMA_26']
    except: pass
    
    try:
        result['SMA_20_Slope'] = result['SMA_20'].diff(5)
        result['EMA_20_Slope'] = result['EMA_20'].diff(5)
    except: pass
    
    for period in [7, 14, 21, 28]:
        try: result[f'RSI_{period}'] = ta.momentum.rsi(df['Close'], window=period)
        except: pass
    
    try: result['RSI_14_Slope'] = result['RSI_14'].diff(3)
    except: pass
    
    try:
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        result['STOCHk_14_3_3'] = stoch.stoch()
        result['STOCHd_14_3_3'] = stoch.stoch_signal()
        result['STOCHh_14_3_3'] = result['STOCHk_14_3_3'] - result['STOCHd_14_3_3']
        stoch_fast = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=5, smooth_window=1)
        result['STOCHk_5_3_1'] = stoch_fast.stoch()
        result['STOCHd_5_3_1'] = stoch_fast.stoch_signal()
        result['STOCHh_5_3_1'] = result['STOCHk_5_3_1'] - result['STOCHd_5_3_1']
    except: pass
    
    try:
        stoch_rsi = ta.momentum.StochRSIIndicator(df['Close'], window=14, smooth1=3, smooth2=3)
        result['STOCHRSIk_14_14_3_3'] = stoch_rsi.stochrsi_k()
        result['STOCHRSId_14_14_3_3'] = stoch_rsi.stochrsi_d()
    except: pass
    
    try: result['WILLR_14'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    except: pass
    
    for period in [14, 20]:
        try: result[f'CCI_{period}'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=period)
        except: pass
    
    try: result['UO'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
    except: pass
    
    try: result['AO'] = ta.momentum.awesome_oscillator(df['High'], df['Low'], window1=5, window2=34)
    except: pass
    
    try:
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        result['MACD_12_26_9']  = macd.macd()
        result['MACDh_12_26_9'] = macd.macd_diff()
        result['MACDs_12_26_9'] = macd.macd_signal()
        result['MACD_ROC']      = result['MACD_12_26_9'].pct_change(5) * 100
        macd_fast = ta.trend.MACD(df['Close'], window_slow=12, window_fast=6, window_sign=5)
        result['MACD_Fast']  = macd_fast.macd()
        result['MACDh_Fast'] = macd_fast.macd_diff()
        result['MACDs_Fast'] = macd_fast.macd_signal()
    except: pass
    
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        result['BBL_20_2.0_2.0'] = bb.bollinger_lband()
        result['BBM_20_2.0_2.0'] = bb.bollinger_mavg()
        result['BBU_20_2.0_2.0'] = bb.bollinger_hband()
        result['BBB_20_2.0_2.0'] = bb.bollinger_wband()
        result['BBP_20_2.0_2.0'] = bb.bollinger_pband()
    except: pass
    
    try:
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        result['KCLe_20_2'] = keltner.keltner_channel_lband()
        result['KCBe_20_2'] = keltner.keltner_channel_mband()
        result['KCUe_20_2'] = keltner.keltner_channel_hband()
    except: pass
    
    try:
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
        result['DCL_20_20'] = donchian.donchian_channel_lband()
        result['DCM_20_20'] = donchian.donchian_channel_mband()
        result['DCU_20_20'] = donchian.donchian_channel_hband()
    except: pass
    
    for period in [7, 14, 20]:
        try: result[f'ATR_{period}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
        except: pass
    
    try: result['ATR_14_Slope'] = result['ATR_14'].diff(5)
    except: pass
    
    for period in [10, 20, 30]:
        try: result[f'HV_{period}'] = df['Close'].pct_change().rolling(window=period).std() * 100
        except: pass
    
    try:
        result['OBV']      = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        result['OBV_SMA20'] = result['OBV'].rolling(window=20).mean()
    except: pass
    
    for period in [5, 10, 20]:
        try: result[f'Volume_MA{period}'] = df['Volume'].rolling(window=period).mean()
        except: pass
    
    try: result['Volume_Ratio'] = df['Volume'] / result['Volume_MA20']
    except: pass
    
    try:
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        result['ADX_14']   = adx.adx()
        result['ADXR_14_2'] = adx.adx()
        result['DMP_14']   = adx.adx_pos()
        result['DMN_14']   = adx.adx_neg()
    except: pass
    
    try:
        aroon = ta.trend.AroonIndicator(df['Close'], window=25)
        result['AROONU_25']   = aroon.aroon_up()
        result['AROOND_25']   = aroon.aroon_down()
        result['AROONOSC_25'] = aroon.aroon_indicator()
    except: pass
    
    try:
        result['SUPERT_10_3']  = df['Close']
        result['SUPERTd_10_3'] = 0
        result['SUPERTl_10_3'] = df['Low']
        result['SUPERTs_10_3'] = 1
    except: pass
    
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        result['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    except: pass
    
    try: result['MFI_14'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except: pass
    
    try: result['CMF_20'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    except: pass
    
    for period in [10, 20]:
        try: result[f'ROC_{period}'] = ta.momentum.roc(df['Close'], window=period)
        except: pass
    
    for period in [10, 20]:
        try: result[f'MOM_{period}'] = df['Close'].diff(period)
        except: pass
    
    try:
        tsi = ta.momentum.TSIIndicator(df['Close'], window_slow=25, window_fast=13)
        result['TSI_13_25_13']  = tsi.tsi()
        result['TSIs_13_25_13'] = tsi.tsi()
    except: pass
    
    try: result['Gap_Pct'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
    except: pass
    
    return result


def calculate_comprehensive_indicators_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators on 5-MINUTE intraday bars"""
    import ta
    
    result = pd.DataFrame(index=df.index)
    result['Close']  = df['Close']
    result['Open']   = df['Open']
    result['High']   = df['High']
    result['Low']    = df['Low']
    result['Volume'] = df['Volume']
    
    for period in [5, 10, 20, 50]:
        try: result[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
        except: pass
    
    for period in [5, 10, 12, 20, 26, 50]:
        try: result[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        except: pass
    
    for period in [7, 14, 21, 28]:
        try: result[f'RSI_{period}'] = ta.momentum.rsi(df['Close'], window=period)
        except: pass
    
    try:
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        result['MACD_12_26_9']  = macd.macd()
        result['MACDh_12_26_9'] = macd.macd_diff()
        result['MACDs_12_26_9'] = macd.macd_signal()
    except: pass
    
    for period in [7, 14, 20]:
        try: result[f'ATR_{period}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
        except: pass
    
    try:
        result['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        result['OBV_SMA20'] = result['OBV'].rolling(window=20).mean()
    except: pass
    
    for period in [5, 10, 20]:
        try: result[f'Volume_MA{period}'] = df['Volume'].rolling(window=period).mean()
        except: pass
    
    try: result['Volume_Ratio'] = df['Volume'] / result['Volume_MA20']
    except: pass
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    prediction_date = get_next_trading_day()
    
    logger.info("=" * 80)
    logger.info("ML SCREENING & PREDICTION")
    logger.info("=" * 80)
    logger.info(f"Prediction date (trading session): {prediction_date}")
    logger.info("=" * 80)
    
    screener = SmartScreener(logger=logger)
    
    try:
        predictor = ExplosionPredictor()
        supabase  = MLPredictionSupabaseClient({})
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # STEP 1: SCREENING
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: INTELLIGENT SCREENING")
    logger.info("=" * 80)
    
    screened_df = screener.screen_with_tradingview(max_results=args.max_results)
    if screened_df.empty:
        logger.error("No stocks passed screening")
        return 1
    logger.info(f"✓ Screened {len(screened_df)} stocks")
    
    symbols = []
    if 'symbol' in screened_df.columns:
        symbols = screened_df['symbol'].str.split(':').str[-1].tolist()
    else:
        logger.error("No symbol column in screened results")
        return 1
    
    # STEP 2: FETCH STOCK DATA
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: FETCH STOCK DATA")
    logger.info("=" * 80)
    logger.info(f"Fetching data for {len(symbols)} stocks...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    enriched_stocks = []
    failed_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_data_for_prediction, sym, logger): sym
                   for sym in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(symbols)} | successful: {len(enriched_stocks)}")
            result = future.result()
            if result:
                enriched_stocks.append(result)
            else:
                failed_count += 1
    
    logger.info(f"✓ Fetched {len(enriched_stocks)} stocks ({failed_count} failed/skipped)")
    
    if not enriched_stocks:
        logger.error("Failed to fetch data for any stocks")
        return 1
    
    # STEP 3: PREPARE FEATURES
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: PREPARE FEATURES")
    logger.info("=" * 80)
    
    features_df = pd.DataFrame(enriched_stocks)
    logger.info(f"✓ Feature matrix: {len(features_df)} stocks × {len(features_df.columns)} raw columns")
    
    # STEP 4: ML PREDICTION
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ML PREDICTION")
    logger.info("=" * 80)
    
    historical_gains = supabase.get_historical_prediction_accuracy(days_back=30)
    
    try:
        predictions_df = predictor.predict_with_targets(features_df, historical_gains)
        logger.info(f"✓ Generated {len(predictions_df)} predictions")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ── PROBABILITY DISTRIBUTION ─────────────────────────────────────────────
    log_probability_distribution(
        predictions_df, logger,
        label=f"All {len(predictions_df)} screened stocks"
    )
    # ─────────────────────────────────────────────────────────────────────────
    
    # STEP 5: TOP PREDICTIONS
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 5: TOP {args.top_n} PREDICTIONS")
    logger.info("=" * 80)
    
    top_predictions = predictions_df.head(args.top_n)
    
    logger.info(f"\nTop {min(20, len(top_predictions))} Predictions for {prediction_date}:")
    logger.info("-" * 100)
    logger.info(f"{'#':<4} {'Symbol':<8} {'Signal':<13} {'Prob':<8} {'Price':<10} {'Target':<10} {'Gain':<8}")
    logger.info("-" * 100)
    
    for rank, (_, row) in enumerate(top_predictions.head(20).iterrows(), 1):
        current_price = row.get('current_price', 0)
        logger.info(
            f"{rank:<4} {row['symbol']:<8} {row['signal']:<13} "
            f"{row['explosion_probability']*100:>6.2f}%  "
            f"${current_price:>8.2f}  "
            f"${row.get('target_price', 0):>8.2f}  "
            f"+{row.get('target_gain_pct', 0):>5.1f}%"
        )
    
    # STEP 6: STORE PREDICTIONS
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: STORE PREDICTIONS")
    logger.info("=" * 80)
    
    predictions_list = [
        {
            'symbol':               row['symbol'],
            'exchange':             row.get('exchange', 'NASDAQ'),
            'prediction_date':      prediction_date,
            'explosion_probability': float(row['explosion_probability']),
            'prediction':           int(row['prediction']),
            'signal':               row['signal'],
            'target_gain_pct':      float(row.get('target_gain_pct', 0)),
            'target_gain_low':      float(row.get('target_gain_low', 0)),
            'target_gain_high':     float(row.get('target_gain_high', 0)),
            'current_price':        float(row.get('current_price', 0)),
            'target_price':         float(row.get('target_price', 0)),
            'target_price_low':     float(row.get('target_price_low', 0)),
            'target_price_high':    float(row.get('target_price_high', 0)),
        }
        for _, row in top_predictions.iterrows()
    ]
    
    if predictions_list:
        count = supabase.write_predictions(predictions_list)
        logger.info(f"✓ Wrote {count} predictions for trading session: {prediction_date}")
    
    # STEP 7: SCREENING LOG
    screening_log = {
        'screening_date':             prediction_date,
        'total_symbols_attempted':    args.max_results,
        'symbols_fetched_successfully': len(screened_df),
        'symbols_after_all_filters':  len(features_df),
        'total_predictions':          len(predictions_df),
        'strong_buy_count':  len(predictions_df[predictions_df['signal'] == 'STRONG BUY']),
        'buy_count':         len(predictions_df[predictions_df['signal'] == 'BUY']),
        'hold_count':        len(predictions_df[predictions_df['signal'] == 'HOLD']),
        'avoid_count':       len(predictions_df[predictions_df['signal'] == 'AVOID']),
        'avg_probability':    float(predictions_df['explosion_probability'].mean()),
        'max_probability':    float(predictions_df['explosion_probability'].max()),
        'min_probability':    float(predictions_df['explosion_probability'].min()),
        'median_probability': float(predictions_df['explosion_probability'].median()),
        'model_version':      'csv_trained_with_t3_t5_t10_prefixes'
    }
    supabase.write_screening_log(screening_log)
    
    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ PREDICTION COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
