#!/usr/bin/env python3
"""
ML Stock Screener & Predictor - FIXED VERSION

CRITICAL FIX: This version extracts features as FLAT (no prefixes) to match 
the CSV-trained model which expects features like: Close, RSI_14, MACD_12_26_9

Uses T-3 as primary timepoint (model was trained on T-3, T-5, T-10 from CSV)
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


def fetch_stock_data_for_prediction(symbol: str, logger: logging.Logger) -> dict:
    """
    Fetch stock data for prediction (T-3 primarily, T-5/T-10 as fallback)
    Returns FLAT features (no prefixes) to match CSV-trained model
    
    Returns dict:
    {
        'symbol': 'AAPL',
        'exchange': 'NASDAQ',
        'Close': 150.0,      # Flat features (no t3_ prefix)
        'RSI_14': 65.0,
        'MACD_12_26_9': 2.5,
        ...
    }
    """
    import yfinance as yf
    import ta
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Fetch daily data (for T-3, T-5, T-10)
        df_daily = ticker.history(period='90d', interval='1d')
        
        if df_daily.empty or len(df_daily) < 20:
            logger.debug(f"{symbol}: Insufficient daily data")
            return None
        
        # Calculate indicators on daily data
        df_indicators = calculate_comprehensive_indicators(df_daily)
        
        if df_indicators.empty:
            return None
        
        # Get available trading days
        available_dates = sorted(df_indicators.index.date, reverse=True)
        
        if len(available_dates) < 3:
            return None
        
        # Get T-3 (3 days ago) - PRIMARY TIMEPOINT
        t3_date = available_dates[3] if len(available_dates) > 3 else available_dates[-1]
        
        # Extract T-3 data as FLAT features (no prefix)
        t3_data = extract_flat_features(df_indicators, t3_date, logger, symbol)
        
        if not t3_data:
            return None
        
        # Add metadata
        result = {
            'symbol': symbol,
            'exchange': 'NASDAQ',
            **t3_data  # Flat features: Close, RSI_14, MACD, etc.
        }
        
        logger.debug(f"{symbol}: Fetched T-3 data with {len(t3_data)} flat features")
        
        return result
        
    except Exception as e:
        logger.debug(f"{symbol}: Error - {e}")
        return None


def extract_flat_features(df: pd.DataFrame, date, logger, symbol: str) -> dict:
    """Extract indicators as FLAT features (matching CSV structure)"""
    
    day_bars = df[df.index.date == date]
    
    if day_bars.empty:
        return None
    
    bar = day_bars.iloc[-1]
    
    # Return flat features (NO prefixes like t3_)
    return {k: (v if pd.notna(v) and not np.isinf(v) else None) 
            for k, v in bar.to_dict().items()}


def calculate_comprehensive_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate indicators matching CSV column names
    This should produce features like: Close, RSI_14, MACD_12_26_9, etc.
    """
    import ta
    
    result = pd.DataFrame(index=df.index)
    
    # Basic OHLCV (matching CSV names)
    result['Close'] = df['Close']
    result['Open'] = df['Open']
    result['High'] = df['High']
    result['Low'] = df['Low']
    result['Volume'] = df['Volume']
    
    # SMAs
    for period in [5, 10, 20, 50]:
        result[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
    
    # EMAs
    for period in [5, 10, 12, 20, 26, 50]:
        result[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
    
    # WMAs
    try:
        result['WMA_10'] = ta.trend.wma_indicator(df['Close'], window=10)
        result['WMA_20'] = ta.trend.wma_indicator(df['Close'], window=20)
    except:
        pass
    
    # VWAP
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        result['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    except:
        pass
    
    # RSI variants
    for period in [7, 14, 21, 28]:
        result[f'RSI_{period}'] = ta.momentum.rsi(df['Close'], window=period)
    
    # Stochastic
    try:
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        result['STOCHk_14_3_3'] = stoch.stoch()
        result['STOCHd_14_3_3'] = stoch.stoch_signal()
    except:
        pass
    
    # Williams %R
    try:
        result['WILLR_14'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    except:
        pass
    
    # CCI
    for period in [14, 20]:
        result[f'CCI_{period}'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=period)
    
    # Ultimate Oscillator
    try:
        result['UO'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
    except:
        pass
    
    # Awesome Oscillator
    try:
        result['AO'] = ta.momentum.awesome_oscillator(df['High'], df['Low'], window1=5, window2=34)
    except:
        pass
    
    # MACD
    try:
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        result['MACD_12_26_9'] = macd.macd()
        result['MACDh_12_26_9'] = macd.macd_diff()
        result['MACDs_12_26_9'] = macd.macd_signal()
    except:
        pass
    
    # Bollinger Bands
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        result['BBL_20_2.0_2.0'] = bb.bollinger_lband()
        result['BBM_20_2.0_2.0'] = bb.bollinger_mavg()
        result['BBU_20_2.0_2.0'] = bb.bollinger_hband()
        result['BBB_20_2.0_2.0'] = bb.bollinger_wband()
        result['BBP_20_2.0_2.0'] = bb.bollinger_pband()
    except:
        pass
    
    # Keltner Channel
    try:
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        result['KCLe_20_2'] = keltner.keltner_channel_lband()
        result['KCBe_20_2'] = keltner.keltner_channel_mband()
        result['KCUe_20_2'] = keltner.keltner_channel_hband()
    except:
        pass
    
    # Donchian Channel
    try:
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
        result['DCL_20_20'] = donchian.donchian_channel_lband()
        result['DCM_20_20'] = donchian.donchian_channel_mband()
        result['DCU_20_20'] = donchian.donchian_channel_hband()
    except:
        pass
    
    # ATR
    for period in [7, 14, 20]:
        result[f'ATR_{period}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
    
    # Volume indicators
    try:
        result['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        result['OBV_SMA20'] = result['OBV'].rolling(window=20).mean()
    except:
        pass
    
    # Volume MAs
    for period in [5, 10, 20]:
        result[f'Volume_MA{period}'] = df['Volume'].rolling(window=period).mean()
    
    try:
        result['Volume_Ratio'] = df['Volume'] / result['Volume_MA20']
    except:
        pass
    
    # ADX
    try:
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        result['ADX_14'] = adx.adx()
        result['DMP_14'] = adx.adx_pos()
        result['DMN_14'] = adx.adx_neg()
    except:
        pass
    
    # Aroon
    try:
        aroon = ta.trend.AroonIndicator(df['Close'], window=25)
        result['AROONU_25'] = aroon.aroon_up()
        result['AROOND_25'] = aroon.aroon_down()
        result['AROONOSC_25'] = aroon.aroon_indicator()
    except:
        pass
    
    # MFI
    try:
        result['MFI_14'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except:
        pass
    
    # CMF
    try:
        result['CMF_20'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    except:
        pass
    
    # ROC
    for period in [10, 20]:
        result[f'ROC_{period}'] = ta.momentum.roc(df['Close'], window=period)
    
    # Momentum
    for period in [10, 20]:
        result[f'MOM_{period}'] = df['Close'].diff(period)
    
    # TSI
    try:
        tsi = ta.momentum.TSIIndicator(df['Close'], window_slow=25, window_fast=13)
        result['TSI_13_25_13'] = tsi.tsi()
    except:
        pass
    
    # Historical Volatility
    for period in [10, 20, 30]:
        result[f'HV_{period}'] = df['Close'].pct_change().rolling(window=period).std() * 100
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    logger.info("="*80)
    logger.info("ML SCREENING & PREDICTION - FIXED VERSION")
    logger.info("="*80)
    logger.info("Using FLAT features (no prefixes) to match CSV-trained model")
    logger.info("Primary timepoint: T-3 (3 days ago)")
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
    
    # STEP 2: FETCH STOCK DATA
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FETCH STOCK DATA (T-3 WITH FLAT FEATURES)")
    logger.info("="*80)
    logger.info(f"Fetching data for {len(symbols)} stocks...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    enriched_stocks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_data_for_prediction, sym, logger): sym 
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
    
    logger.info(f"✓ Fetched data for {len(enriched_stocks)} stocks")
    
    # STEP 3: PREPARE FEATURES (FLAT - NO PREFIXES)
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PREPARE FLAT FEATURES")
    logger.info("="*80)
    
    features_df = pd.DataFrame(enriched_stocks)
    logger.info(f"✓ Prepared {len(features_df)} stocks with {len(features_df.columns)} flat features")
    
    # STEP 4: ML PREDICTION
    logger.info("\n" + "="*80)
    logger.info("STEP 4: ML PREDICTION")
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
            # Try to get from Close feature
            for col in row.index:
                if col.lower() in ['close']:
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
                if col.lower() in ['close']:
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
        'model_version': 'csv_trained_flat_features'
    }
    
    supabase.write_screening_log(screening_log)
    
    # Export CSV
    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PREDICTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Predictions for: {prediction_date}")
    logger.info(f"Model type: CSV-trained (T-3, T-5, T-10 flat features)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
