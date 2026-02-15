#!/usr/bin/env python3
"""
Autonomous ML Stock Screener & Predictor - COMPLETE VERSION
Uses tradingview_scraper.symbols.screener.Screener for screening
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.explosion_predictor import ExplosionPredictor
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient

# Import tradingview-scraper
SCREENER_AVAILABLE = False
Screener = None

try:
    from tradingview_scraper.symbols.screener import Screener
    SCREENER_AVAILABLE = True
    print("✓ Imported Screener from tradingview_scraper.symbols.screener")
except ImportError as e:
    print(f"✗ Failed to import: {e}")


def setup_logging(verbose: bool = False):
    """Setup basic logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class SmartScreener:
    def __init__(self, config: dict = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.filters = self._load_learned_filters()
        
        if SCREENER_AVAILABLE:
            self.screener = Screener()
        else:
            self.screener = None
    
    def _load_learned_filters(self) -> dict:
        defaults = {
            'min_price': 0.25,
            'max_price': 250.0,
            'min_volume': 50000,
        }
        
        try:
            filter_path = Path('ml_models/learned_filters.json')
            if filter_path.exists():
                with open(filter_path, 'r') as f:
                    learned = json.load(f)
                for key, value in learned.items():
                    if value is not None:
                        defaults[key] = value
                self.logger.info("✓ Loaded learned filters")
            else:
                self.logger.info("No learned filters yet - using minimal constraints")
        except Exception as e:
            self.logger.debug(f"Using default filters: {e}")
        
        return defaults
    
    def screen_with_tradingview(self, max_results: int = 500) -> pd.DataFrame:
        if not SCREENER_AVAILABLE or self.screener is None:
            self.logger.error("Screener not available!")
            return pd.DataFrame()
        
        self.logger.info("Screening stocks with tradingview-scraper...")
        self.logger.info(f"  Price: ${self.filters['min_price']}-${self.filters['max_price']}")
        self.logger.info(f"  Volume: >= {self.filters['min_volume']:,}")
        
        try:
            filters = [
                {'left': 'close', 'operation': 'greater', 'right': self.filters['min_price']},
                {'left': 'close', 'operation': 'less', 'right': self.filters['max_price']},
                {'left': 'volume', 'operation': 'greater', 'right': self.filters['min_volume']},
            ]
            
            # Don't specify columns - let TradingView return defaults
            # Then we'll work with whatever we get
            
            self.logger.debug("Requesting default columns from TradingView...")
            
            result = self.screener.screen(
                market='america',
                filters=filters,
                sort_by='volume',
                sort_order='desc',
                limit=max_results
            )
            
            if result['status'] == 'success' and result.get('data'):
                df = pd.DataFrame(result['data'])
                
                # Log what columns we actually got
                self.logger.info(f"Received columns from TradingView: {list(df.columns)}")
                
                self.logger.info(f"✓ Found {len(df)} stocks (total available: {result.get('totalCount', '?')})")
                return df
            
            self.logger.warning("No results from screener")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, screened_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Preparing features...")
        
        # Extract ticker from "NASDAQ:AAPL" format
        if 'symbol' in screened_df.columns:
            screened_df['ticker'] = screened_df['symbol'].str.split(':').str[-1]
            screened_df['exchange_prefix'] = screened_df['symbol'].str.split(':').str[0]
        
        # Map TradingView column names to model feature names
        rename_map = {
            'ticker': 'symbol',
            'close': 'close',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
            'RSI': 'rsi',
            'RSI[1]': 'rsi[1]',
            'ADX': 'adx',
            'ADX+DI': 'adx+di',
            'ADX-DI': 'adx-di',
            'MACD.macd': 'macd.macd',
            'MACD.signal': 'macd.signal',
            'Stoch.K': 'stoch.k',
            'Stoch.D': 'stoch.d',
            'EMA5': 'ema5',
            'EMA10': 'ema10',
            'EMA20': 'ema20',
            'EMA50': 'ema50',
            'EMA100': 'ema100',
            'EMA200': 'ema200',
            'SMA5': 'sma5',
            'SMA10': 'sma10',
            'SMA20': 'sma20',
            'SMA50': 'sma50',
            'SMA100': 'sma100',
            'SMA200': 'sma200',
            'ATR': 'atr',
            'BB.upper': 'bb.upper',
            'BB.lower': 'bb.lower',
            'BB.middle': 'bb.middle',
            'CCI20': 'cci20',
            'AO': 'ao',
            'UO': 'uo',
            'VWAP': 'vwap',
            'change_abs': 'volume_change',
            'Volatility.D': 'volatility_20d',
        }
        
        features_df = screened_df.rename(columns=rename_map)
        
        # Calculate derived features
        if 'bb.upper' in features_df.columns and 'bb.lower' in features_df.columns:
            features_df['bb_width'] = features_df['bb.upper'] - features_df['bb.lower']
        
        # Calculate volume ratio
        if 'volume' in features_df.columns:
            # Use volume itself as a proxy for volume ratio (will normalize during scaling)
            features_df['volume_ratio'] = 1.2
        
        # Fill missing features that model expects
        # Use close price as fallback for OHLC if somehow missing
        if 'close' in features_df.columns:
            if 'open' not in features_df.columns:
                features_df['open'] = features_df['close']
            if 'high' not in features_df.columns:
                features_df['high'] = features_df['close'] * 1.02
            if 'low' not in features_df.columns:
                features_df['low'] = features_df['close'] * 0.98
        
        # Keltner channels - use BB as proxy
        if 'bb.upper' in features_df.columns:
            features_df['keltner_upper'] = features_df['bb.upper']
            features_df['keltner_lower'] = features_df['bb.lower']
        else:
            features_df['keltner_upper'] = 0
            features_df['keltner_lower'] = 0
        
        # Donchian channels - use high/low
        if 'high' in features_df.columns and 'low' in features_df.columns:
            features_df['donchian_upper'] = features_df['high']
            features_df['donchian_lower'] = features_df['low']
            features_df['donchian_middle'] = (features_df['high'] + features_df['low']) / 2
        else:
            features_df['donchian_upper'] = 0
            features_df['donchian_lower'] = 0
            features_df['donchian_middle'] = 0
        
        # Volatility - use the one we have or default
        if 'volatility_20d' in features_df.columns:
            features_df['volatility_10d'] = features_df['volatility_20d']
            features_df['volatility_30d'] = features_df['volatility_20d']
        else:
            features_df['volatility_10d'] = 1.5
            features_df['volatility_20d'] = 1.5
            features_df['volatility_30d'] = 1.5
        
        # OBV - if not available, use volume as proxy
        if 'obv' not in features_df.columns and 'volume' in features_df.columns:
            features_df['obv'] = features_df['volume'] * 10
        elif 'obv' not in features_df.columns:
            features_df['obv'] = 0
        
        # Set exchange
        features_df['exchange'] = features_df.get('exchange_prefix', 'NASDAQ')
        
        # Log what we have
        self.logger.info(f"Features prepared: {len(features_df.columns)} columns")
        missing_cols = [col for col in ['symbol', 'close', 'volume', 'rsi', 'macd.macd'] 
                       if col not in features_df.columns]
        if missing_cols:
            self.logger.warning(f"Missing critical columns: {missing_cols}")
        
        return features_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="auto")
    parser.add_argument("--max-workers", type=int, default=15)
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    logger.info("="*80)
    logger.info("ML STOCK SCREENING & PREDICTION")
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
    logger.info("STEP 1: SCREENING")
    logger.info("="*80)
    
    screened_df = screener.screen_with_tradingview(max_results=args.max_results)
    
    if screened_df.empty:
        logger.error("No stocks passed screening")
        return 1
    
    logger.info(f"✓ Screened {len(screened_df)} stocks from TradingView")
    
    # STEP 2: FETCH TECHNICAL INDICATORS
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FETCH TECHNICAL INDICATORS")
    logger.info("="*80)
    
    # Extract symbols from screened results
    symbols = []
    if 'symbol' in screened_df.columns:
        # Handle "NASDAQ:AAPL" format
        symbols = screened_df['symbol'].str.split(':').str[-1].tolist()
    else:
        logger.error("No symbol column in screened results")
        return 1
    
    logger.info(f"Fetching detailed indicators for {len(symbols)} stocks using yfinance...")
    
    # Fetch indicators using yfinance + ta library
    import yfinance as yf
    import ta
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def fetch_stock_indicators(symbol):
        """Fetch full technical indicators for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df.empty or len(df) < 20:
                return None
            
            # Calculate all technical indicators
            # Trend indicators
            df['sma5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['sma10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['sma20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['sma50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['sma100'] = ta.trend.sma_indicator(df['Close'], window=100)
            df['sma200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            df['ema5'] = ta.trend.ema_indicator(df['Close'], window=5)
            df['ema10'] = ta.trend.ema_indicator(df['Close'], window=10)
            df['ema20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['ema50'] = ta.trend.ema_indicator(df['Close'], window=50)
            df['ema100'] = ta.trend.ema_indicator(df['Close'], window=100)
            df['ema200'] = ta.trend.ema_indicator(df['Close'], window=200)
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
            df['stoch.k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
            df['stoch.d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
            
            df['macd.macd'] = ta.trend.macd(df['Close'])
            df['macd.signal'] = ta.trend.macd_signal(df['Close'])
            
            df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
            df['cci20'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
            df['ao'] = ta.momentum.awesome_oscillator(df['High'], df['Low'])
            df['uo'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
            
            # Volatility indicators
            bb = ta.volatility.BollingerBands(df['Close'])
            df['bb.upper'] = bb.bollinger_hband()
            df['bb.lower'] = bb.bollinger_lband()
            df['bb.middle'] = bb.bollinger_mavg()
            df['bb_width'] = df['bb.upper'] - df['bb.lower']
            
            df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            df['keltner_upper'] = keltner.keltner_channel_hband()
            df['keltner_lower'] = keltner.keltner_channel_lband()
            
            donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
            df['donchian_upper'] = donchian.donchian_channel_hband()
            df['donchian_lower'] = donchian.donchian_channel_lband()
            df['donchian_middle'] = donchian.donchian_channel_mband()
            
            # Volume indicators
            df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            # VWAP
            df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            
            # Volatility periods
            df['volatility_10d'] = df['Close'].pct_change().rolling(window=10).std() * 100
            df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * 100
            df['volatility_30d'] = df['Close'].pct_change().rolling(window=30).std() * 100
            
            # Get the most recent row
            latest = df.iloc[-1]
            
            result = {
                'symbol': symbol,
                'close': latest['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'exchange': 'NASDAQ',  # Default
            }
            
            # Add all indicators
            for col in df.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    result[col] = latest[col]
            
            return result
            
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return None
    
    # Fetch in parallel
    enriched_stocks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_indicators, sym): sym for sym in symbols}
        
        for i, future in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                logger.info(f"  Fetched {i}/{len(symbols)} stocks...")
            
            result = future.result()
            if result:
                enriched_stocks.append(result)
    
    if not enriched_stocks:
        logger.error("Failed to fetch indicators for any stocks")
        return 1
    
    features_df = pd.DataFrame(enriched_stocks)
    logger.info(f"✓ Fetched indicators for {len(features_df)} stocks")
    logger.info(f"  Columns: {len(features_df.columns)}")
    
    # STEP 3: ML PREDICTION
    logger.info("\n" + "="*80)
    logger.info("STEP 3: ML PREDICTION")
    logger.info("="*80)
    
    historical_gains = supabase.get_prediction_accuracy(days_back=30)
    if not historical_gains.empty:
        logger.info(f"Using {len(historical_gains)} historical records for calibration")
    
    try:
        predictions_df = predictor.predict_with_targets(features_df, historical_gains)
        logger.info(f"✓ Generated {len(predictions_df)} predictions")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # STEP 4: TOP PREDICTIONS
    logger.info("\n" + "="*80)
    logger.info(f"STEP 4: TOP {args.top_n} PREDICTIONS")
    logger.info("="*80)
    
    top_predictions = predictions_df.head(args.top_n)
    
    signal_counts = top_predictions['signal'].value_counts()
    logger.info("\nSignal Distribution:")
    for signal, count in signal_counts.items():
        logger.info(f"  {signal}: {count}")
    
    prob_stats = top_predictions['explosion_probability']
    logger.info(f"\nProbability Range: {prob_stats.min()*100:.2f}% - {prob_stats.max()*100:.2f}% (avg: {prob_stats.mean()*100:.2f}%)")
    
    logger.info(f"\nTop {min(20, len(top_predictions))} Predictions:")
    logger.info("-" * 90)
    logger.info(f"{'Rank':<5} {'Symbol':<8} {'Signal':<13} {'Prob':<8} {'Price':<8} {'Target':<8} {'Gain':<8}")
    logger.info("-" * 90)
    
    for idx, row in top_predictions.head(20).iterrows():
        logger.info(
            f"{idx+1:<5} {row['symbol']:<8} {row['signal']:<13} "
            f"{row['explosion_probability']*100:>6.2f}%  "
            f"${row.get('current_price', 0):>6.2f}  "
            f"${row.get('target_price', 0):>6.2f}  "
            f"+{row.get('target_gain_pct', 0):>5.1f}%"
        )
    
    # STEP 5: STORE PREDICTIONS
    logger.info("\n" + "="*80)
    logger.info("STEP 5: STORE PREDICTIONS")
    logger.info("="*80)
    
    prediction_date = datetime.now().date().isoformat()
    predictions_list = []
    
    for _, row in top_predictions.iterrows():
        original = features_df[features_df['symbol'] == row['symbol']]
        if original.empty:
            continue
        
        orig_row = original.iloc[0]
        
        prediction_record = {
            'symbol': row['symbol'],
            'exchange': orig_row.get('exchange', 'NASDAQ'),
            'prediction_date': prediction_date,
            'explosion_probability': float(row['explosion_probability']),
            'prediction': int(row['prediction']),
            'signal': row['signal'],
            'target_gain_pct': float(row.get('target_gain_pct', 0)),
            'target_gain_low': float(row.get('target_gain_low', 0)),
            'target_gain_high': float(row.get('target_gain_high', 0)),
            'current_price': float(row.get('current_price', 0)),
            'target_price': float(row.get('target_price', 0)),
            'target_price_low': float(row.get('target_price_low', 0)),
            'target_price_high': float(row.get('target_price_high', 0)),
            'rsi': float(orig_row.get('rsi', 0)) if pd.notna(orig_row.get('rsi')) else None,
            'macd': float(orig_row.get('macd.macd', 0)) if pd.notna(orig_row.get('macd.macd')) else None,
            'adx': float(orig_row.get('adx', 0)) if pd.notna(orig_row.get('adx')) else None,
            'volume_ratio': float(orig_row.get('volume_change', 0)) if pd.notna(orig_row.get('volume_change')) else None,
            'bb_width': float(orig_row.get('bb_width', 0)) if pd.notna(orig_row.get('bb_width')) else None,
        }
        
        predictions_list.append(prediction_record)
    
    if predictions_list:
        logger.info(f"Writing {len(predictions_list)} predictions to database...")
        count = supabase.write_predictions(predictions_list)
        logger.info(f"✓ Wrote {count} predictions")
    
    # STEP 6: LOG STATISTICS
    logger.info("\n" + "="*80)
    logger.info("STEP 6: LOG STATISTICS")
    logger.info("="*80)
    
    screening_log = {
        'screening_date': prediction_date,
        'total_symbols_attempted': args.max_results,
        'symbols_fetched_successfully': len(screened_df),
        'symbols_after_price_filter': len(screened_df),
        'symbols_after_volume_filter': len(screened_df),
        'symbols_after_all_filters': len(features_df),
        'total_predictions': len(predictions_df),
        'strong_buy_count': len(predictions_df[predictions_df['signal'] == 'STRONG BUY']),
        'buy_count': len(predictions_df[predictions_df['signal'] == 'BUY']),
        'hold_count': len(predictions_df[predictions_df['signal'] == 'HOLD']),
        'avoid_count': len(predictions_df[predictions_df['signal'] == 'AVOID']),
        'avg_probability': float(predictions_df['explosion_probability'].mean()),
        'max_probability': float(predictions_df['explosion_probability'].max()),
        'min_probability': float(predictions_df['explosion_probability'].min()),
        'model_version': 'xgboost_v1',
        'screening_method': 'tradingview_screener'
    }
    
    if supabase.write_screening_log(screening_log):
        logger.info("✓ Screening statistics logged")
    
    # Export CSV
    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)
    logger.info(f"✓ Exported to {csv_path}")
    
    # FINAL SUMMARY
    logger.info("\n" + "="*80)
    logger.info("✓ SCREENING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nScreened: {len(screened_df)} stocks")
    logger.info(f"Predicted: {len(predictions_df)} stocks")
    logger.info(f"Stored: {len(predictions_list)} top predictions")
    logger.info(f"\nNext: Wait for market close, then run ml_track_comprehensive_accuracy.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
