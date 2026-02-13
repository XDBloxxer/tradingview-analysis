#!/usr/bin/env python3
"""
Autonomous ML Stock Screener & Predictor
Screens large universe, predicts explosions, tracks comprehensive accuracy
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tradingview_ta import TA_Handler, Interval
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config
from src.ml_predictor.explosion_predictor import ExplosionPredictor
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


class StockScreener:
    """Autonomous stock screener with learned filters"""
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Load learned filters from model metadata if available
        self.filters = self._load_learned_filters()
    
    def _load_learned_filters(self) -> dict:
        """Load screening filters learned from model training"""
        
        # Default filters based on typical explosion characteristics
        defaults = {
            'min_price': 3.0,        # Avoid penny stocks
            'max_price': 500.0,      # Avoid high-priced stocks
            'min_volume': 500000,    # Minimum liquidity
            'min_avg_volume': 300000, # 20-day average volume
            'max_market_cap': 50e9,  # Focus on smaller caps (more explosive potential)
            'exchanges': ['NASDAQ', 'NYSE', 'AMEX']
        }
        
        # Try to load from model metadata
        try:
            import json
            metadata_path = Path('ml_models/model_metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Override with learned filters if available
                if 'screening_filters' in metadata:
                    defaults.update(metadata['screening_filters'])
                    self.logger.info("Loaded learned screening filters from model")
        except Exception as e:
            self.logger.debug(f"Using default filters: {e}")
        
        return defaults
    
    def get_stock_universe(self, source: str = 'auto') -> list:
        """
        Get comprehensive stock universe
        
        Args:
            source: 'auto', 'sp500', 'nasdaq', 'russell2000', 'all'
        """
        
        symbols = set()
        
        if source in ['auto', 'all']:
            # Get multiple sources for comprehensive coverage
            self.logger.info("Building comprehensive stock universe...")
            
            # S&P 500
            try:
                url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
                df = pd.read_html(url)[0]
                sp500 = df['Symbol'].str.replace('.', '-').tolist()
                symbols.update(sp500)
                self.logger.info(f"Added {len(sp500)} S&P 500 stocks")
            except Exception as e:
                self.logger.warning(f"Failed to load S&P 500: {e}")
            
            # NASDAQ 100
            try:
                url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
                df = pd.read_html(url)[4]
                nasdaq100 = df['Ticker'].tolist()
                symbols.update(nasdaq100)
                self.logger.info(f"Added {len(nasdaq100)} NASDAQ-100 stocks")
            except Exception as e:
                self.logger.warning(f"Failed to load NASDAQ-100: {e}")
            
            # Russell 2000 (small caps - more explosive)
            try:
                url = 'https://en.wikipedia.org/wiki/Russell_2000_Index'
                tables = pd.read_html(url)
                if len(tables) > 0:
                    russell = tables[2]['Ticker'].tolist()
                    symbols.update(russell)
                    self.logger.info(f"Added {len(russell)} Russell 2000 stocks")
            except Exception as e:
                self.logger.warning(f"Failed to load Russell 2000: {e}")
            
            # Add high-volume NASDAQ stocks
            try:
                url = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt'
                df = pd.read_csv(url, sep='|')
                df = df[df['Test Issue'] == 'N']
                df = df[df['Financial Status'] == 'N']
                nasdaq = df['Symbol'].tolist()[:500]  # Top 500 by listing
                symbols.update(nasdaq)
                self.logger.info(f"Added {len(nasdaq)} additional NASDAQ stocks")
            except Exception as e:
                self.logger.warning(f"Failed to load NASDAQ: {e}")
        
        elif source == 'sp500':
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(url)[0]
            symbols = set(df['Symbol'].str.replace('.', '-').tolist())
        
        symbols_list = sorted(list(symbols))
        self.logger.info(f"Total universe: {len(symbols_list)} stocks")
        
        return symbols_list
    
    def fetch_stock_data(self, symbol: str, exchange: str = 'NASDAQ') -> dict:
        """Fetch comprehensive stock data with rate limiting"""
        
        try:
            # Rate limiting
            time.sleep(0.1)  # 10 requests/second max
            
            handler = TA_Handler(
                symbol=symbol,
                exchange=exchange,
                screener="america",
                interval=Interval.INTERVAL_1_DAY,
                timeout=10
            )
            
            analysis = handler.get_analysis()
            indicators = analysis.indicators
            
            # Calculate additional metrics
            close = indicators.get('close', 0)
            volume = indicators.get('volume', 0)
            
            # Volume average (approximate)
            volume_sma20 = indicators.get('volume|20', volume)
            volume_ratio = volume / volume_sma20 if volume_sma20 > 0 else 1.0
            
            result = {
                'symbol': symbol,
                'exchange': exchange,
                'close': close,
                'open': indicators.get('open'),
                'high': indicators.get('high'),
                'low': indicators.get('low'),
                'volume': volume,
                'volume_ratio': volume_ratio,
                
                # All indicators
                **{k: v for k, v in indicators.items() if v is not None}
            }
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Failed to fetch {symbol}: {e}")
            return None
    
    def screen_stocks_parallel(self, symbols: list, max_workers: int = 10) -> pd.DataFrame:
        """Screen stocks in parallel with progress tracking"""
        
        self.logger.info(f"Screening {len(symbols)} stocks with {max_workers} workers...")
        
        results = []
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_stock_data, symbol): symbol
                for symbol in symbols
            }
            
            for i, future in enumerate(as_completed(future_to_symbol), 1):
                symbol = future_to_symbol[future]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        failed += 1
                    
                    if i % 100 == 0:
                        self.logger.info(f"Progress: {i}/{len(symbols)} ({len(results)} successful, {failed} failed)")
                        
                except Exception as e:
                    self.logger.debug(f"Error processing {symbol}: {e}")
                    failed += 1
        
        self.logger.info(f"Screening complete: {len(results)} successful, {failed} failed")
        
        return pd.DataFrame(results)
    
    def apply_learned_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent filters based on explosion characteristics"""
        
        self.logger.info(f"Applying learned filters to {len(df)} stocks...")
        
        initial_count = len(df)
        
        # Price filters
        df = df[
            (df['close'] >= self.filters['min_price']) &
            (df['close'] <= self.filters['max_price'])
        ]
        self.logger.info(f"  Price filter: {len(df)} stocks (${self.filters['min_price']}-${self.filters['max_price']})")
        
        # Volume filters
        df = df[df['volume'] >= self.filters['min_volume']]
        self.logger.info(f"  Volume filter: {len(df)} stocks (>= {self.filters['min_volume']:,.0f})")
        
        # Remove NaN close
        df = df[df['close'].notna()]
        
        # Additional learned filters (if model has identified patterns)
        # For example: stocks with RSI 30-70, positive momentum, etc.
        
        self.logger.info(f"Filters applied: {initial_count} → {len(df)} stocks ({len(df)/initial_count*100:.1f}% pass rate)")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Autonomous ML stock screening and prediction")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--universe", default="auto",
                       choices=['auto', 'sp500', 'nasdaq', 'all'],
                       help="Stock universe to screen")
    parser.add_argument("--max-workers", type=int, default=15,
                       help="Parallel workers for data fetching")
    parser.add_argument("--top-n", type=int, default=50,
                       help="Number of top predictions to store")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, config.get("logging", {}))
    
    logger.info("="*80)
    logger.info("AUTONOMOUS ML STOCK SCREENING & PREDICTION")
    logger.info("="*80)
    
    # Initialize
    screener = StockScreener(config)
    
    try:
        predictor = ExplosionPredictor()
        supabase = MLPredictionSupabaseClient(config)
    except Exception as e:
        logger.error(f"Failed to initialize ML system: {e}")
        return 1
    
    # Step 1: Build stock universe
    logger.info("\n" + "="*80)
    logger.info("STEP 1: BUILD STOCK UNIVERSE")
    logger.info("="*80)
    
    symbols = screener.get_stock_universe(args.universe)
    
    if not symbols:
        logger.error("Failed to build stock universe")
        return 1
    
    # Step 2: Screen and fetch data
    logger.info("\n" + "="*80)
    logger.info("STEP 2: SCREEN & FETCH STOCK DATA")
    logger.info("="*80)
    logger.info("This will take 5-15 minutes depending on universe size...")
    
    stock_data = screener.screen_stocks_parallel(symbols, args.max_workers)
    
    if stock_data.empty:
        logger.error("Failed to fetch any stock data")
        return 1
    
    logger.info(f"Successfully fetched {len(stock_data)} stocks")
    
    # Step 3: Apply learned filters
    logger.info("\n" + "="*80)
    logger.info("STEP 3: APPLY LEARNED FILTERS")
    logger.info("="*80)
    
    filtered_data = screener.apply_learned_filters(stock_data)
    
    if filtered_data.empty:
        logger.error("No stocks passed filters")
        return 1
    
    # Step 4: Prepare features and predict
    logger.info("\n" + "="*80)
    logger.info("STEP 4: ML PREDICTION")
    logger.info("="*80)
    logger.info(f"Running predictions on {len(filtered_data)} filtered stocks...")
    
    features_df = predictor.prepare_features_from_daily_winners(filtered_data)
    
    # Get historical gains for calibration
    historical_gains = supabase.get_historical_prediction_accuracy(days_back=30)
    if not historical_gains.empty:
        logger.info(f"Using {len(historical_gains)} historical records for calibration")
    
    predictions_df = predictor.predict_with_targets(features_df, historical_gains)
    
    logger.info(f"Generated {len(predictions_df)} predictions")
    
    # Step 5: Select top predictions
    logger.info("\n" + "="*80)
    logger.info(f"STEP 5: TOP {args.top_n} PREDICTIONS")
    logger.info("="*80)
    
    top_predictions = predictions_df.head(args.top_n)
    
    # Display summary by signal
    signal_counts = top_predictions['signal'].value_counts()
    logger.info("\nSignal Distribution:")
    for signal, count in signal_counts.items():
        logger.info(f"  {signal}: {count}")
    
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
    
    # Step 6: Store predictions in database
    logger.info("\n" + "="*80)
    logger.info("STEP 6: STORE PREDICTIONS")
    logger.info("="*80)
    
    prediction_date = datetime.now().date().isoformat()
    
    predictions_list = []
    
    for _, row in top_predictions.iterrows():
        original_data = filtered_data[filtered_data['symbol'] == row['symbol']]
        
        if original_data.empty:
            continue
        
        original_row = original_data.iloc[0]
        
        prediction_record = {
            'symbol': row['symbol'],
            'exchange': original_row.get('exchange', 'NASDAQ'),
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
            'rsi': float(original_row.get('RSI', 0)) if pd.notna(original_row.get('RSI')) else None,
            'macd': float(original_row.get('MACD.macd', 0)) if pd.notna(original_row.get('MACD.macd')) else None,
            'adx': float(original_row.get('ADX', 0)) if pd.notna(original_row.get('ADX')) else None,
            'volume_ratio': float(original_row.get('volume_ratio', 0)) if pd.notna(original_row.get('volume_ratio')) else None,
            'hv_20': float(original_row.get('Volatility.D', 0)) if pd.notna(original_row.get('Volatility.D')) else None,
            'bb_width': float(original_row.get('BB.upper', 0) - original_row.get('BB.lower', 0)) if pd.notna(original_row.get('BB.upper')) else None,
        }
        
        predictions_list.append(prediction_record)
    
    if predictions_list:
        logger.info(f"Writing {len(predictions_list)} top predictions to database...")
        count = supabase.write_predictions(predictions_list)
        logger.info(f"✓ Successfully wrote {count} predictions")
    else:
        logger.warning("No predictions to write")
    
    # Step 7: Export results
    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Exported top {len(top_predictions)} predictions to {csv_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ AUTONOMOUS SCREENING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nScreened: {len(symbols)} stocks")
    logger.info(f"Filtered: {len(filtered_data)} stocks passed criteria")
    logger.info(f"Predicted: {len(predictions_df)} stocks analyzed")
    logger.info(f"Stored: {len(predictions_list)} top predictions")
    logger.info(f"\nResults saved to: {csv_path}")
    logger.info(f"Database table: ml_explosion_predictions")
    logger.info(f"\nNext: Wait for market close, then run ml_track_comprehensive_accuracy.py")


    # Step 8: Log screening statistics
    logger.info("\n" + "="*80)
    logger.info("STEP 7: LOG SCREENING STATISTICS")
    logger.info("="*80)
    
    screening_log = {
        'screening_date': prediction_date,
        'total_symbols_attempted': len(symbols),
        'symbols_fetched_successfully': len(stock_data),
        'symbols_failed_fetch': len(symbols) - len(stock_data),
        'symbols_after_price_filter': len(filtered_data),
        'symbols_after_volume_filter': len(filtered_data),
        'symbols_after_all_filters': len(filtered_data),
        'total_predictions': len(predictions_df),
        'strong_buy_count': len(predictions_df[predictions_df['signal'] == 'STRONG BUY']),
        'buy_count': len(predictions_df[predictions_df['signal'] == 'BUY']),
        'hold_count': len(predictions_df[predictions_df['signal'] == 'HOLD']),
        'avoid_count': len(predictions_df[predictions_df['signal'] == 'AVOID']),
        'avg_probability': float(predictions_df['explosion_probability'].mean()),
        'max_probability': float(predictions_df['explosion_probability'].max()),
        'min_probability': float(predictions_df['explosion_probability'].min()),
        'screening_duration_seconds': None,  # Add timing if needed
        'prediction_duration_seconds': None,
        'model_version': 'xgboost_v1',
        'screening_universe': args.universe
    }
    
    if supabase.write_screening_log(screening_log):
        logger.info("✓ Screening statistics logged")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
