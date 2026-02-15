#!/usr/bin/env python3
"""
Autonomous ML Stock Screener & Predictor - CORRECT VERSION
Uses tradingview_scraper.symbols.screener.Screener for smart pre-filtering
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.explosion_predictor import ExplosionPredictor
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient

# Import tradingview-scraper CORRECTLY
print("Attempting to import tradingview-scraper Screener...")

SCREENER_AVAILABLE = False
Screener = None

try:
    from tradingview_scraper.symbols.screener import Screener
    SCREENER_AVAILABLE = True
    print("✓ SUCCESS: Imported Screener from tradingview_scraper.symbols.screener")
except ImportError as e:
    print(f"✗ FAILED: {e}")
    print("⚠️  tradingview-scraper Screener not available")
    print("   Install with: pip install tradingview-scraper")


def setup_logging(verbose: bool = False):
    """Setup basic logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class SmartScreener:
    """
    Intelligent screener that uses tradingview-scraper's Screener
    Adapts filters based on model performance
    """
    
    def __init__(self, config: dict = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Load learned filters
        self.filters = self._load_learned_filters()
        
        # Initialize screener if available
        if SCREENER_AVAILABLE:
            self.screener = Screener()
        else:
            self.screener = None
    
    def _load_learned_filters(self) -> dict:
        """
        Load screening filters - FULLY LEARNABLE
        These are updated weekly based on what actually exploded
        """
        
        # STARTING defaults (will be replaced by learning)
        defaults = {
            'min_price': 3.0,
            'max_price': 500.0,
            'min_volume': 500000,
            
            # LEARNABLE filters (updated weekly)
            'min_rsi': None,      # No filter initially - let model learn
            'max_rsi': None,
            'min_adx': None,
            'max_adx': None,
            'min_macd': None,
            'max_macd': None,
            'min_volume_change': None,
            'max_volume_change': None,
            
            # Market cap filter (optional)
            'market_cap_max': None,  # None = no limit
        }
        
        # Load learned filters from weekly analysis
        try:
            filter_path = Path('ml_models/learned_filters.json')
            if filter_path.exists():
                with open(filter_path, 'r') as f:
                    learned = json.load(f)
                    
                # Override defaults with learned values
                for key, value in learned.items():
                    if value is not None:  # Only use non-None learned values
                        defaults[key] = value
                
                self.logger.info("✓ Loaded learned screening filters")
                self.logger.info(f"  Active filters: {sum(1 for v in learned.values() if v is not None)}")
            else:
                self.logger.info("No learned filters yet - using minimal constraints")
                
        except Exception as e:
            self.logger.debug(f"Using default filters: {e}")
        
        return defaults
    
    def screen_with_tradingview(self, max_results: int = 500) -> pd.DataFrame:
        """
        Use tradingview-scraper's Screener with DYNAMIC learnable filters
        Only applies filters that have been learned (not None)
        """
        
        if not SCREENER_AVAILABLE or self.screener is None:
            self.logger.error("tradingview-scraper Screener not available!")
            return pd.DataFrame()
        
        self.logger.info("Screening stocks with tradingview-scraper Screener...")
        self.logger.info("Active filters:")
        self.logger.info(f"  Price: ${self.filters['min_price']}-${self.filters['max_price']}")
        self.logger.info(f"  Volume: >= {self.filters['min_volume']:,}")
        
        # Log learned filters (only non-None)
        learned_active = {k: v for k, v in self.filters.items() 
                         if v is not None and k not in ['min_price', 'max_price', 'min_volume']}
        if learned_active:
            self.logger.info("  Learned filters:")
            for key, value in learned_active.items():
                self.logger.info(f"    {key}: {value}")
        else:
            self.logger.info("  No learned filters active yet (first run)")
        
        try:
            # Build filter list for Screener API
            screener_filters = [
                {'left': 'close', 'operation': 'in_range', 'right': [self.filters['min_price'], self.filters['max_price']]},
                {'left': 'volume', 'operation': 'greater', 'right': self.filters['min_volume']},
            ]
            
            # Add learned filters ONLY if they exist (not None)
            if self.filters['min_rsi'] is not None and self.filters['max_rsi'] is not None:
                screener_filters.append({
                    'left': 'RSI',
                    'operation': 'in_range',
                    'right': [self.filters['min_rsi'], self.filters['max_rsi']]
                })
            
            if self.filters['min_adx'] is not None:
                screener_filters.append({
                    'left': 'ADX',
                    'operation': 'greater',
                    'right': self.filters['min_adx']
                })
            
            if self.filters['max_adx'] is not None:
                screener_filters.append({
                    'left': 'ADX',
                    'operation': 'less',
                    'right': self.filters['max_adx']
                })
            
            if self.filters['min_macd'] is not None:
                screener_filters.append({
                    'left': 'MACD.macd',
                    'operation': 'greater',
                    'right': self.filters['min_macd']
                })
            
            if self.filters['max_macd'] is not None:
                screener_filters.append({
                    'left': 'MACD.macd',
                    'operation': 'less',
                    'right': self.filters['max_macd']
                })
            
            if self.filters['market_cap_max'] is not None:
                screener_filters.append({
                    'left': 'market_cap_basic',
                    'operation': 'less',
                    'right': self.filters['market_cap_max']
                })
            
            # Define columns to retrieve
            columns = [
                'name', 'close', 'volume', 'market_cap_basic',
                'RSI', 'ADX', 'MACD.macd', 'MACD.signal',
                'Stoch.K', 'Stoch.D', 'Rec.All',
                'EMA20', 'SMA20', 'ATR', 'BB.upper', 'BB.lower',
                'change'
            ]
            
            # Execute screening
            self.logger.info(f"Calling screener with {len(screener_filters)} filters...")
            result = self.screener.screen(
                market='america',
                filters=screener_filters,
                columns=columns,
                sort_by='volume',
                sort_order='desc',
                limit=max_results
            )
            
            if result['status'] != 'success' or not result.get('data'):
                self.logger.warning("No stocks matched screening criteria")
                self.logger.warning("Filters may be too restrictive - will be adjusted in weekly learning")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(result['data'])
            
            self.logger.info(f"✓ Found {len(df)} stocks matching criteria")
            self.logger.info(f"  Total available: {result.get('totalCount', 'unknown')}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def screen_from_supabase_fallback(self, supabase_client, max_results: int = 500) -> pd.DataFrame:
        """
        FALLBACK: Screen from Supabase day_prior_close data
        Used when tradingview-scraper is not available
        """
        
        self.logger.info("Using Supabase fallback screening...")
        self.logger.info("Active filters:")
        self.logger.info(f"  Price: ${self.filters['min_price']}-${self.filters['max_price']}")
        self.logger.info(f"  Volume: >= {self.filters['min_volume']:,}")
        
        try:
            # Get recent day_prior_close data
            stocks_df = supabase_client.get_latest_day_prior_close(limit=max_results * 2)
            
            if stocks_df.empty:
                self.logger.error("No data in Supabase")
                return pd.DataFrame()
            
            # Apply basic filters
            filtered = stocks_df[
                (stocks_df['close'] >= self.filters['min_price']) &
                (stocks_df['close'] <= self.filters['max_price']) &
                (stocks_df['volume'] >= self.filters['min_volume'])
            ]
            
            # Sort by volume
            filtered = filtered.sort_values('volume', ascending=False)
            
            self.logger.info(f"✓ Screened {len(filtered)} stocks from Supabase")
            
            return filtered.head(max_results)
            
        except Exception as e:
            self.logger.error(f"Supabase screening failed: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, screened_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert screener output to model features
        """
        
        self.logger.info("Preparing features for model...")
        
        # Handle symbol field - Screener returns 'symbol' with exchange prefix
        if 'symbol' in screened_df.columns:
            # Extract just the ticker from "NASDAQ:AAPL" format
            screened_df['ticker'] = screened_df['symbol'].str.split(':').str[1]
            screened_df['exchange_prefix'] = screened_df['symbol'].str.split(':').str[0]
        
        # Rename columns to match what model expects
        rename_map = {
            'ticker': 'symbol',
            'close': 'close',
            'volume': 'volume',
            'RSI': 'rsi',
            'ADX': 'adx',
            'MACD.macd': 'macd.macd',
            'MACD.signal': 'macd.signal',
            'Stoch.K': 'stoch.k',
            'Stoch.D': 'stoch.d',
            'EMA20': 'ema20',
            'SMA20': 'sma20',
            'ATR': 'atr',
            'BB.upper': 'bb.upper',
            'BB.lower': 'bb.lower',
        }
        
        features_df = screened_df.rename(columns=rename_map)
        
        # Add calculated features
        if 'bb.upper' in features_df.columns and 'bb.lower' in features_df.columns:
            features_df['bb_width'] = features_df['bb.upper'] - features_df['bb.lower']
        
        # Add exchange
        if 'exchange_prefix' in features_df.columns:
            features_df['exchange'] = features_df['exchange_prefix']
        elif 'exchange' not in features_df.columns:
            features_df['exchange'] = 'NASDAQ'
        
        return features_df


def main():
    parser = argparse.ArgumentParser(description="ML stock screening with tradingview-scraper")
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
    
    # Step 1: Smart screening
    logger.info("\n" + "="*80)
    logger.info("STEP 1: SCREENING")
    logger.info("="*80)
    
    if SCREENER_AVAILABLE:
        screened_df = screener.screen_with_tradingview(max_results=args.max_results)
    else:
        logger.warning("tradingview-scraper not available, using Supabase fallback")
        screened_df = screener.screen_from_supabase_fallback(supabase, max_results=args.max_results)
    
    if screened_df.empty:
        logger.error("No stocks passed screening")
        return 1
    
    # Step 2: Prepare features
    logger.info("\n" + "="*80)
    logger.info("STEP 2: PREPARE FEATURES")
    logger.info("="*80)
    
    features_df = screener.prepare_features(screened_df)
    
    # Step 3: ML Predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 3: ML PREDICTION")
    logger.info("="*80)
    logger.info(f"Running predictions on {len(features_df)} stocks...")
    
    # Get historical gains for calibration
    historical_gains = supabase.get_historical_prediction_accuracy(days_back=30)
    if not historical_gains.empty:
        logger.info(f"Using {len(historical_gains)} historical records for calibration")
    
    # Predict
    try:
        predictions_df = predictor.predict_with_targets(features_df, historical_gains)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info(f"Generated {len(predictions_df)} predictions")
    
    # Step 4: Select top predictions
    logger.info("\n" + "="*80)
    logger.info(f"STEP 4: TOP {args.top_n} PREDICTIONS")
    logger.info("="*80)
    
    top_predictions = predictions_df.head(args.top_n)
    
    # Display summary
    signal_counts = top_predictions['signal'].value_counts()
    logger.info("\nSignal Distribution:")
    for signal, count in signal_counts.items():
        logger.info(f"  {signal}: {count}")
    
    # Show probability distribution
    prob_min = top_predictions['explosion_probability'].min()
    prob_max = top_predictions['explosion_probability'].max()
    prob_mean = top_predictions['explosion_probability'].mean()
    
    logger.info(f"\nProbability Distribution:")
    logger.info(f"  Min:  {prob_min*100:.2f}%")
    logger.info(f"  Max:  {prob_max*100:.2f}%")
    logger.info(f"  Mean: {prob_mean*100:.2f}%")
    
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
    
    # Step 5: Store in database
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
        logger.info(f"✓ Successfully wrote {count} predictions")
    
    # Step 6: Log screening statistics
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
        'screening_method': 'tradingview_screener' if SCREENER_AVAILABLE else 'supabase_fallback'
    }
    
    if supabase.write_screening_log(screening_log):
        logger.info("✓ Screening statistics logged")
    
    # Export CSV
    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Exported to {csv_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ SCREENING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nScreened: {len(screened_df)} stocks")
    logger.info(f"Predicted: {len(predictions_df)} stocks")
    logger.info(f"Stored: {len(predictions_list)} top predictions")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
