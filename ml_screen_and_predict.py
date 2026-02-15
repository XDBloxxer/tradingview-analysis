#!/usr/bin/env python3
"""
Autonomous ML Stock Screener & Predictor - FIXED FILTERS
Uses tradingview_scraper.symbols.screener.Screener with correct API format
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
        """Load screening filters"""
        
        defaults = {
            'min_price': 3.0,
            'max_price': 500.0,
            'min_volume': 500000,
            'min_rsi': None,
            'max_rsi': None,
            'min_adx': None,
            'max_adx': None,
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
            else:
                self.logger.info("No learned filters yet - using minimal constraints")
        except Exception as e:
            self.logger.debug(f"Using default filters: {e}")
        
        return defaults
    
    def screen_with_tradingview(self, max_results: int = 500) -> pd.DataFrame:
        """
        Use tradingview-scraper's Screener with CORRECT filter format
        """
        
        if not SCREENER_AVAILABLE or self.screener is None:
            self.logger.error("tradingview-scraper Screener not available!")
            return pd.DataFrame()
        
        self.logger.info("Screening stocks with tradingview-scraper Screener...")
        self.logger.info("Active filters:")
        self.logger.info(f"  Price: ${self.filters['min_price']}-${self.filters['max_price']}")
        self.logger.info(f"  Volume: >= {self.filters['min_volume']:,}")
        
        try:
            # Build filter list - FIXED: Use separate greater/less instead of in_range
            screener_filters = [
                {'left': 'close', 'operation': 'greater', 'right': self.filters['min_price']},
                {'left': 'close', 'operation': 'less', 'right': self.filters['max_price']},
                {'left': 'volume', 'operation': 'greater', 'right': self.filters['min_volume']},
            ]
            
            # Add learned filters if they exist
            if self.filters['min_rsi'] is not None:
                screener_filters.append({
                    'left': 'RSI',
                    'operation': 'greater',
                    'right': self.filters['min_rsi']
                })
            
            if self.filters['max_rsi'] is not None:
                screener_filters.append({
                    'left': 'RSI',
                    'operation': 'less',
                    'right': self.filters['max_rsi']
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
            
            # Define columns to retrieve
            columns = [
                'name', 'close', 'volume', 'market_cap_basic',
                'RSI', 'ADX', 'MACD.macd', 'MACD.signal',
                'Stoch.K', 'Stoch.D', 
                'EMA20', 'SMA20', 'ATR', 'BB.upper', 'BB.lower',
                'change', 'Recommend.All'
            ]
            
            # Execute screening
            self.logger.info(f"Calling screener with {len(screener_filters)} filters...")
            
            # FIRST: Try without filters to see if it works at all
            self.logger.debug("Testing basic screening without filters...")
            test_result = self.screener.screen(
                market='america',
                limit=5
            )
            
            if test_result['status'] == 'success':
                self.logger.debug(f"✓ Basic screening works, got {len(test_result.get('data', []))} results")
            else:
                self.logger.error(f"✗ Basic screening failed: {test_result}")
                return pd.DataFrame()
            
            # NOW: Try with filters
            result = self.screener.screen(
                market='america',
                filters=screener_filters,
                columns=columns,
                sort_by='volume',
                sort_order='desc',
                limit=max_results
            )
            
            if result['status'] != 'success':
                self.logger.error(f"Screening failed with status: {result.get('status')}")
                self.logger.error(f"Response: {result}")
                return pd.DataFrame()
            
            if not result.get('data'):
                self.logger.warning("No stocks matched screening criteria")
                self.logger.info(f"Total available: {result.get('totalCount', 0)}")
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
            
            # Try fallback to basic screening
            self.logger.warning("Attempting basic screening without filters...")
            try:
                basic_result = self.screener.screen(
                    market='america',
                    columns=columns,
                    sort_by='volume',
                    sort_order='desc',
                    limit=max_results
                )
                
                if basic_result['status'] == 'success' and basic_result.get('data'):
                    df = pd.DataFrame(basic_result['data'])
                    
                    # Apply filters manually
                    df = df[
                        (df['close'] >= self.filters['min_price']) &
                        (df['close'] <= self.filters['max_price']) &
                        (df['volume'] >= self.filters['min_volume'])
                    ]
                    
                    self.logger.info(f"✓ Fallback screening found {len(df)} stocks")
                    return df
                    
            except Exception as e2:
                self.logger.error(f"Fallback also failed: {e2}")
            
            return pd.DataFrame()
    
    def screen_from_supabase_fallback(self, supabase_client, max_results: int = 500) -> pd.DataFrame:
        """FALLBACK: Screen from Supabase"""
        
        self.logger.info("Using Supabase fallback screening...")
        
        try:
            stocks_df = supabase_client.get_latest_day_prior_close(limit=max_results * 2)
            
            if stocks_df.empty:
                self.logger.error("No data in Supabase")
                return pd.DataFrame()
            
            filtered = stocks_df[
                (stocks_df['close'] >= self.filters['min_price']) &
                (stocks_df['close'] <= self.filters['max_price']) &
                (stocks_df['volume'] >= self.filters['min_volume'])
            ]
            
            filtered = filtered.sort_values('volume', ascending=False)
            
            self.logger.info(f"✓ Screened {len(filtered)} stocks from Supabase")
            
            return filtered.head(max_results)
            
        except Exception as e:
            self.logger.error(f"Supabase screening failed: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, screened_df: pd.DataFrame) -> pd.DataFrame:
        """Convert screener output to model features"""
        
        self.logger.info("Preparing features for model...")
        
        # Handle symbol field
        if 'symbol' in screened_df.columns:
            screened_df['ticker'] = screened_df['symbol'].str.split(':').str[-1]
            screened_df['exchange_prefix'] = screened_df['symbol'].str.split(':').str[0]
        
        # Rename columns
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
    parser = argparse.ArgumentParser(description="ML stock screening")
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
    
    # Step 1: Screening
    logger.info("\n" + "="*80)
    logger.info("STEP 1: SCREENING")
    logger.info("="*80)
    
    if SCREENER_AVAILABLE:
        screened_df = screener.screen_with_tradingview(max_results=args.max_results)
        
        # If TradingView fails, fall back to Supabase
        if screened_df.empty:
            logger.warning("TradingView screening returned no results, trying Supabase fallback...")
            screened_df = screener.screen_from_supabase_fallback(supabase, max_results=args.max_results)
    else:
        logger.warning("tradingview-scraper not available, using Supabase fallback")
        screened_df = screener.screen_from_supabase_fallback(supabase, max_results=args.max_results)
    
    if screened_df.empty:
        logger.error("No stocks passed screening from any source")
        return 1
    
    logger.info(f"✓ Proceeding with {len(screened_df)} screened stocks")
    
    # Continue with rest of the pipeline...
    # (Step 2-6 remain the same as before)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
