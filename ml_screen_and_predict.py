#!/usr/bin/env python3
"""
Autonomous ML Stock Screener & Predictor - DEBUG VERSION
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

# Import tradingview-scraper WITH DETAILED ERROR REPORTING
print("=" * 80)
print("ATTEMPTING TO IMPORT tradingview-scraper...")
print("=" * 80)

try:
    print("Step 1: Importing tradingview_scraper module...")
    import tradingview_scraper
    print(f"✓ SUCCESS - Module imported from: {tradingview_scraper.__file__}")
    
    print("\nStep 2: Checking module contents...")
    available = [x for x in dir(tradingview_scraper) if not x.startswith('_')]
    print(f"Available items: {available}")
    
    print("\nStep 3: Importing Query and Column...")
    from tradingview_scraper import Query, Column
    print("✓ SUCCESS - Query and Column imported")
    
    SCREENER_AVAILABLE = True
    print("\n" + "=" * 80)
    print("✓✓✓ tradingview-scraper IS AVAILABLE ✓✓✓")
    print("=" * 80 + "\n")
    
except ImportError as e:
    print(f"\n✗✗✗ IMPORT FAILED ✗✗✗")
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
    print("\nPython path:")
    for p in sys.path:
        print(f"  - {p}")
    
    print("\n" + "=" * 80)
    SCREENER_AVAILABLE = False
    print("⚠️  tradingview-scraper not available")
    print("   Install with: pip install tradingview-scraper")
    print("=" * 80 + "\n")


def setup_logging(verbose: bool = False):
    """Setup basic logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# [Rest of the file remains the same - just copying key parts for brevity]

class SmartScreener:
    """
    Intelligent screener that uses tradingview-scraper
    """
    
    def __init__(self, config: dict = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.filters = self._load_learned_filters()
    
    def _load_learned_filters(self) -> dict:
        defaults = {
            'min_price': 3.0,
            'max_price': 500.0,
            'min_volume': 500000,
            'min_rsi': None,
            'max_rsi': None,
            'min_adx': None,
            'max_adx': None,
            'min_macd': None,
            'max_macd': None,
            'min_volume_change': None,
            'max_volume_change': None,
            'market_cap_max': None,
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
        if not SCREENER_AVAILABLE:
            self.logger.error("tradingview-scraper not installed!")
            self.logger.error("Check the import debug output above for details")
            return pd.DataFrame()
        
        self.logger.info("Screening stocks with tradingview-scraper...")
        # ... rest of implementation
        return pd.DataFrame()  # Placeholder
    
    def prepare_features(self, screened_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Preparing features for model...")
        return screened_df  # Placeholder


def main():
    parser = argparse.ArgumentParser(description="DEBUG ML screening")
    parser.add_argument("--universe", type=str, default="auto")
    parser.add_argument("--max-workers", type=int, default=15)
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    logger.info("="*80)
    logger.info("DEBUG ML STOCK SCREENING")
    logger.info("="*80)
    
    if not SCREENER_AVAILABLE:
        logger.error("Cannot proceed - tradingview-scraper not available")
        logger.error("Review the detailed import error above")
        return 1
    
    logger.info("✓ All imports successful, proceeding with screening...")
    # ... rest would continue normally
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
