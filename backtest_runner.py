#!/usr/bin/env python3
"""
Backtest Runner - FIXED VERSION
Runs existing strategies instead of creating duplicates
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config

# Import NEW versions
from backtesting.strategy_backtester import StrategyBacktester
from backtesting.backtest_supabase_client import BacktestSupabaseClient


def run_backtest(
    strategy_config: Dict[str, Any],
    config: Optional[Dict] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run a backtest with given strategy configuration
    DOES NOT create a new strategy - expects strategy_id in config
    
    Args:
        strategy_config: Strategy configuration dictionary (must include 'id' or 'strategy_id')
        config: Optional system config (loads from file if not provided)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with backtest results and strategy_id
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        backtester = StrategyBacktester(config)
        supabase = BacktestSupabaseClient(config)
        
        # Get strategy_id from config
        strategy_id = strategy_config.get('id') or strategy_config.get('strategy_id')
        
        if not strategy_id:
            raise ValueError("Strategy config must include 'id' or 'strategy_id'")
        
        logger.info(f"Running backtest for strategy ID: {strategy_id}")
        
        # Update status to running
        supabase.update_strategy_status(strategy_id, 'running')
        
        # Run backtest
        logger.info("Running backtest...")
        results = backtester.run_backtest(
            strategy_config, 
            supabase,
            progress_callback
        )
        
        # Write results to Supabase
        logger.info("Writing results to Supabase...")
        
        # Write daily results
        supabase.write_daily_results(strategy_id, results['daily_results'])
        
        # Write trades
        supabase.write_trades(strategy_id, results['trades'])
        
        # Update strategy summary
        supabase.update_strategy_summary(strategy_id, results['overall_stats'])
        
        # Update status to completed
        supabase.update_strategy_status(strategy_id, 'completed')
        
        logger.info(f"✓ Backtest completed successfully. Strategy ID: {strategy_id}")
        
        return {
            'strategy_id': strategy_id,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        
        # Update status to failed if we have strategy_id
        try:
            if 'strategy_id' in locals():
                supabase.update_strategy_status(strategy_id, 'failed')
        except:
            pass
        
        raise


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description="Run a trading strategy backtest"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--strategy-id",
        type=int,
        required=True,
        help="Strategy ID from database"
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, config.get("logging", {}))
    
    try:
        # Initialize Supabase client
        supabase = BacktestSupabaseClient(config)
        
        # Fetch strategy from database
        strategy_config = supabase.get_strategy(args.strategy_id)
        
        if not strategy_config:
            logger.error(f"Strategy {args.strategy_id} not found in database")
            return 1
        
        logger.info(f"Loaded strategy: {strategy_config['name']}")
        
        # Run backtest
        result = run_backtest(strategy_config, config)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Strategy ID: {result['strategy_id']}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
