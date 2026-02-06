#!/usr/bin/env python3
"""
Backtest Runner - UPDATED VERSION
Uses database queries instead of dynamic market scanning
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
    
    Args:
        strategy_config: Strategy configuration dictionary
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
        
        # Create strategy record
        logger.info("Creating strategy record...")
        strategy_id = supabase.create_strategy(strategy_config)
        
        # Update status to running
        supabase.update_strategy_status(strategy_id, 'running')
        
        # Run backtest (PASS supabase client to backtester)
        logger.info("Running backtest...")
        results = backtester.run_backtest(
            strategy_config, 
            supabase,  # NEW: Pass client to backtester
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
        "--name",
        required=True,
        help="Strategy name"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--target-gain",
        type=float,
        default=5.0,
        help="Target gain percentage (default: 5.0)"
    )
    parser.add_argument(
        "--target-days",
        type=int,
        default=1,
        help="Days to hold (default: 1)"
    )
    parser.add_argument(
        "--criteria",
        required=True,
        help="Indicator criteria as JSON string"
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.25,
        help="Minimum stock price"
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=100000,
        help="Minimum volume"
    )
    parser.add_argument(
        "--verbose",
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
    
    # Parse criteria JSON
    import json
    try:
        criteria = json.loads(args.criteria)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid criteria JSON: {e}")
        return 1
    
    # Build strategy config
    strategy_config = {
        'name': args.name,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'target_min_gain_pct': args.target_gain,
        'target_days': args.target_days,
        'indicator_criteria': criteria,
        'min_price': args.min_price,
        'min_volume': args.min_volume
    }
    
    try:
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
