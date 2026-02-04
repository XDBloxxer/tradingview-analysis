#!/usr/bin/env python3
"""
Strategy Backtest Runner - Main entry point
Run backtests from command line or programmatically
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

from strategy_backtester import StrategyBacktester
from backtest_supabase_client import BacktestSupabaseClient
from src.utils import setup_logging, load_config


def main():
    """Main execution function for strategy backtest"""
    parser = argparse.ArgumentParser(
        description="Strategy Backtester - Test indicator-based strategies"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--target-gain",
        type=float,
        required=True,
        help="Target gain percentage (e.g., 5.0 for 5%%)"
    )
    parser.add_argument(
        "--holding-days",
        type=int,
        default=1,
        help="Number of days to hold position (default: 1)"
    )
    parser.add_argument(
        "--criteria",
        type=str,
        help="JSON string with strategy criteria (e.g., '{\"volume\": {\"min\": 5000000}, \"rsi\": {\"max\": 30}}')"
    )
    parser.add_argument(
        "--criteria-file",
        type=str,
        help="Path to JSON file with strategy criteria"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs='+',
        help="Specific symbols to test (optional, default uses built-in universe)"
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="Custom Strategy",
        help="Name for this strategy"
    )
    parser.add_argument(
        "--save-to-supabase",
        action="store_true",
        help="Save results to Supabase"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(log_level, config.get("logging", {}))
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1
    
    # Parse criteria
    if args.criteria_file:
        with open(args.criteria_file, 'r') as f:
            criteria = json.load(f)
    elif args.criteria:
        criteria = json.loads(args.criteria)
    else:
        # Default example criteria
        criteria = {
            'volume': {'min': 5000000},
            'rsi': {'max': 30},
            'price': {'min': 1.0, 'max': 50.0}
        }
        logger.info("Using default example criteria")
    
    try:
        logger.info("=" * 60)
        logger.info("STRATEGY BACKTEST")
        logger.info("=" * 60)
        logger.info(f"Strategy: {args.strategy_name}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Target gain: {args.target_gain}%")
        logger.info(f"Holding period: {args.holding_days} days")
        logger.info(f"Criteria: {json.dumps(criteria, indent=2)}")
        
        # Initialize backtester
        backtester = StrategyBacktester(config)
        
        # Run backtest
        results = backtester.backtest_strategy(
            strategy_criteria=criteria,
            start_date=start_date,
            end_date=end_date,
            target_gain_pct=args.target_gain,
            holding_days=args.holding_days,
            symbol_universe=args.symbols
        )
        
        # Add strategy name
        results['strategy_name'] = args.strategy_name
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 60)
        summary = results['summary']
        logger.info(f"Total Signals: {summary['total_signals']}")
        logger.info(f"Successful Hits: {summary['successful_hits']} ({summary['success_rate']:.1f}%)")
        logger.info(f"False Positives: {summary['false_positives']} ({summary['false_positive_rate']:.1f}%)")
        logger.info(f"Missed Opportunities: {summary['missed_opportunities']}")
        logger.info(f"Avg Gain on Hits: {summary['avg_gain_on_hits']:.2f}%")
        logger.info(f"Avg Loss on Misses: {summary['avg_loss_on_misses']:.2f}%")
        logger.info(f"Total Return: {summary['total_return']:.2f}%")
        
        # Save to Supabase if requested
        if args.save_to_supabase:
            logger.info("")
            logger.info("Saving results to Supabase...")
            supabase_client = BacktestSupabaseClient(config)
            backtest_id = supabase_client.write_backtest_results(results)
            logger.info(f"✓ Saved to Supabase with ID: {backtest_id}")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Backtest failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
