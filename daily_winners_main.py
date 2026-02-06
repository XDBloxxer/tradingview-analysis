#!/usr/bin/env python3
"""
Daily Winners Tracker - Main entry point
Tracks top 10 daily winners and captures indicators at market open, close, and T-1 (open & close)
Completely separate from the spike/grinder analysis system
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.daily_winners_detector import DailyWinnersDetector
from src.intraday_data_collector import IntradayDataCollector
from src.daily_winners_supabase_client import DailyWinnersSupabaseClient
from src.utils import setup_logging, load_config


def main():
    """Main execution function for daily winners tracker"""
    parser = argparse.ArgumentParser(
        description="Daily Winners Tracker - Track top performers with intraday indicators"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top winners to track (default: 10)"
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
    
    # Parse target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return 1
    else:
        target_date = datetime.now()
    
    target_date_str = target_date.date().isoformat()
    
    try:
        logger.info("=" * 60)
        logger.info("DAILY WINNERS TRACKER (ENHANCED)")
        logger.info(f"Target Date: {target_date_str}")
        logger.info(f"Top N: {args.top_n}")
        logger.info("=" * 60)
        
        # Step 1: Detect Top Winners (4pm NYC)
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 1: DETECT TOP WINNERS (Market Close 4pm)")
        logger.info("=" * 60)
        logger.info("This identifies the top performers at end of day")
        
        detector = DailyWinnersDetector(config)
        winners = detector.detect_top_winners(top_n=args.top_n, target_date=target_date)
        
        if not winners:
            logger.warning("No winners detected. Exiting.")
            return 0
        
        logger.info(f"✓ Detected {len(winners)} winners")
        logger.info(f"  Top winner: {winners[0]['symbol']} (+{winners[0]['change_pct']:.2f}%)")
        logger.info("")
        logger.info(f"  Winners: {', '.join([w['symbol'] for w in winners])}")
        
        # Write winners to Supabase
        supabase = DailyWinnersSupabaseClient(config)
        count = supabase.write_winners(winners)
        logger.info(f"✓ Written {count} winners to Supabase")
        
        # Step 2: Collect Intraday Indicator Data for THE SAME WINNERS
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: COLLECT ENHANCED INTRADAY INDICATOR DATA FOR WINNERS")
        logger.info(f"Collecting indicators for the same {len(winners)} winning stocks")
        logger.info("=" * 60)
        logger.info("Capturing indicators at FOUR time points:")
        logger.info("  - Market Open (9:30am) - Current Day")
        logger.info("  - Market Close (4pm) - Current Day")
        logger.info("  - Day Prior Open (9:30am) - Previous Day")
        logger.info("  - Day Prior Close (4pm) - Previous Day")
        logger.info("")
        logger.info("Enhanced with comprehensive technical indicators:")
        logger.info("  • Momentum: RSI, Stochastic, Williams %R, ROC, TSI, KAMA")
        logger.info("  • Trend: MACD, ADX, CCI, Aroon, PSAR, Vortex, Mass Index")
        logger.info("  • Volatility: Bollinger Bands, ATR, Keltner, Donchian")
        logger.info("  • Volume: OBV, CMF, Force Index, EOM, VPT, NVI")
        logger.info("  • Moving Averages: EMA/SMA (5,10,20,50,100,200)")
        logger.info("  • Price Analysis: Multi-period changes, 52-week levels, gaps")
        logger.info("  • Candlestick Patterns: Doji, Hammer, Engulfing")
        logger.info("=" * 60)
        
        collector = IntradayDataCollector(config)
        intraday_data = collector.collect_intraday_data(winners, target_date)
        
        logger.info(f"✓ Collected enhanced indicator data for the SAME {len(winners)} winners:")
        logger.info(f"  - Market Open (9:30am): {len(intraday_data['market_open'])} symbols")
        logger.info(f"  - Market Close (4pm): {len(intraday_data['market_close'])} symbols")
        logger.info(f"  - Day Prior Open (9:30am T-1): {len(intraday_data['day_prior_open'])} symbols")
        logger.info(f"  - Day Prior Close (4pm T-1): {len(intraday_data['day_prior_close'])} symbols")
        
        # Write intraday data to Supabase
        counts = supabase.write_intraday_data(intraday_data)
        logger.info(f"✓ Written intraday data to Supabase:")
        for data_type, count in counts.items():
            logger.info(f"  - {data_type}: {count} rows")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ DAILY WINNERS TRACKER COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("")
        logger.info(f"Data for {target_date_str} saved to Supabase:")
        logger.info(f"  • Identified {len(winners)} top winners at market close (4pm)")
        logger.info(f"  • Captured comprehensive indicators for these SAME stocks at:")
        logger.info(f"    - Market Open (9:30am): {len(intraday_data['market_open'])} stocks")
        logger.info(f"    - Market Close (4pm): {len(intraday_data['market_close'])} stocks")
        logger.info(f"    - Day Prior Open (9:30am T-1): {len(intraday_data['day_prior_open'])} stocks")
        logger.info(f"    - Day Prior Close (4pm T-1): {len(intraday_data['day_prior_close'])} stocks")
        logger.info("")
        logger.info(f"  Winners: {', '.join([w['symbol'] for w in winners])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Daily winners tracker failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
