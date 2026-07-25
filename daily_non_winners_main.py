#!/usr/bin/env python3
"""
Daily NON-Winners Tracker
Collects negative examples (stocks that did NOT explode) for ML training

This is critical for model training to learn what NOT to predict.
Without negative examples, the model only sees winners and can't learn discrimination.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from pandas.tseries.holiday import USFederalHolidayCalendar

from src.daily_non_winners_detector import DailyNonWinnersDetector
from src.intraday_data_collector import IntradayDataCollector
from src.daily_non_winners_supabase_client import DailyNonWinnersSupabaseClient
from src.multiday_feature_collector import MultidayFeatureCollector
from src.utils import setup_logging, load_config


def is_trading_day(d) -> bool:
    """Return True if d is a NYSE trading day (not a weekend or US federal holiday)."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    holidays = USFederalHolidayCalendar().holidays(start=str(d), end=str(d))
    return len(holidays) == 0


def trading_days_between(start_date, end_date):
    """Yield each NYSE trading day from start_date to end_date, inclusive."""
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    d = start_date
    one_day = timedelta(days=1)
    while d <= end_date:
        if is_trading_day(d):
            yield d
        d += one_day


def run_for_date(target_date: datetime, args, config, logger) -> int:
    """
    Runs the full non-winners-detection pipeline for a single target_date.
    Identical to the previous single-date main() body — used for both
    normal (single-day) runs and each day inside a --start-date/--end-date
    backfill range.
    """
    target_date_str = target_date.date().isoformat()

    # ── Market holiday / weekend guard ──────────────────────────────────────
    if not is_trading_day(target_date.date()):
        if target_date.date().weekday() >= 5:
            reason = "weekend"
        else:
            reason = "US market holiday"
        logger.info("=" * 60)
        logger.info("DAILY NON-WINNERS TRACKER — SKIPPED")
        logger.info(f"  {target_date_str} is a {reason}.")
        logger.info("  No market data to collect. Exiting cleanly.")
        logger.info("=" * 60)
        return 0
    # ────────────────────────────────────────────────────────────────────────

    try:
        logger.info("=" * 60)
        logger.info("DAILY NON-WINNERS TRACKER")
        logger.info(f"Target Date: {target_date_str}")
        logger.info(f"Top N: {args.top_n}")
        if args.allow_append:
            logger.warning("ALLOW APPEND: enabled — existing date records CAN be supplemented")
        else:
            logger.info("Allow Overwrite: disabled (safe default)")
        logger.info("=" * 60)
        logger.info("")
        logger.info("PURPOSE: Collect NEGATIVE examples for ML training")
        logger.info("These are stocks that did NOT explode - essential for learning!")
        logger.info("")

        # Step 1: Detect Non-Winners
        logger.info("=" * 60)
        logger.info("STEP 1: DETECT NON-WINNERS")
        logger.info("=" * 60)
        logger.info("Finding stocks that did NOT explode today")

        detector = DailyNonWinnersDetector(config)
        non_winners = detector.detect_non_winners(top_n=args.top_n, target_date=target_date)

        if not non_winners:
            logger.warning("No non-winners detected. Exiting.")
            return 0

        logger.info(f"✓ Detected {len(non_winners)} non-winners")
        logger.info(f"  Sample: {non_winners[0]['symbol']} ({non_winners[0]['change_pct']:+.2f}%)")
        logger.info("")
        logger.info(f"  Non-winners: {', '.join([w['symbol'] for w in non_winners])}")

        # Write non-winners to Supabase
        supabase = DailyNonWinnersSupabaseClient(config)
        count = supabase.write_non_winners(non_winners, allow_append=args.allow_append)
        logger.info(f"✓ Written {count} non-winners to Supabase")

        # Step 2: Collect Indicator Data
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: COLLECT INDICATOR DATA FOR NON-WINNERS")
        logger.info(f"Collecting indicators for the same {len(non_winners)} non-winning stocks")
        logger.info("=" * 60)
        logger.info("Capturing indicators at FOUR time points:")
        logger.info("  - Market Open (9:30am) - Current Day")
        logger.info("  - Market Close (4pm) - Current Day")
        logger.info("  - Day Prior Open (9:30am T-1)")
        logger.info("  - Day Prior Close (4pm T-1)")
        logger.info("")
        logger.info("These will serve as NEGATIVE examples in training:")
        logger.info("  Label: 0 (did NOT explode)")
        logger.info("")

        collector = IntradayDataCollector(config)
        intraday_data = collector.collect_intraday_data(non_winners, target_date)

        logger.info(f"✓ Collected indicator data for {len(non_winners)} non-winners:")
        logger.info(f"  - Market Open (9:30am): {len(intraday_data['market_open'])} symbols")
        logger.info(f"  - Market Close (4pm): {len(intraday_data['market_close'])} symbols")
        logger.info(f"  - Day Prior Open (9:30am T-1): {len(intraday_data['day_prior_open'])} symbols")
        logger.info(f"  - Day Prior Close (4pm T-1): {len(intraday_data['day_prior_close'])} symbols")

        # Write ALL intraday data to Supabase (all four timepoints)
        counts = supabase.write_intraday_data(intraday_data, allow_append=args.allow_append)
        logger.info(f"✓ Written intraday data to Supabase:")
        for data_type, count in counts.items():
            logger.info(f"  - {data_type}: {count} rows")

        # Step 3: Compute and store T-3 / T-5 / T-10 multiday features
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: COMPUTE T-3/T-5/T-10 MULTIDAY FEATURES FOR NON-WINNERS")
        logger.info("=" * 60)
        multiday_collector = MultidayFeatureCollector(config)
        multiday_count = multiday_collector.collect_and_write(
            stocks=intraday_data["day_prior_close"],   # has symbol + detection_date
            table="non_winners_multiday",
            allow_append=args.allow_append,
        )
        logger.info(f"✓ Written {multiday_count} multiday feature rows for non-winners")

        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ DAILY NON-WINNERS TRACKER COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("")
        logger.info(f"Data for {target_date_str} saved to Supabase:")
        logger.info(f"  • Identified {len(non_winners)} non-winners (negative examples)")
        logger.info(f"  • Captured indicators for these stocks at all timepoints")
        logger.info(f"    - Market Open: {len(intraday_data['market_open'])} stocks")
        logger.info(f"    - Market Close: {len(intraday_data['market_close'])} stocks")
        logger.info(f"    - Day Prior Open: {len(intraday_data['day_prior_open'])} stocks")
        logger.info(f"    - Day Prior Close: {len(intraday_data['day_prior_close'])} stocks")
        logger.info("")
        logger.info(f"  Non-winners: {', '.join([w['symbol'] for w in non_winners])}")
        logger.info("")
        logger.info("These negative examples will be used in model training to teach")
        logger.info("the model what patterns do NOT lead to explosions.")
        logger.info("")
        logger.info("NOTE: According to your workflow, T-1 data (day_prior_close & day_prior_open)")
        logger.info("      is MOST important for training. Same-day data is supplementary.")

        return 0

    except Exception as e:
        logger.error(f"✗ Daily non-winners tracker failed for {target_date_str}: {str(e)}", exc_info=True)
        return 1


def main():
    """Main execution function for daily non-winners tracker"""
    parser = argparse.ArgumentParser(
        description="Daily NON-Winners Tracker - Collect negative training examples"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD). Defaults to today. Ignored if --start-date is given."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Backfill: first date (YYYY-MM-DD) of a range to process, inclusive. Requires --end-date."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Backfill: last date (YYYY-MM-DD) of a range to process, inclusive. Requires --start-date."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=60,
        help="Number of non-winners to track (default: 60)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--allow-append",
        action="store_true",
        default=False,
        help=(
            "Allow appending stocks to dates that already exist in the database. "
            "Disabled by default to prevent accidental overwrites during scheduled runs. "
            "Only use this flag when running manually."
        )
    )

    args = parser.parse_args()

    if bool(args.start_date) != bool(args.end_date):
        print("Error: --start-date and --end-date must be provided together.", file=sys.stderr)
        return 1

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(log_level, config.get("logging", {}))

    # ── Range backfill mode ──────────────────────────────────────────────
    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Invalid --start-date/--end-date: {e}. Use YYYY-MM-DD")
            return 1

        dates = list(trading_days_between(start_date, end_date))
        logger.info("=" * 60)
        logger.info("DAILY NON-WINNERS TRACKER — BACKFILL RANGE")
        logger.info(f"  {start_date.isoformat()} → {end_date.isoformat()} "
                    f"({len(dates)} trading days, weekends/holidays skipped)")
        logger.info("=" * 60)

        results = {}
        for d in dates:
            target_date = datetime.combine(d, datetime.min.time())
            results[d.isoformat()] = run_for_date(target_date, args, config, logger)

        failed = [d for d, rc in results.items() if rc != 0]
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKFILL RANGE SUMMARY")
        logger.info(f"  Processed: {len(results)} trading days")
        logger.info(f"  Succeeded: {len(results) - len(failed)}")
        if failed:
            logger.info(f"  Failed:    {len(failed)} → {', '.join(failed)}")
        logger.info("=" * 60)

        return 1 if failed else 0

    # ── Single-date mode (normal GitHub Actions runs, or one-off backfill) ──
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return 1
    else:
        target_date = datetime.now()

    return run_for_date(target_date, args, config, logger)


if __name__ == "__main__":
    sys.exit(main())
