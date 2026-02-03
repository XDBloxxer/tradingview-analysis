#!/usr/bin/env python3
"""
Main entry point for TradingView Stock Event Analysis System
Writes to Supabase (primary) and Google Sheets (sample for validation)
"""

import argparse
import logging
import sys
from pathlib import Path

from src.event_detector import EventDetector
from src.data_collector import DataCollector
from src.analyzer import Analyzer
from src.utils import setup_logging, load_config


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="TradingView Stock Event Analysis System"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run event detection only"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Run data collection only"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline (detect → collect → analyze)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--skip-sheets",
        action="store_true",
        help="Skip Google Sheets sample writes (Supabase only)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logging(log_level, config.get("logging", {}))
    
    # Determine what to run
    run_detect = args.detect or args.all
    run_collect = args.collect or args.all
    run_analyze = args.analyze or args.all
    
    # Default to running all if no flags specified
    if not (run_detect or run_collect or run_analyze):
        run_detect = run_collect = run_analyze = True
    
    try:
        # Step 1: Event Detection
        if run_detect:
            logger.info("=" * 60)
            logger.info("STEP 1: EVENT DETECTION")
            logger.info("=" * 60)
            
            detector = EventDetector(config)
            events = detector.detect_events()
            
            logger.info(f"✓ Detected {len(events)} events")
            logger.info(f"  - Spikers: {len([e for e in events if e['Event_Type'] == 'Spiker'])}")
            logger.info(f"  - Grinders: {len([e for e in events if e['Event_Type'] == 'Grinder'])}")
            
            # Write to Supabase (all data)
            from src.supabase_client import SupabaseClient
            supabase = SupabaseClient(config)
            count = supabase.write_candidates(events)
            logger.info(f"✓ Written {count} events to Supabase")
            
            # Write sample to Google Sheets for validation
            if not args.skip_sheets:
                detector.write_to_sheets(events)
                logger.info("✓ Written sample to Google Sheets (Candidates)")
        
        # Step 2: Data Collection
        if run_collect:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 2: INDICATOR DATA COLLECTION")
            logger.info("=" * 60)
            
            collector = DataCollector(config)
            raw_data = collector.collect_indicator_data()
            
            logger.info(f"✓ Collected data for {len(raw_data)} symbol-indicator combinations")
            
            # Write to Supabase (all data)
            collector.write_to_supabase(raw_data)
            logger.info("✓ Written all data to Supabase (Raw Data)")
            
            # Write sample to Google Sheets for validation
            if not args.skip_sheets:
                collector.write_to_sheets(raw_data)
                logger.info("✓ Written sample to Google Sheets (Raw Data)")
        
        # Step 3: Analysis
        if run_analyze:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 3: DATA ANALYSIS")
            logger.info("=" * 60)
            
            analyzer = Analyzer(config)
            analysis_results = analyzer.analyze()
            
            logger.info(f"✓ Generated analysis with {len(analysis_results['summary'])} metrics")
            
            # Write to Supabase (all data)
            analyzer.write_to_supabase(analysis_results)
            logger.info("✓ Written all analysis to Supabase")
            
            # Write sample to Google Sheets for validation
            if not args.skip_sheets:
                analyzer.write_to_sheets(analysis_results)
                logger.info("✓ Written sample to Google Sheets (Analysis & Summary Stats)")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Data Storage:")
        logger.info("  • Primary: Supabase (all data)")
        if not args.skip_sheets:
            logger.info("  • Validation: Google Sheets (sample data)")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
