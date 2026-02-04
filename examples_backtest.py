#!/usr/bin/env python3
"""
Example usage of the backtesting system
Run this to see how to use the backtester programmatically
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.strategy_backtester import StrategyBacktester
from backtesting.backtest_supabase_client import BacktestSupabaseClient
from src.utils import load_config, setup_logging


def example_1_simple_rsi_strategy():
    """
    Example 1: Simple oversold RSI strategy
    Buy when RSI < 30, expect 5% gain in 1 day
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Oversold RSI Strategy")
    print("=" * 60)
    
    config = load_config()
    setup_logging("INFO", config.get("logging", {}))
    
    # Define strategy
    strategy_config = {
        'name': 'Oversold RSI Strategy',
        'description': 'Buy oversold stocks with RSI < 30 and high volume',
        'start_date': '2024-01-01',
        'end_date': '2024-03-31',  # 3 months
        'target_min_gain_pct': 5.0,
        'target_days': 1,
        'indicator_criteria': [
            {'indicator': 'rsi', 'operator': '<', 'value': 30},
            {'indicator': 'volume', 'operator': '>', 'value': 1000000},
            {'indicator': 'close', 'operator': '>', 'value': 5.0}
        ],
        'min_price': 5.0,
        'min_volume': 1000000,
        'exchanges': ['NASDAQ']
    }
    
    # Initialize components
    backtester = StrategyBacktester(config)
    supabase = BacktestSupabaseClient(config)
    
    # Create strategy record
    print("\nCreating strategy in database...")
    strategy_id = supabase.create_strategy(strategy_config)
    print(f"Created strategy ID: {strategy_id}")
    
    # Run backtest
    print("\nRunning backtest...")
    supabase.update_strategy_status(strategy_id, 'running')
    
    results = backtester.run_backtest(strategy_config)
    
    # Save results
    print("\nSaving results...")
    supabase.write_daily_results(strategy_id, results['daily_results'])
    supabase.write_trades(strategy_id, results['trades'])
    supabase.update_strategy_summary(strategy_id, results['overall_stats'])
    supabase.update_strategy_status(strategy_id, 'completed')
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    stats = results['overall_stats']
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Total Matches: {stats['total_matches']}")
    print(f"True Positives: {stats['true_positives']}")
    print(f"False Positives: {stats['false_positives']}")
    print(f"Missed Opportunities: {stats['missed_opportunities']}")
    print(f"Accuracy: {stats['accuracy_pct']}%")
    if stats['avg_gain_pct']:
        print(f"Average Gain: {stats['avg_gain_pct']}%")
    
    return strategy_id


def example_2_volume_breakout():
    """
    Example 2: Volume breakout strategy
    Buy when volume spikes above 2x average
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Volume Breakout Strategy")
    print("=" * 60)
    
    config = load_config()
    
    strategy_config = {
        'name': 'Volume Breakout Strategy',
        'description': 'Buy when volume spikes 2x+ with positive momentum',
        'start_date': '2024-06-01',
        'end_date': '2024-06-30',  # 1 month
        'target_min_gain_pct': 3.0,  # Lower target
        'target_days': 1,
        'indicator_criteria': [
            {'indicator': 'volume_ratio', 'operator': '>', 'value': 2.0},
            {'indicator': 'rsi', 'operator': '>', 'value': 50},
            {'indicator': 'close', 'operator': '>', 'value': 10.0}
        ],
        'min_price': 10.0,
        'min_volume': 500000,
        'exchanges': ['NASDAQ', 'NYSE']
    }
    
    # Run via the main runner function
    from backtest_runner import run_backtest
    
    result = run_backtest(strategy_config, config)
    
    print(f"\nBacktest completed! Strategy ID: {result['strategy_id']}")
    return result['strategy_id']


def example_3_quick_test():
    """
    Example 3: Quick test with just 1 week
    Good for testing the system
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Quick Test (1 Week)")
    print("=" * 60)
    
    config = load_config()
    
    # Use recent dates
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=7)
    
    strategy_config = {
        'name': 'Quick Test Strategy',
        'description': 'Testing the system with 1 week of data',
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'target_min_gain_pct': 5.0,
        'target_days': 1,
        'indicator_criteria': [
            {'indicator': 'volume', 'operator': '>', 'value': 1000000},
            {'indicator': 'close', 'operator': '>', 'value': 1.0}
        ],
        'min_price': 1.0,
        'min_volume': 1000000,
        'exchanges': ['NASDAQ']
    }
    
    from backtest_runner import run_backtest
    
    result = run_backtest(strategy_config, config)
    
    print(f"\nQuick test completed! Strategy ID: {result['strategy_id']}")
    return result['strategy_id']


def example_4_view_results(strategy_id: int):
    """
    Example 4: How to retrieve and analyze results
    """
    print("\n" + "=" * 60)
    print(f"EXAMPLE 4: Viewing Results for Strategy {strategy_id}")
    print("=" * 60)
    
    config = load_config()
    supabase = BacktestSupabaseClient(config)
    
    # Get strategy
    strategy = supabase.get_strategy(strategy_id)
    if not strategy:
        print("Strategy not found!")
        return
    
    print(f"\nStrategy: {strategy['name']}")
    print(f"Period: {strategy['start_date']} to {strategy['end_date']}")
    print(f"Status: {strategy['run_status']}")
    
    if strategy['run_status'] == 'completed':
        print("\nPerformance:")
        print(f"  Total Matches: {strategy.get('total_matches', 0)}")
        print(f"  True Positives: {strategy.get('true_positives', 0)}")
        print(f"  False Positives: {strategy.get('false_positives', 0)}")
        print(f"  Missed Opportunities: {strategy.get('missed_opportunities', 0)}")
        print(f"  Accuracy: {strategy.get('accuracy_pct', 0)}%")
        
        # Get daily results
        daily_df = supabase.get_daily_results(strategy_id)
        if not daily_df.empty:
            print(f"\nDaily Results: {len(daily_df)} days")
            print(daily_df[['test_date', 'criteria_matches', 'true_positives', 'false_positives']].head())
        
        # Get sample trades
        trades_df = supabase.get_trades(strategy_id, limit=10)
        if not trades_df.empty:
            print(f"\nSample Trades:")
            print(trades_df[['symbol', 'signal_date', 'trade_type', 'actual_gain_pct']].head())


def main():
    """Run examples"""
    print("Strategy Backtesting Examples")
    print("=" * 60)
    print("\nChoose an example to run:")
    print("1. Simple RSI Strategy (3 months)")
    print("2. Volume Breakout Strategy (1 month)")
    print("3. Quick Test (1 week) - RECOMMENDED FOR FIRST RUN")
    print("4. View existing results")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-4): ").strip()
    
    if choice == '1':
        strategy_id = example_1_simple_rsi_strategy()
        print(f"\n✓ Strategy created with ID: {strategy_id}")
        print("View results in the dashboard or run example 4")
    
    elif choice == '2':
        strategy_id = example_2_volume_breakout()
        print(f"\n✓ Strategy created with ID: {strategy_id}")
    
    elif choice == '3':
        strategy_id = example_3_quick_test()
        print(f"\n✓ Quick test completed with ID: {strategy_id}")
    
    elif choice == '4':
        strategy_id = input("Enter strategy ID: ").strip()
        try:
            example_4_view_results(int(strategy_id))
        except ValueError:
            print("Invalid strategy ID")
    
    elif choice == '0':
        print("Goodbye!")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
