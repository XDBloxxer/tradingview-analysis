"""
Backtesting module for strategy testing
"""

__version__ = "1.0.0"

from .strategy_backtester import StrategyBacktester
from .backtest_supabase_client import BacktestSupabaseClient

__all__ = [
    'StrategyBacktester',
    'BacktestSupabaseClient'
]
