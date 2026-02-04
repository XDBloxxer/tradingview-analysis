"""
Strategy Backtester - Test indicator-based strategies against historical data
Finds stocks matching criteria, tracks hits, misses, and false positives
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yfinance as yf

from src.rate_limiter import RateLimiter
from src.utils import get_indicator_mapping

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, UltimateOscillator, AwesomeOscillatorIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class StrategyBacktester:
    """
    Backtests trading strategies by:
    1. Finding stocks matching indicator criteria on each date
    2. Checking if price went up by target amount
    3. Tracking hits, misses, and false positives
    """
    
    MAX_WORKERS = 10
    LOOKBACK_DAYS = 200  # Days of historical data to fetch
    
    def __init__(self, config: dict):
        """
        Initialize strategy backtester
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        
        # Stock universe filters
        detection_config = config.get("detection", {})
        self.exchanges = detection_config.get("exchanges", ["NASDAQ", "NYSE", "AMEX"])
        
        # Cache for historical data
        self.cache = {}
        
        # Statistics
        self.stats = {
            'total_symbols_scanned': 0,
            'data_fetch_success': 0,
            'data_fetch_failed': 0,
            'cached_hits': 0
        }
        
        self.logger.info("Strategy backtester initialized")
    
    def backtest_strategy(
        self,
        strategy_criteria: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        target_gain_pct: float,
        holding_days: int = 1,
        symbol_universe: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy across date range
        
        Args:
            strategy_criteria: Dictionary of indicator conditions
                Example: {
                    'volume': {'min': 5000000},
                    'rsi': {'max': 30},
                    'price': {'min': 1.0, 'max': 50.0}
                }
            start_date: Start date for backtest
            end_date: End date for backtest
            target_gain_pct: Target percentage gain to consider success
            holding_days: Number of days to hold position (default: 1)
            symbol_universe: Optional list of symbols to test (if None, uses default universe)
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING STRATEGY BACKTEST")
        self.logger.info("=" * 60)
        self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Target gain: {target_gain_pct}% in {holding_days} days")
        self.logger.info(f"Strategy criteria: {strategy_criteria}")
        
        # Get symbol universe
        if symbol_universe is None:
            symbol_universe = self._get_default_symbol_universe()
        
        self.logger.info(f"Testing {len(symbol_universe)} symbols")
        
        # Generate trading dates (business days)
        all_dates = pd.bdate_range(start_date, end_date).tolist()
        self.logger.info(f"Testing {len(all_dates)} trading days")
        
        # Results tracking
        all_results = []
        
        # Test each date
        for test_date in tqdm(all_dates, desc="Testing dates"):
            test_date_obj = test_date.to_pydatetime()
            
            # Find stocks matching criteria on this date
            matches = self._find_matching_stocks(
                symbol_universe,
                test_date_obj,
                strategy_criteria
            )
            
            if not matches:
                continue
            
            # Check outcomes for each match
            for match in matches:
                outcome = self._check_outcome(
                    match['symbol'],
                    test_date_obj,
                    target_gain_pct,
                    holding_days
                )
                
                if outcome:
                    result = {
                        'date': test_date_obj.date().isoformat(),
                        'symbol': match['symbol'],
                        'entry_price': outcome['entry_price'],
                        'exit_price': outcome['exit_price'],
                        'actual_gain_pct': outcome['actual_gain_pct'],
                        'hit_target': outcome['hit_target'],
                        'indicator_values': match['indicator_values']
                    }
                    all_results.append(result)
        
        # Also find missed opportunities (price went up but didn't match criteria)
        missed_opportunities = self._find_missed_opportunities(
            symbol_universe,
            all_dates,
            strategy_criteria,
            target_gain_pct,
            holding_days
        )
        
        # Calculate summary statistics
        summary = self._calculate_summary(
            all_results,
            missed_opportunities,
            target_gain_pct
        )
        
        self.logger.info("=" * 60)
        self.logger.info("BACKTEST COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total signals: {summary['total_signals']}")
        self.logger.info(f"Successful hits: {summary['successful_hits']} ({summary['success_rate']:.1f}%)")
        self.logger.info(f"False positives: {summary['false_positives']} ({summary['false_positive_rate']:.1f}%)")
        self.logger.info(f"Missed opportunities: {summary['missed_opportunities']}")
        
        return {
            'summary': summary,
            'detailed_results': all_results,
            'missed_opportunities': missed_opportunities,
            'strategy_criteria': strategy_criteria,
            'date_range': {
                'start': start_date.date().isoformat(),
                'end': end_date.date().isoformat()
            },
            'target_gain_pct': target_gain_pct,
            'holding_days': holding_days
        }
    
    def _get_default_symbol_universe(self) -> List[str]:
        """
        Get default universe of symbols to test
        Uses a predefined list of liquid stocks
        
        Returns:
            List of stock symbols
        """
        # Common liquid stocks across exchanges
        # In production, you'd fetch this from TradingView screener
        default_symbols = [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'LLY', 'DHR', 'BMY',
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'DIS', 'SBUX',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
            # Industrial
            'BA', 'GE', 'CAT', 'HON', 'UPS', 'LMT', 'MMM', 'DE', 'FDX', 'RTX'
        ]
        
        self.logger.info(f"Using default universe of {len(default_symbols)} symbols")
        return default_symbols
    
    def _find_matching_stocks(
        self,
        symbols: List[str],
        test_date: datetime,
        criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find stocks matching criteria on a specific date
        
        Args:
            symbols: List of symbols to check
            test_date: Date to check criteria on
            criteria: Strategy criteria
            
        Returns:
            List of matching stocks with their indicator values
        """
        matches = []
        
        # Process in parallel for speed
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(self._check_criteria_for_symbol, symbol, test_date, criteria): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                try:
                    result = future.result()
                    if result:
                        matches.append(result)
                except Exception as e:
                    # Silently skip errors
                    pass
        
        return matches
    
    def _check_criteria_for_symbol(
        self,
        symbol: str,
        test_date: datetime,
        criteria: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a symbol matches criteria on a date
        
        Args:
            symbol: Stock symbol
            test_date: Date to check
            criteria: Strategy criteria
            
        Returns:
            Dictionary with symbol and indicator values if match, None otherwise
        """
        # Get historical data
        indicators_df = self._get_historical_data(symbol, test_date)
        
        if indicators_df is None or indicators_df.empty:
            return None
        
        # Convert index to date-only
        indicators_df.index = pd.to_datetime(indicators_df.index).date
        test_date_only = test_date.date()
        
        # Get data for test date
        if test_date_only not in indicators_df.index:
            # Try to find closest prior date
            available_dates = [d for d in indicators_df.index if d <= test_date_only]
            if not available_dates:
                return None
            test_date_only = available_dates[-1]
        
        day_data = indicators_df.loc[test_date_only]
        
        # Check each criterion
        indicator_values = {}
        
        for indicator_name, conditions in criteria.items():
            # Get indicator value
            value = day_data.get(indicator_name.lower())
            
            if pd.isna(value):
                return None  # Missing required indicator
            
            indicator_values[indicator_name] = float(value)
            
            # Check conditions
            if 'min' in conditions:
                if value < conditions['min']:
                    return None
            
            if 'max' in conditions:
                if value > conditions['max']:
                    return None
            
            if 'equals' in conditions:
                if value != conditions['equals']:
                    return None
        
        # All criteria met
        return {
            'symbol': symbol,
            'date': test_date_only.isoformat(),
            'indicator_values': indicator_values
        }
    
    def _check_outcome(
        self,
        symbol: str,
        entry_date: datetime,
        target_gain_pct: float,
        holding_days: int
    ) -> Optional[Dict[str, Any]]:
        """
        Check the outcome of a trade
        
        Args:
            symbol: Stock symbol
            entry_date: Entry date
            target_gain_pct: Target gain percentage
            holding_days: Number of days to hold
            
        Returns:
            Dictionary with outcome details
        """
        # Get historical data
        indicators_df = self._get_historical_data(entry_date, entry_date + timedelta(days=holding_days + 5))
        
        if indicators_df is None or indicators_df.empty:
            return None
        
        indicators_df.index = pd.to_datetime(indicators_df.index).date
        entry_date_only = entry_date.date()
        
        # Get entry price
        if entry_date_only not in indicators_df.index:
            available_dates = [d for d in indicators_df.index if d <= entry_date_only]
            if not available_dates:
                return None
            entry_date_only = available_dates[-1]
        
        entry_price = indicators_df.loc[entry_date_only, 'close']
        
        # Get exit date
        exit_date = entry_date_only + timedelta(days=holding_days)
        
        # Find closest available exit date
        available_dates = [d for d in indicators_df.index if d >= exit_date]
        if not available_dates:
            return None
        
        exit_date_actual = available_dates[0]
        exit_price = indicators_df.loc[exit_date_actual, 'close']
        
        # Calculate gain
        actual_gain_pct = ((exit_price - entry_price) / entry_price) * 100
        hit_target = actual_gain_pct >= target_gain_pct
        
        return {
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'actual_gain_pct': float(actual_gain_pct),
            'hit_target': hit_target,
            'entry_date': entry_date_only.isoformat(),
            'exit_date': exit_date_actual.isoformat()
        }
    
    def _find_missed_opportunities(
        self,
        symbols: List[str],
        dates: List[pd.Timestamp],
        criteria: Dict[str, Any],
        target_gain_pct: float,
        holding_days: int
    ) -> List[Dict[str, Any]]:
        """
        Find stocks that hit target gain but didn't match criteria
        
        Args:
            symbols: Symbol universe
            dates: Dates to check
            criteria: Strategy criteria
            target_gain_pct: Target gain
            holding_days: Holding period
            
        Returns:
            List of missed opportunities
        """
        self.logger.info("Searching for missed opportunities...")
        
        missed = []
        
        # Sample a subset to avoid excessive computation
        sample_size = min(len(symbols) * len(dates) // 20, 1000)  # Limit to reasonable number
        
        count = 0
        for test_date in tqdm(dates, desc="Finding missed opportunities", disable=True):
            if count >= sample_size:
                break
            
            test_date_obj = test_date.to_pydatetime()
            
            for symbol in symbols:
                if count >= sample_size:
                    break
                
                # Check if price went up
                outcome = self._check_outcome(symbol, test_date_obj, target_gain_pct, holding_days)
                
                if outcome and outcome['hit_target']:
                    # Check if criteria was NOT met
                    match = self._check_criteria_for_symbol(symbol, test_date_obj, criteria)
                    
                    if not match:
                        # This is a missed opportunity
                        missed.append({
                            'date': test_date_obj.date().isoformat(),
                            'symbol': symbol,
                            'actual_gain_pct': outcome['actual_gain_pct']
                        })
                        count += 1
        
        self.logger.info(f"Found {len(missed)} missed opportunities (sampled)")
        return missed
    
    def _get_historical_data(
        self,
        symbol: str,
        around_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data with indicators for a symbol
        Uses cache to avoid redundant fetches
        
        Args:
            symbol: Stock symbol
            around_date: Date to center data around
            
        Returns:
            DataFrame with indicators
        """
        cache_key = f"{symbol}"
        
        if cache_key in self.cache:
            self.stats['cached_hits'] += 1
            return self.cache[cache_key]
        
        try:
            # Fetch data
            end_date = around_date + timedelta(days=10)
            start_date = around_date - timedelta(days=self.LOOKBACK_DAYS)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 50:
                self.stats['data_fetch_failed'] += 1
                return None
            
            self.stats['data_fetch_success'] += 1
            
            # Calculate indicators
            indicators_df = self._calculate_all_indicators(df)
            
            # Cache it
            self.cache[cache_key] = indicators_df
            
            return indicators_df
            
        except Exception as e:
            self.stats['data_fetch_failed'] += 1
            return None
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators
        """
        result = pd.DataFrame(index=df.index)
        
        # Basic price data
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        # RSI
        try:
            rsi = RSIIndicator(close=df['Close'], window=14)
            result['rsi'] = rsi.rsi()
        except:
            pass
        
        # MACD
        try:
            macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
        except:
            pass
        
        # Stochastic
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            result['stoch_k'] = stoch.stoch()
            result['stoch_d'] = stoch.stoch_signal()
        except:
            pass
        
        # ADX
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['adx'] = adx.adx()
        except:
            pass
        
        # Bollinger Bands
        try:
            bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
            result['bb_upper'] = bb.bollinger_hband()
            result['bb_lower'] = bb.bollinger_lband()
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / df['Close'] * 100
        except:
            pass
        
        # ATR
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            result['atr'] = atr.average_true_range()
        except:
            pass
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            try:
                result[f'ema{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
                result[f'sma{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
            except:
                pass
        
        # Volume indicators
        try:
            result['volume_sma20'] = result['volume'].rolling(window=20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma20']
        except:
            pass
        
        # Price change
        try:
            result['price_change_1d'] = df['Close'].pct_change(1) * 100
        except:
            pass
        
        return result
    
    def _calculate_summary(
        self,
        results: List[Dict[str, Any]],
        missed_opportunities: List[Dict[str, Any]],
        target_gain_pct: float
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics for backtest
        
        Args:
            results: Detailed results
            missed_opportunities: Missed opportunities
            target_gain_pct: Target gain percentage
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {
                'total_signals': 0,
                'successful_hits': 0,
                'false_positives': 0,
                'missed_opportunities': len(missed_opportunities),
                'success_rate': 0.0,
                'false_positive_rate': 0.0,
                'avg_gain_on_hits': 0.0,
                'avg_loss_on_misses': 0.0
            }
        
        total_signals = len(results)
        successful_hits = sum(1 for r in results if r['hit_target'])
        false_positives = total_signals - successful_hits
        
        # Calculate averages
        gains_on_hits = [r['actual_gain_pct'] for r in results if r['hit_target']]
        losses_on_misses = [r['actual_gain_pct'] for r in results if not r['hit_target']]
        
        avg_gain_on_hits = np.mean(gains_on_hits) if gains_on_hits else 0.0
        avg_loss_on_misses = np.mean(losses_on_misses) if losses_on_misses else 0.0
        
        return {
            'total_signals': total_signals,
            'successful_hits': successful_hits,
            'false_positives': false_positives,
            'missed_opportunities': len(missed_opportunities),
            'success_rate': (successful_hits / total_signals * 100) if total_signals > 0 else 0.0,
            'false_positive_rate': (false_positives / total_signals * 100) if total_signals > 0 else 0.0,
            'avg_gain_on_hits': float(avg_gain_on_hits),
            'avg_loss_on_misses': float(avg_loss_on_misses),
            'total_return': sum(r['actual_gain_pct'] for r in results)
        }
