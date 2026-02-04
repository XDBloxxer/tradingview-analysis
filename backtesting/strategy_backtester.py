#!/usr/bin/env python3
"""
Strategy Backtester - Core backtesting engine
Evaluates trading strategies against historical data
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yfinance as yf

# Technical analysis
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from tradingview_scraper.symbols.screener import Screener


class StrategyBacktester:
    """
    Backtests trading strategies by:
    1. Finding stocks matching indicator criteria
    2. Checking if they hit target gains
    3. Tracking true positives, false positives, and missed opportunities
    """
    
    # Parallel processing
    MAX_WORKERS = 5
    
    # Historical data lookback
    LOOKBACK_DAYS = 120
    
    def __init__(self, config: dict):
        """
        Initialize backtester
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize screener for getting stock universe
        self.screener = Screener()
        
        # Cache for historical data
        self.cache = {}
        
        # Statistics
        self.stats = {
            'total_scanned': 0,
            'data_fetched': 0,
            'data_failed': 0,
            'criteria_evaluated': 0
        }
        
        self.logger.info("Strategy Backtester initialized")
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run a complete backtest for a strategy
        
        Args:
            strategy_config: Strategy configuration dictionary with:
                - start_date: Start date for backtest
                - end_date: End date for backtest
                - indicator_criteria: List of indicator conditions
                - target_min_gain_pct: Minimum gain to consider success
                - target_days: Days to hold (1 = same day, 2 = next day, etc.)
                - min_price, max_price, min_volume: Stock filters
                - exchanges: List of exchanges to scan
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING STRATEGY BACKTEST")
        self.logger.info("=" * 60)
        
        # Parse dates
        start_date = pd.to_datetime(strategy_config['start_date']).date()
        end_date = pd.to_datetime(strategy_config['end_date']).date()
        
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Target gain: {strategy_config['target_min_gain_pct']}% in {strategy_config['target_days']} day(s)")
        self.logger.info(f"Indicator criteria: {len(strategy_config['indicator_criteria'])} conditions")
        
        # Generate list of trading days
        trading_days = self._get_trading_days(start_date, end_date)
        self.logger.info(f"Testing {len(trading_days)} trading days")
        
        # Results storage
        all_trades = []
        daily_results = []
        
        # Process each day
        for idx, test_date in enumerate(trading_days):
            if progress_callback:
                progress_callback(idx + 1, len(trading_days), test_date)
            
            self.logger.info(f"Processing {test_date} ({idx + 1}/{len(trading_days)})...")
            
            # Get stock universe for this date
            stocks = self._get_stock_universe(
                test_date,
                strategy_config.get('exchanges', ['NASDAQ', 'NYSE', 'AMEX']),
                strategy_config.get('min_price', 0.50),
                strategy_config.get('max_price'),
                strategy_config.get('min_volume', 100000)
            )
            
            if not stocks:
                self.logger.warning(f"No stocks found for {test_date}")
                continue
            
            self.logger.info(f"  Found {len(stocks)} stocks to evaluate")
            
            # Evaluate each stock
            day_trades = self._evaluate_stocks(
                stocks,
                test_date,
                strategy_config['indicator_criteria'],
                strategy_config['target_min_gain_pct'],
                strategy_config['target_days']
            )
            
            all_trades.extend(day_trades)
            
            # Aggregate daily results
            daily_result = self._aggregate_daily_results(test_date, day_trades)
            daily_results.append(daily_result)
            
            self.logger.info(
                f"  Day results: {daily_result['criteria_matches']} matches, "
                f"{daily_result['true_positives']} wins, "
                f"{daily_result['false_positives']} losses, "
                f"{daily_result['missed_opportunities']} missed"
            )
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(all_trades, daily_results)
        
        results = {
            'trades': all_trades,
            'daily_results': daily_results,
            'overall_stats': overall_stats,
            'stats': self.stats
        }
        
        self.logger.info("=" * 60)
        self.logger.info("BACKTEST COMPLETED")
        self.logger.info("=" * 60)
        self._log_summary(overall_stats)
        
        return results
    
    def _get_trading_days(
        self,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> List[datetime.date]:
        """
        Get list of trading days between start and end date
        Uses market calendar to exclude weekends and holidays
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        # Simple approach: get all weekdays, filter out major holidays
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Convert to dates
        trading_days = [d.date() for d in all_days]
        
        # Filter out major US holidays (simplified)
        # In production, use a proper market calendar library
        holidays = self._get_us_holidays(start_date.year, end_date.year)
        trading_days = [d for d in trading_days if d not in holidays]
        
        return trading_days
    
    def _get_us_holidays(self, start_year: int, end_year: int) -> set:
        """Get approximate US market holidays"""
        # Simplified - just major fixed holidays
        # In production, use pandas_market_calendars or similar
        holidays = set()
        
        for year in range(start_year, end_year + 1):
            # New Year's Day
            holidays.add(datetime(year, 1, 1).date())
            # Independence Day
            holidays.add(datetime(year, 7, 4).date())
            # Christmas
            holidays.add(datetime(year, 12, 25).date())
        
        return holidays
    
    def _get_stock_universe(
        self,
        date: datetime.date,
        exchanges: List[str],
        min_price: float,
        max_price: Optional[float],
        min_volume: int
    ) -> List[Dict[str, Any]]:
        """
        Get universe of stocks to test for a given date
        
        Args:
            date: Date to get stocks for
            exchanges: List of exchanges
            min_price: Minimum stock price
            max_price: Maximum stock price (optional)
            min_volume: Minimum volume
            
        Returns:
            List of stock dictionaries with symbol, exchange, price, volume
        """
        all_stocks = []
        
        for exchange in exchanges:
            try:
                # Get stocks from screener
                filters = [
                    {'left': 'close', 'operation': 'greater', 'right': min_price},
                    {'left': 'volume', 'operation': 'greater', 'right': min_volume}
                ]
                
                if max_price:
                    filters.append({'left': 'close', 'operation': 'less', 'right': max_price})
                
                # Map exchange to market
                market = 'america' if exchange in ['NASDAQ', 'NYSE', 'AMEX'] else exchange.lower()
                
                results = self.screener.screen(
                    market=market,
                    filters=filters,
                    limit=1000
                )
                
                if results and results.get('status') == 'success':
                    data = results.get('data', [])
                    
                    for item in data:
                        symbol_full = item.get('symbol', '')
                        if ':' in symbol_full:
                            item_exchange, symbol = symbol_full.split(':', 1)
                        else:
                            symbol = symbol_full
                            item_exchange = exchange
                        
                        all_stocks.append({
                            'symbol': symbol,
                            'exchange': item_exchange,
                            'price': item.get('close', 0),
                            'volume': item.get('volume', 0)
                        })
                
            except Exception as e:
                self.logger.debug(f"Error getting stocks for {exchange}: {e}")
                continue
        
        self.stats['total_scanned'] += len(all_stocks)
        return all_stocks
    
    def _evaluate_stocks(
        self,
        stocks: List[Dict[str, Any]],
        test_date: datetime.date,
        indicator_criteria: List[Dict[str, Any]],
        target_gain_pct: float,
        target_days: int
    ) -> List[Dict[str, Any]]:
        """
        Evaluate stocks against criteria and check outcomes
        
        Args:
            stocks: List of stock dictionaries
            test_date: Date to evaluate
            indicator_criteria: List of indicator conditions
            target_gain_pct: Target gain percentage
            target_days: Days to hold
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single_stock,
                    stock,
                    test_date,
                    indicator_criteria,
                    target_gain_pct,
                    target_days
                ): stock
                for stock in stocks
            }
            
            for future in as_completed(futures):
                try:
                    trade = future.result()
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    stock = futures[future]
                    self.logger.debug(f"Error evaluating {stock['symbol']}: {e}")
        
        return trades
    
    def _evaluate_single_stock(
        self,
        stock: Dict[str, Any],
        test_date: datetime.date,
        indicator_criteria: List[Dict[str, Any]],
        target_gain_pct: float,
        target_days: int
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single stock
        
        Args:
            stock: Stock dictionary
            test_date: Test date
            indicator_criteria: Indicator conditions
            target_gain_pct: Target gain
            target_days: Holding period
            
        Returns:
            Trade dictionary or None
        """
        symbol = stock['symbol']
        
        # Fetch historical data
        hist_data = self._fetch_historical_data(symbol, test_date)
        
        if hist_data is None or hist_data.empty:
            self.stats['data_failed'] += 1
            return None
        
        self.stats['data_fetched'] += 1
        
        # Get indicator values for test date
        if test_date not in hist_data.index:
            # Find closest prior date
            prior_dates = [d for d in hist_data.index if d <= test_date]
            if not prior_dates:
                return None
            test_date_actual = prior_dates[-1]
        else:
            test_date_actual = test_date
        
        indicators = hist_data.loc[test_date_actual]
        
        # Check if criteria is met
        matched_criteria = self._check_criteria(indicators, indicator_criteria)
        
        # Calculate actual outcome
        entry_price = indicators['close']
        exit_price, actual_gain = self._calculate_outcome(
            hist_data,
            test_date_actual,
            target_days
        )
        
        # Determine if target was hit
        hit_target = actual_gain >= target_gain_pct if actual_gain is not None else False
        
        # Classify trade
        if matched_criteria and hit_target:
            trade_type = 'true_positive'
        elif matched_criteria and not hit_target:
            trade_type = 'false_positive'
        elif not matched_criteria and hit_target:
            trade_type = 'false_negative'  # Missed opportunity
        else:
            trade_type = 'true_negative'
        
        # Build trade record
        trade = {
            'symbol': symbol,
            'exchange': stock['exchange'],
            'signal_date': test_date,
            'entry_price': float(entry_price),
            'entry_volume': int(stock.get('volume', 0)),
            'indicator_values': self._extract_indicator_values(indicators),
            'matched_criteria': matched_criteria,
            'hit_target': hit_target,
            'actual_gain_pct': float(actual_gain) if actual_gain is not None else None,
            'exit_price': float(exit_price) if exit_price is not None else None,
            'trade_type': trade_type
        }
        
        self.stats['criteria_evaluated'] += 1
        
        return trade
    
    def _fetch_historical_data(
        self,
        symbol: str,
        test_date: datetime.date
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data and calculate indicators
        
        Args:
            symbol: Stock symbol
            test_date: Test date
            
        Returns:
            DataFrame with indicators
        """
        # Check cache
        cache_key = f"{symbol}:{test_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Fetch data
            end_date = test_date + timedelta(days=30)  # Extra days for outcome calculation
            start_date = test_date - timedelta(days=self.LOOKBACK_DAYS)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 50:
                return None
            
            # Calculate indicators
            indicators_df = self._calculate_indicators(df)
            
            # Convert index to date
            indicators_df.index = pd.to_datetime(indicators_df.index).date
            
            # Cache it
            self.cache[cache_key] = indicators_df
            
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        result = pd.DataFrame(index=df.index)
        
        # Price data
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
            macd = MACD(close=df['Close'])
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
        except:
            pass
        
        # Stochastic
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
            result['stoch_k'] = stoch.stoch()
            result['stoch_d'] = stoch.stoch_signal()
        except:
            pass
        
        # ADX
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
            result['adx'] = adx.adx()
        except:
            pass
        
        # Bollinger Bands
        try:
            bb = BollingerBands(close=df['Close'])
            result['bb_upper'] = bb.bollinger_hband()
            result['bb_lower'] = bb.bollinger_lband()
            result['bb_middle'] = bb.bollinger_mavg()
        except:
            pass
        
        # ATR
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
            result['atr'] = atr.average_true_range()
        except:
            pass
        
        # EMAs
        for period in [10, 20, 50, 200]:
            try:
                result[f'ema_{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            except:
                pass
        
        # SMAs
        for period in [10, 20, 50, 200]:
            try:
                result[f'sma_{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
            except:
                pass
        
        # Volume SMA
        try:
            result['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            result['volume_ratio'] = df['Volume'] / result['volume_sma_20']
        except:
            pass
        
        return result
    
    def _check_criteria(
        self,
        indicators: pd.Series,
        criteria: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if indicators meet all criteria
        
        Args:
            indicators: Series of indicator values
            criteria: List of condition dicts with indicator, operator, value
            
        Returns:
            True if all criteria met
        """
        for condition in criteria:
            indicator_name = condition['indicator']
            operator = condition['operator']
            target_value = condition['value']
            
            # Get indicator value
            if indicator_name not in indicators.index:
                return False
            
            actual_value = indicators[indicator_name]
            
            # Check if value is valid
            if pd.isna(actual_value):
                return False
            
            # Evaluate condition
            if operator == '>':
                if not actual_value > target_value:
                    return False
            elif operator == '<':
                if not actual_value < target_value:
                    return False
            elif operator == '>=':
                if not actual_value >= target_value:
                    return False
            elif operator == '<=':
                if not actual_value <= target_value:
                    return False
            elif operator == '==':
                if not actual_value == target_value:
                    return False
            elif operator == '!=':
                if not actual_value != target_value:
                    return False
        
        return True
    
    def _calculate_outcome(
        self,
        hist_data: pd.DataFrame,
        entry_date: datetime.date,
        hold_days: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate the outcome after holding for specified days
        
        Args:
            hist_data: Historical data
            entry_date: Entry date
            hold_days: Days to hold
            
        Returns:
            Tuple of (exit_price, gain_percentage)
        """
        try:
            entry_price = hist_data.loc[entry_date, 'close']
            
            # Find exit date
            future_dates = [d for d in hist_data.index if d > entry_date]
            
            if not future_dates or len(future_dates) < hold_days:
                return None, None
            
            exit_date = future_dates[hold_days - 1]
            exit_price = hist_data.loc[exit_date, 'close']
            
            gain_pct = ((exit_price - entry_price) / entry_price) * 100
            
            return exit_price, gain_pct
            
        except Exception as e:
            self.logger.debug(f"Error calculating outcome: {e}")
            return None, None
    
    def _extract_indicator_values(self, indicators: pd.Series) -> Dict[str, Any]:
        """Extract indicator values to dict, handling NaN"""
        values = {}
        for key, value in indicators.items():
            if pd.notna(value):
                try:
                    values[key] = float(value)
                except (ValueError, TypeError):
                    values[key] = None
            else:
                values[key] = None
        return values
    
    def _aggregate_daily_results(
        self,
        date: datetime.date,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results for a single day"""
        if not trades:
            return {
                'test_date': date,
                'total_scanned': 0,
                'criteria_matches': 0,
                'true_positives': 0,
                'false_positives': 0,
                'missed_opportunities': 0,
                'avg_match_gain_pct': None,
                'avg_miss_gain_pct': None,
                'max_gain_pct': None,
                'min_gain_pct': None
            }
        
        matches = [t for t in trades if t['matched_criteria']]
        misses = [t for t in trades if not t['matched_criteria'] and t['hit_target']]
        
        match_gains = [t['actual_gain_pct'] for t in matches if t['actual_gain_pct'] is not None]
        miss_gains = [t['actual_gain_pct'] for t in misses if t['actual_gain_pct'] is not None]
        all_gains = [t['actual_gain_pct'] for t in trades if t['actual_gain_pct'] is not None]
        
        return {
            'test_date': date,
            'total_scanned': len(trades),
            'criteria_matches': len(matches),
            'true_positives': len([t for t in trades if t['trade_type'] == 'true_positive']),
            'false_positives': len([t for t in trades if t['trade_type'] == 'false_positive']),
            'missed_opportunities': len([t for t in trades if t['trade_type'] == 'false_negative']),
            'avg_match_gain_pct': np.mean(match_gains) if match_gains else None,
            'avg_miss_gain_pct': np.mean(miss_gains) if miss_gains else None,
            'max_gain_pct': max(all_gains) if all_gains else None,
            'min_gain_pct': min(all_gains) if all_gains else None
        }
    
    def _calculate_overall_stats(
        self,
        trades: List[Dict[str, Any]],
        daily_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall backtest statistics"""
        if not trades:
            return {
                'total_trades': 0,
                'total_matches': 0,
                'true_positives': 0,
                'false_positives': 0,
                'missed_opportunities': 0,
                'accuracy_pct': 0,
                'avg_gain_pct': None,
                'max_gain_pct': None,
                'min_gain_pct': None
            }
        
        matches = [t for t in trades if t['matched_criteria']]
        true_pos = [t for t in trades if t['trade_type'] == 'true_positive']
        false_pos = [t for t in trades if t['trade_type'] == 'false_positive']
        missed = [t for t in trades if t['trade_type'] == 'false_negative']
        
        match_gains = [t['actual_gain_pct'] for t in matches if t['actual_gain_pct'] is not None]
        all_gains = [t['actual_gain_pct'] for t in trades if t['actual_gain_pct'] is not None]
        
        accuracy = (len(true_pos) / len(matches) * 100) if matches else 0
        
        return {
            'total_trades': len(trades),
            'total_matches': len(matches),
            'true_positives': len(true_pos),
            'false_positives': len(false_pos),
            'missed_opportunities': len(missed),
            'accuracy_pct': round(accuracy, 2),
            'avg_gain_pct': round(np.mean(match_gains), 2) if match_gains else None,
            'max_gain_pct': round(max(all_gains), 2) if all_gains else None,
            'min_gain_pct': round(min(all_gains), 2) if all_gains else None
        }
    
    def _log_summary(self, stats: Dict[str, Any]):
        """Log summary statistics"""
        self.logger.info("Overall Results:")
        self.logger.info(f"  Total Trades: {stats['total_trades']}")
        self.logger.info(f"  Criteria Matches: {stats['total_matches']}")
        self.logger.info(f"  True Positives (Wins): {stats['true_positives']}")
        self.logger.info(f"  False Positives (Losses): {stats['false_positives']}")
        self.logger.info(f"  Missed Opportunities: {stats['missed_opportunities']}")
        self.logger.info(f"  Accuracy: {stats['accuracy_pct']}%")
        if stats['avg_gain_pct']:
            self.logger.info(f"  Average Gain: {stats['avg_gain_pct']}%")
