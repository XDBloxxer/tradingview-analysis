#!/usr/bin/env python3
"""
Strategy Backtester - Core backtesting engine
Evaluates trading strategies against historical data

FIXED VERSION:
1. Gets top N winners for each day
2. Finds stocks matching strategy criteria (configurable limit)
3. Identifies:
   - True positives (criteria match + winner)
   - False positives (criteria match but not winner)
   - Missed opportunities (winner but no criteria match)
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
    1. Finding top N daily winners
    2. Finding stocks matching indicator criteria (with limit)
    3. Checking if matches hit target gains
    4. Tracking true positives, false positives, and missed opportunities
    """
    
    # HARDCODED LIMITS - Change these values to adjust analysis scope
    TOP_WINNERS_PER_DAY = 10  # How many top daily winners to track
    MAX_CRITERIA_MATCHES = 50  # Max stocks to analyze that match criteria
    
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
        self.logger.info(f"Top winners to track: {self.TOP_WINNERS_PER_DAY} (hardcoded)")
        self.logger.info(f"Max criteria matches to analyze: {self.MAX_CRITERIA_MATCHES} (hardcoded)")
        
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
            
            # Step 1: Get top winners for this day
            winners = self._get_top_winners(
                test_date,
                self.TOP_WINNERS_PER_DAY,
                strategy_config.get('exchanges', ['NASDAQ', 'NYSE', 'AMEX']),
                strategy_config.get('min_price', 0.50),
                strategy_config.get('min_volume', 100000)
            )
            
            self.logger.info(f"  Found {len(winners)} top winners")
            
            # Step 2: Get stocks matching criteria (separate from winners)
            criteria_matches = self._get_criteria_matches(
                test_date,
                strategy_config['indicator_criteria'],
                self.MAX_CRITERIA_MATCHES,
                strategy_config.get('exchanges', ['NASDAQ', 'NYSE', 'AMEX']),
                strategy_config.get('min_price', 0.50),
                strategy_config.get('max_price'),
                strategy_config.get('min_volume', 100000)
            )
            
            self.logger.info(f"  Found {len(criteria_matches)} stocks matching criteria")
            
            # Step 3: Evaluate outcomes
            day_trades = self._evaluate_day(
                test_date,
                winners,
                criteria_matches,
                strategy_config['target_min_gain_pct'],
                strategy_config['target_days']
            )
            
            all_trades.extend(day_trades)
            
            # Aggregate daily results
            daily_result = self._aggregate_daily_results(test_date, day_trades, len(winners), len(criteria_matches))
            daily_results.append(daily_result)
            
            self.logger.info(
                f"  Day results: {daily_result['criteria_matches']} criteria matches, "
                f"{daily_result['true_positives']} wins (struck gold), "
                f"{daily_result['false_positives']} false flags, "
                f"{daily_result['missed_opportunities']} missed winners"
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
    
    def _get_top_winners(
        self,
        date: datetime.date,
        count: int,
        exchanges: List[str],
        min_price: float,
        min_volume: int
    ) -> List[Dict[str, Any]]:
        """
        Get top N winners for a specific date
        
        Args:
            date: Date to get winners for
            count: Number of top winners
            exchanges: List of exchanges
            min_price: Minimum stock price
            min_volume: Minimum volume
            
        Returns:
            List of winner dictionaries
        """
        all_stocks = []
        
        for exchange in exchanges:
            try:
                filters = [
                    {'left': 'close', 'operation': 'greater', 'right': min_price},
                    {'left': 'volume', 'operation': 'greater', 'right': min_volume},
                    {'left': 'change_abs', 'operation': 'greater', 'right': 0}  # Only gainers
                ]
                
                market = 'america' if exchange in ['NASDAQ', 'NYSE', 'AMEX'] else exchange.lower()
                
                results = self.screener.screen(
                    market=market,
                    filters=filters,
                    limit=count * 2,  # Get extra to account for filtering
                    sort_by='change_abs',
                    sort_order='desc'
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
                        
                        change_pct = item.get('change', item.get('change_abs', 0))
                        
                        all_stocks.append({
                            'symbol': symbol,
                            'exchange': item_exchange,
                            'price': item.get('close', 0),
                            'volume': item.get('volume', 0),
                            'change_pct': change_pct
                        })
                
            except Exception as e:
                self.logger.debug(f"Error getting winners for {exchange}: {e}")
                continue
        
        # Sort by change_pct and take top N
        all_stocks.sort(key=lambda x: x['change_pct'], reverse=True)
        return all_stocks[:count]
    
    def _get_criteria_matches(
        self,
        date: datetime.date,
        indicator_criteria: List[Dict[str, Any]],
        max_matches: int,
        exchanges: List[str],
        min_price: float,
        max_price: Optional[float],
        min_volume: int
    ) -> List[Dict[str, Any]]:
        """
        Get stocks that match the indicator criteria
        
        Args:
            date: Date to evaluate
            indicator_criteria: List of indicator conditions
            max_matches: Maximum number of matches to return
            exchanges: List of exchanges
            min_price: Minimum stock price
            max_price: Maximum stock price (optional)
            min_volume: Minimum volume
            
        Returns:
            List of stock dictionaries that match criteria
        """
        # Get a broader universe of stocks to evaluate
        universe = []
        
        for exchange in exchanges:
            try:
                filters = [
                    {'left': 'close', 'operation': 'greater', 'right': min_price},
                    {'left': 'volume', 'operation': 'greater', 'right': min_volume}
                ]
                
                if max_price:
                    filters.append({'left': 'close', 'operation': 'less', 'right': max_price})
                
                market = 'america' if exchange in ['NASDAQ', 'NYSE', 'AMEX'] else exchange.lower()
                
                results = self.screener.screen(
                    market=market,
                    filters=filters,
                    limit=max_matches * 3  # Get more than needed for filtering
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
                        
                        universe.append({
                            'symbol': symbol,
                            'exchange': item_exchange,
                            'price': item.get('close', 0),
                            'volume': item.get('volume', 0)
                        })
                
            except Exception as e:
                self.logger.debug(f"Error getting universe for {exchange}: {e}")
                continue
        
        # Now evaluate which stocks match the criteria
        matches = []
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._check_stock_criteria, stock, date, indicator_criteria): stock
                for stock in universe[:max_matches * 2]  # Limit how many we check
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        matches.append(result)
                        if len(matches) >= max_matches:
                            break
                except Exception as e:
                    continue
        
        return matches[:max_matches]
    
    def _check_stock_criteria(
        self,
        stock: Dict[str, Any],
        date: datetime.date,
        indicator_criteria: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a stock matches indicator criteria
        
        Args:
            stock: Stock dictionary
            date: Date to evaluate
            indicator_criteria: Indicator conditions
            
        Returns:
            Stock dict if it matches, None otherwise
        """
        symbol = stock['symbol']
        
        # Fetch historical data
        hist_data = self._fetch_historical_data(symbol, date)
        
        if hist_data is None or hist_data.empty:
            return None
        
        # Get indicator values for date
        if date not in hist_data.index:
            prior_dates = [d for d in hist_data.index if d <= date]
            if not prior_dates:
                return None
            date_actual = prior_dates[-1]
        else:
            date_actual = date
        
        indicators = hist_data.loc[date_actual]
        
        # Check if criteria is met
        if self._check_criteria(indicators, indicator_criteria):
            stock['indicators'] = self._extract_indicator_values(indicators)
            stock['entry_price'] = float(indicators['close'])
            return stock
        
        return None
    
    def _evaluate_day(
        self,
        date: datetime.date,
        winners: List[Dict[str, Any]],
        criteria_matches: List[Dict[str, Any]],
        target_gain_pct: float,
        target_days: int
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the day's results
        
        Args:
            date: Test date
            winners: List of top winners
            criteria_matches: List of stocks matching criteria
            target_gain_pct: Target gain percentage
            target_days: Holding period
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        
        # Create sets for easy lookup
        winner_symbols = {w['symbol'] for w in winners}
        match_symbols = {m['symbol'] for m in criteria_matches}
        
        # Process winners to check outcomes
        winner_outcomes = {}
        for winner in winners:
            hist_data = self._fetch_historical_data(winner['symbol'], date)
            if hist_data is not None:
                entry_price = winner['price']
                exit_price, actual_gain = self._calculate_outcome(hist_data, date, target_days)
                
                hit_target = actual_gain >= target_gain_pct if actual_gain is not None else False
                
                winner_outcomes[winner['symbol']] = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'actual_gain_pct': actual_gain,
                    'hit_target': hit_target
                }
        
        # TRUE POSITIVES: In criteria matches AND is a winner that hit target
        for match in criteria_matches:
            symbol = match['symbol']
            
            if symbol in winner_symbols and symbol in winner_outcomes:
                outcome = winner_outcomes[symbol]
                
                if outcome['hit_target']:
                    trades.append({
                        'symbol': symbol,
                        'exchange': match['exchange'],
                        'signal_date': date.isoformat(),
                        'entry_price': match['entry_price'],
                        'entry_volume': int(match.get('volume', 0)),
                        'indicator_values': match.get('indicators', {}),
                        'matched_criteria': True,
                        'hit_target': True,
                        'actual_gain_pct': outcome['actual_gain_pct'],
                        'exit_price': outcome['exit_price'],
                        'trade_type': 'true_positive'
                    })
            else:
                # FALSE POSITIVE: In criteria but not a winner (or didn't hit target)
                # Get outcome for this stock
                hist_data = self._fetch_historical_data(symbol, date)
                if hist_data is not None:
                    exit_price, actual_gain = self._calculate_outcome(hist_data, date, target_days)
                    
                    trades.append({
                        'symbol': symbol,
                        'exchange': match['exchange'],
                        'signal_date': date.isoformat(),
                        'entry_price': match['entry_price'],
                        'entry_volume': int(match.get('volume', 0)),
                        'indicator_values': match.get('indicators', {}),
                        'matched_criteria': True,
                        'hit_target': False,
                        'actual_gain_pct': actual_gain,
                        'exit_price': exit_price,
                        'trade_type': 'false_positive'
                    })
        
        # MISSED OPPORTUNITIES: Winners that hit target but not in criteria
        for winner in winners:
            symbol = winner['symbol']
            
            if symbol not in match_symbols and symbol in winner_outcomes:
                outcome = winner_outcomes[symbol]
                
                if outcome['hit_target']:
                    trades.append({
                        'symbol': symbol,
                        'exchange': winner['exchange'],
                        'signal_date': date.isoformat(),
                        'entry_price': winner['price'],
                        'entry_volume': int(winner.get('volume', 0)),
                        'indicator_values': {},
                        'matched_criteria': False,
                        'hit_target': True,
                        'actual_gain_pct': outcome['actual_gain_pct'],
                        'exit_price': outcome['exit_price'],
                        'trade_type': 'false_negative'
                    })
        
        return trades
    
    def _get_trading_days(
        self,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> List[datetime.date]:
        """Get list of trading days between start and end date"""
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')
        trading_days = [d.date() for d in all_days]
        holidays = self._get_us_holidays(start_date.year, end_date.year)
        trading_days = [d for d in trading_days if d not in holidays]
        return trading_days
    
    def _get_us_holidays(self, start_year: int, end_year: int) -> set:
        """Get approximate US market holidays"""
        holidays = set()
        for year in range(start_year, end_year + 1):
            holidays.add(datetime(year, 1, 1).date())
            holidays.add(datetime(year, 7, 4).date())
            holidays.add(datetime(year, 12, 25).date())
        return holidays
    
    def _fetch_historical_data(
        self,
        symbol: str,
        test_date: datetime.date
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data and calculate indicators"""
        cache_key = f"{symbol}:{test_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            end_date = test_date + timedelta(days=30)
            start_date = test_date - timedelta(days=self.LOOKBACK_DAYS)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty or len(df) < 50:
                return None
            
            indicators_df = self._calculate_indicators(df)
            indicators_df.index = pd.to_datetime(indicators_df.index).date
            
            self.cache[cache_key] = indicators_df
            return indicators_df
            
        except Exception as e:
            self.logger.debug(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        result = pd.DataFrame(index=df.index)
        
        result['close'] = df['Close']
        result['open'] = df['Open']
        result['high'] = df['High']
        result['low'] = df['Low']
        result['volume'] = df['Volume']
        
        try:
            rsi = RSIIndicator(close=df['Close'], window=14)
            result['rsi'] = rsi.rsi()
        except:
            pass
        
        try:
            macd = MACD(close=df['Close'])
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
        except:
            pass
        
        try:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
            result['stoch_k'] = stoch.stoch()
            result['stoch_d'] = stoch.stoch_signal()
        except:
            pass
        
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
            result['adx'] = adx.adx()
        except:
            pass
        
        try:
            bb = BollingerBands(close=df['Close'])
            result['bb_upper'] = bb.bollinger_hband()
            result['bb_lower'] = bb.bollinger_lband()
            result['bb_middle'] = bb.bollinger_mavg()
        except:
            pass
        
        try:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
            result['atr'] = atr.average_true_range()
        except:
            pass
        
        for period in [10, 20, 50, 200]:
            try:
                result[f'ema_{period}'] = EMAIndicator(close=df['Close'], window=period).ema_indicator()
            except:
                pass
        
        for period in [10, 20, 50, 200]:
            try:
                result[f'sma_{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()
            except:
                pass
        
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
        """Check if indicators meet all criteria"""
        for condition in criteria:
            indicator_name = condition['indicator']
            operator = condition['operator']
            target_value = condition['value']
            
            if indicator_name not in indicators.index:
                return False
            
            actual_value = indicators[indicator_name]
            
            if pd.isna(actual_value):
                return False
            
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
        """Calculate the outcome after holding for specified days"""
        try:
            entry_price = hist_data.loc[entry_date, 'close']
            
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
        trades: List[Dict[str, Any]],
        winners_count: int,
        criteria_matches_count: int
    ) -> Dict[str, Any]:
        """Aggregate results for a single day"""
        if not trades:
            return {
                'test_date': date.isoformat(),
                'total_scanned': winners_count + criteria_matches_count,
                'criteria_matches': criteria_matches_count,
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
            'test_date': date.isoformat(),
            'total_scanned': winners_count + criteria_matches_count,
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
        self.logger.info(f"  True Positives (Struck Gold): {stats['true_positives']}")
        self.logger.info(f"  False Positives (False Flags): {stats['false_positives']}")
        self.logger.info(f"  Missed Opportunities (Missed Winners): {stats['missed_opportunities']}")
        self.logger.info(f"  Accuracy: {stats['accuracy_pct']}%")
        if stats['avg_gain_pct']:
            self.logger.info(f"  Average Gain: {stats['avg_gain_pct']}%")
