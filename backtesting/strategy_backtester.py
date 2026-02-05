"""
Strategy Backtester - Processes each date individually
Finds actual daily gainers and criteria-matching stocks for each trading day
FIXED: Uses TradingView screener to dynamically scan the market on each date
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from tradingview_scraper.symbols.screener import Screener


class StrategyBacktester:
    """
    Backtester that processes each date individually to find:
    1. Top daily gainers (actual market winners)
    2. Stocks matching strategy criteria
    3. Overlap analysis and missed opportunities
    
    Uses TradingView Screener to dynamically scan the market on each date
    """
    
    def __init__(self, config: dict):
        """Initialize backtester"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Processing settings
        self.max_workers = 10
        self.request_delay = 0.5  # Delay between requests
        
        self.logger.info("Strategy backtester initialized (dynamic market scanning)")
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy
        
        Args:
            strategy_config: Strategy configuration
            progress_callback: Optional callback(current, total, date)
            
        Returns:
            Dictionary with results
        """
        start_date = pd.to_datetime(strategy_config['start_date']).date()
        end_date = pd.to_datetime(strategy_config['end_date']).date()
        target_gain_pct = strategy_config['target_min_gain_pct']
        target_days = strategy_config.get('target_days', 1)
        criteria = strategy_config['indicator_criteria']
        
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        self.logger.info(f"Target: {target_gain_pct}% in {target_days} days")
        self.logger.info(f"Criteria: {len(criteria)} conditions")
        
        # Get all trading days
        trading_days = self._get_trading_days(start_date, end_date)
        self.logger.info(f"Found {len(trading_days)} trading days")
        
        # Process each date
        all_trades = []
        daily_results = []
        
        for i, test_date in enumerate(trading_days):
            if progress_callback:
                progress_callback(i + 1, len(trading_days), test_date)
            
            self.logger.info(f"Processing {test_date} ({i+1}/{len(trading_days)})")
            
            # Process this single date
            day_trades = self._process_date(
                test_date,
                target_gain_pct,
                target_days,
                criteria,
                strategy_config
            )
            
            if day_trades:
                all_trades.extend(day_trades)
                
                # Calculate daily stats
                daily_stats = self._calculate_daily_stats(test_date, day_trades)
                daily_results.append(daily_stats)
            
            # Rate limiting
            time.sleep(self.request_delay)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(all_trades)
        
        return {
            'trades': all_trades,
            'daily_results': daily_results,
            'overall_stats': overall_stats
        }
    
    def _get_trading_days(self, start_date, end_date) -> List[datetime.date]:
        """Get list of trading days between dates"""
        # Use SPY as market proxy to get trading days
        spy = yf.Ticker("SPY")
        hist = spy.history(start=start_date, end=end_date + timedelta(days=1), interval='1d')
        
        if hist.empty:
            self.logger.warning("Could not get trading days from SPY, using date range")
            # Fallback: all weekdays
            days = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # Monday=0, Friday=4
                    days.append(current)
                current += timedelta(days=1)
            return days
        
        trading_days = [d.date() for d in hist.index]
        return trading_days
    
    def _process_date(
        self,
        test_date: datetime.date,
        target_gain_pct: float,
        target_days: int,
        criteria: List[Dict],
        strategy_config: Dict
    ) -> List[Dict[str, Any]]:
        """
        Process a single date:
        1. Scan market for top 20 daily gainers
        2. Scan market for 20 stocks matching criteria
        3. Check which ones hit target
        4. Classify all stocks
        """
        self.logger.debug(f"Processing {test_date}")
        
        # Get top gainers for this date (dynamic market scan)
        gainers = self._get_daily_gainers(test_date, top_n=20)
        
        if not gainers:
            self.logger.warning(f"No gainers found for {test_date}")
            return []
        
        self.logger.debug(f"Found {len(gainers)} gainers")
        
        # Get stocks matching criteria (dynamic market scan)
        criteria_matches = self._get_criteria_matches(
            test_date,
            criteria,
            strategy_config,
            top_n=20
        )
        
        self.logger.debug(f"Found {len(criteria_matches)} criteria matches")
        
        # Combine all unique symbols
        all_symbols = set()
        all_symbols.update(gainers)
        all_symbols.update(criteria_matches)
        
        # Check if each stock hit target
        trades = []
        
        for symbol in all_symbols:
            try:
                # Get entry price (close on test_date)
                entry_price = self._get_price_on_date(symbol, test_date, 'close')
                
                if entry_price is None:
                    continue
                
                # Get exit price (close on test_date + target_days)
                exit_date = test_date + timedelta(days=target_days)
                exit_price = self._get_price_on_date(symbol, exit_date, 'close')
                
                if exit_price is None:
                    continue
                
                # Calculate actual gain
                actual_gain_pct = ((exit_price - entry_price) / entry_price) * 100
                
                # Get indicator values at signal time
                indicator_values = self._get_indicators_on_date(symbol, test_date, criteria)
                
                # Determine if matched criteria
                matched_criteria = symbol in criteria_matches
                
                # Determine if hit target (>= target_gain_pct)
                hit_target = actual_gain_pct >= target_gain_pct
                
                # Classify trade type
                if matched_criteria and hit_target:
                    trade_type = 'true_positive'
                elif matched_criteria and not hit_target:
                    trade_type = 'false_positive'
                elif not matched_criteria and hit_target:
                    trade_type = 'false_negative'  # Missed opportunity
                else:
                    trade_type = 'true_negative'
                
                # Get volume
                volume = self._get_price_on_date(symbol, test_date, 'volume')
                
                # Determine exchange
                exchange = self._get_exchange(symbol)
                
                trades.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'signal_date': test_date.isoformat(),
                    'entry_price': float(entry_price),
                    'entry_volume': int(volume) if volume else None,
                    'indicator_values': indicator_values,
                    'matched_criteria': matched_criteria,
                    'hit_target': hit_target,
                    'actual_gain_pct': float(actual_gain_pct),
                    'exit_price': float(exit_price),
                    'trade_type': trade_type
                })
                
            except Exception as e:
                self.logger.debug(f"Error processing {symbol} on {test_date}: {e}")
                continue
        
        return trades
    
    def _get_daily_gainers(self, date: datetime.date, top_n: int = 20) -> List[str]:
        """
        Get top daily gainers for a specific date by:
        1. Screening the entire market using TradingView
        2. Getting historical performance for each stock on that date
        3. Finding the actual top gainers
        """
        self.logger.info(f"Scanning market for top gainers on {date}")
        
        # Use TradingView screener to get all tradable stocks
        screener = Screener()
        
        # Get all stocks with basic filters
        results = screener.screen(
            market='america',
            filters=[
                {'left': 'close', 'operation': 'greater', 'right': 0.5},
                {'left': 'volume', 'operation': 'greater', 'right': 100000}
            ],
            limit=10000,  # Get as many as possible
            sort_by='volume',
            sort_order='desc'
        )
        
        if not results or results.get('status') != 'success':
            self.logger.error("Screener failed")
            return []
        
        symbols = []
        for item in results.get('data', []):
            symbol_full = item.get('symbol', '')
            if ':' in symbol_full:
                _, symbol = symbol_full.split(':', 1)
                symbols.append(symbol)
        
        self.logger.info(f"Screener returned {len(symbols)} symbols")
        
        # Get performance for each symbol on the target date
        gainers = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._get_daily_performance, symbol, date): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result:
                    gainers.append(result)
        
        # Sort by change_pct descending
        gainers.sort(key=lambda x: x['change_pct'], reverse=True)
        
        self.logger.info(f"Found {len(gainers)} gainers, returning top {top_n}")
        
        # Return top N symbols
        return [g['symbol'] for g in gainers[:top_n]]
    
    def _get_daily_performance(self, symbol: str, date: datetime.date) -> Optional[Dict]:
        """Get a stock's performance on a specific date"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get 2 days of data (previous day + target day)
            start = date - timedelta(days=5)  # Extra buffer for weekends
            end = date + timedelta(days=1)
            
            hist = ticker.history(start=start, end=end, interval='1d')
            
            if hist.empty or len(hist) < 2:
                return None
            
            # Find the row for our target date
            hist.index = pd.to_datetime(hist.index).date
            
            if date not in hist.index:
                return None
            
            target_row = hist.loc[date]
            
            # Get previous day
            prior_dates = [d for d in hist.index if d < date]
            if not prior_dates:
                return None
            
            prior_date = prior_dates[-1]
            prior_row = hist.loc[prior_date]
            
            # Calculate change
            close = target_row['Close']
            prior_close = prior_row['Close']
            change_pct = ((close - prior_close) / prior_close) * 100
            
            # Only return if positive gain
            if change_pct <= 0:
                return None
            
            return {
                'symbol': symbol,
                'date': date,
                'change_pct': change_pct,
                'close': close,
                'volume': target_row['Volume']
            }
            
        except Exception as e:
            self.logger.debug(f"Error getting performance for {symbol}: {e}")
            return None
    
    def _get_criteria_matches(
        self,
        date: datetime.date,
        criteria: List[Dict],
        strategy_config: Dict,
        top_n: int = 20
    ) -> List[str]:
        """
        Find stocks matching criteria on a specific date by:
        1. Screening the entire market using TradingView
        2. Checking each stock's indicators on that date
        3. Finding matches
        """
        min_price = strategy_config.get('min_price', 0.50)
        max_price = strategy_config.get('max_price')
        min_volume = strategy_config.get('min_volume', 100000)
        
        self.logger.info(f"Scanning market for criteria matches on {date}")
        
        # Use TradingView screener to get all tradable stocks
        screener = Screener()
        
        # Get all stocks with basic filters
        results = screener.screen(
            market='america',
            filters=[
                {'left': 'close', 'operation': 'greater', 'right': min_price},
                {'left': 'volume', 'operation': 'greater', 'right': min_volume}
            ],
            limit=10000,
            sort_by='volume',
            sort_order='desc'
        )
        
        if not results or results.get('status') != 'success':
            self.logger.error("Screener failed")
            return []
        
        symbols = []
        for item in results.get('data', []):
            symbol_full = item.get('symbol', '')
            if ':' in symbol_full:
                _, symbol = symbol_full.split(':', 1)
                symbols.append(symbol)
        
        self.logger.info(f"Screener returned {len(symbols)} symbols, checking criteria")
        
        # Check criteria for each symbol
        matches = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._check_criteria_match,
                    symbol,
                    date,
                    criteria,
                    min_price,
                    max_price,
                    min_volume
                ): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                if future.result():
                    matches.append(future.result())
        
        self.logger.info(f"Found {len(matches)} criteria matches")
        
        return matches[:top_n]
    
    def _check_criteria_match(
        self,
        symbol: str,
        date: datetime.date,
        criteria: List[Dict],
        min_price: float,
        max_price: Optional[float],
        min_volume: int
    ) -> Optional[str]:
        """Check if a stock matches criteria on a date"""
        try:
            # Get price and volume
            close = self._get_price_on_date(symbol, date, 'close')
            volume = self._get_price_on_date(symbol, date, 'volume')
            
            if close is None or volume is None:
                return None
            
            # Check price/volume filters
            if close < min_price:
                return None
            if max_price and close > max_price:
                return None
            if volume < min_volume:
                return None
            
            # Get indicators
            indicators = self._get_indicators_on_date(symbol, date, criteria)
            
            # Check all criteria
            for condition in criteria:
                indicator_name = condition['indicator'].lower()
                operator = condition['operator']
                target_value = condition['value']
                
                # Get indicator value
                if indicator_name not in indicators:
                    return None
                
                actual_value = indicators[indicator_name]
                
                if actual_value is None:
                    return None
                
                # Check condition
                if operator == '>':
                    if not actual_value > target_value:
                        return None
                elif operator == '<':
                    if not actual_value < target_value:
                        return None
                elif operator == '>=':
                    if not actual_value >= target_value:
                        return None
                elif operator == '<=':
                    if not actual_value <= target_value:
                        return None
                elif operator == '==':
                    if not actual_value == target_value:
                        return None
                elif operator == '!=':
                    if not actual_value != target_value:
                        return None
            
            # All criteria matched
            return symbol
            
        except Exception as e:
            self.logger.debug(f"Error checking criteria for {symbol}: {e}")
            return None
    
    def _get_price_on_date(
        self,
        symbol: str,
        date: datetime.date,
        field: str
    ) -> Optional[float]:
        """Get price/volume for a symbol on a specific date"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get data around the date
            start = date - timedelta(days=5)
            end = date + timedelta(days=1)
            
            hist = ticker.history(start=start, end=end, interval='1d')
            
            if hist.empty:
                return None
            
            # Find the target date
            hist.index = pd.to_datetime(hist.index).date
            
            if date not in hist.index:
                # Try to find closest prior date
                prior_dates = [d for d in hist.index if d <= date]
                if not prior_dates:
                    return None
                date = prior_dates[-1]
            
            row = hist.loc[date]
            
            field_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            yf_field = field_map.get(field.lower())
            if not yf_field:
                return None
            
            return row[yf_field]
            
        except Exception as e:
            self.logger.debug(f"Error getting {field} for {symbol} on {date}: {e}")
            return None
    
    def _get_indicators_on_date(
        self,
        symbol: str,
        date: datetime.date,
        criteria: List[Dict]
    ) -> Dict[str, float]:
        """Calculate technical indicators for a symbol on a date"""
        try:
            # Get historical data (need enough for indicator calculations)
            ticker = yf.Ticker(symbol)
            start = date - timedelta(days=200)  # 200 days for MAs
            end = date + timedelta(days=1)
            
            hist = ticker.history(start=start, end=end, interval='1d')
            
            if hist.empty or len(hist) < 20:
                return {}
            
            # Calculate indicators
            indicators = {}
            
            # Find target date index
            hist.index = pd.to_datetime(hist.index).date
            if date not in hist.index:
                prior_dates = [d for d in hist.index if d <= date]
                if not prior_dates:
                    return {}
                date = prior_dates[-1]
            
            idx = list(hist.index).index(date)
            
            # Extract needed indicators from criteria
            needed = set()
            for c in criteria:
                needed.add(c['indicator'].lower())
            
            # Calculate each needed indicator
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
            from ta.volatility import BollingerBands, AverageTrueRange
            
            # RSI
            if 'rsi' in needed:
                try:
                    rsi_ind = RSIIndicator(close=hist['Close'], window=14)
                    indicators['rsi'] = rsi_ind.rsi().iloc[idx]
                except:
                    pass
            
            # Stochastic
            if 'stoch_k' in needed or 'stoch.k' in needed:
                try:
                    stoch = StochasticOscillator(
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        window=14,
                        smooth_window=3
                    )
                    indicators['stoch_k'] = stoch.stoch().iloc[idx]
                    indicators['stoch.k'] = indicators['stoch_k']
                except:
                    pass
            
            if 'stoch_d' in needed or 'stoch.d' in needed:
                try:
                    stoch = StochasticOscillator(
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        window=14,
                        smooth_window=3
                    )
                    indicators['stoch_d'] = stoch.stoch_signal().iloc[idx]
                    indicators['stoch.d'] = indicators['stoch_d']
                except:
                    pass
            
            # MACD
            if any(x in needed for x in ['macd', 'macd.macd', 'macd_signal', 'macd.signal']):
                try:
                    macd = MACD(close=hist['Close'], window_slow=26, window_fast=12, window_sign=9)
                    indicators['macd'] = macd.macd().iloc[idx]
                    indicators['macd.macd'] = indicators['macd']
                    indicators['macd_signal'] = macd.macd_signal().iloc[idx]
                    indicators['macd.signal'] = indicators['macd_signal']
                except:
                    pass
            
            # ADX
            if 'adx' in needed:
                try:
                    adx = ADXIndicator(
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        window=14
                    )
                    indicators['adx'] = adx.adx().iloc[idx]
                except:
                    pass
            
            # Moving averages
            for period in [10, 20, 50, 200]:
                ema_key = f'ema_{period}'
                if ema_key in needed or f'ema{period}' in needed:
                    try:
                        ema = EMAIndicator(close=hist['Close'], window=period)
                        val = ema.ema_indicator().iloc[idx]
                        indicators[ema_key] = val
                        indicators[f'ema{period}'] = val
                    except:
                        pass
                
                sma_key = f'sma_{period}'
                if sma_key in needed or f'sma{period}' in needed:
                    try:
                        sma = SMAIndicator(close=hist['Close'], window=period)
                        val = sma.sma_indicator().iloc[idx]
                        indicators[sma_key] = val
                        indicators[f'sma{period}'] = val
                    except:
                        pass
            
            # ATR
            if 'atr' in needed:
                try:
                    atr = AverageTrueRange(
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        window=14
                    )
                    indicators['atr'] = atr.average_true_range().iloc[idx]
                except:
                    pass
            
            # Volume
            if 'volume' in needed:
                indicators['volume'] = hist['Volume'].iloc[idx]
            
            if 'volume_ratio' in needed:
                try:
                    vol_sma = hist['Volume'].rolling(window=20).mean()
                    indicators['volume_ratio'] = hist['Volume'].iloc[idx] / vol_sma.iloc[idx]
                except:
                    pass
            
            # Basic price fields
            if 'close' in needed:
                indicators['close'] = hist['Close'].iloc[idx]
            if 'open' in needed:
                indicators['open'] = hist['Open'].iloc[idx]
            
            return indicators
            
        except Exception as e:
            self.logger.debug(f"Error calculating indicators for {symbol}: {e}")
            return {}
    
    def _get_exchange(self, symbol: str) -> str:
        """Determine exchange for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            exchange = info.get('exchange', 'NASDAQ')
            
            # Map exchange codes
            exchange_map = {
                'NMS': 'NASDAQ',
                'NGM': 'NASDAQ',
                'NCM': 'NASDAQ',
                'NYQ': 'NYSE',
                'NYS': 'NYSE',
                'ASE': 'AMEX',
                'PCX': 'AMEX'
            }
            
            return exchange_map.get(exchange, 'NASDAQ')
        except:
            return 'NASDAQ'
    
    def _calculate_daily_stats(self, test_date: datetime.date, trades: List[Dict]) -> Dict:
        """Calculate statistics for a single day"""
        total_scanned = len(trades)
        criteria_matches = sum(1 for t in trades if t['matched_criteria'])
        true_positives = sum(1 for t in trades if t['trade_type'] == 'true_positive')
        false_positives = sum(1 for t in trades if t['trade_type'] == 'false_positive')
        missed_opportunities = sum(1 for t in trades if t['trade_type'] == 'false_negative')
        
        # Calculate average gains
        matched_trades = [t for t in trades if t['matched_criteria']]
        avg_match_gain = np.mean([t['actual_gain_pct'] for t in matched_trades]) if matched_trades else None
        
        missed_trades = [t for t in trades if t['trade_type'] == 'false_negative']
        avg_miss_gain = np.mean([t['actual_gain_pct'] for t in missed_trades]) if missed_trades else None
        
        all_gains = [t['actual_gain_pct'] for t in trades]
        max_gain = max(all_gains) if all_gains else None
        min_gain = min(all_gains) if all_gains else None
        
        return {
            'test_date': test_date.isoformat(),
            'total_scanned': total_scanned,
            'criteria_matches': criteria_matches,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'missed_opportunities': missed_opportunities,
            'avg_match_gain_pct': avg_match_gain,
            'avg_miss_gain_pct': avg_miss_gain,
            'max_gain_pct': max_gain,
            'min_gain_pct': min_gain
        }
    
    def _calculate_overall_stats(self, all_trades: List[Dict]) -> Dict:
        """Calculate overall backtest statistics"""
        if not all_trades:
            return {
                'total_trades': 0,
                'total_matches': 0,
                'true_positives': 0,
                'false_positives': 0,
                'missed_opportunities': 0,
                'accuracy_pct': 0,
                'avg_gain_pct': None
            }
        
        total_matches = sum(1 for t in all_trades if t['matched_criteria'])
        true_positives = sum(1 for t in all_trades if t['trade_type'] == 'true_positive')
        false_positives = sum(1 for t in all_trades if t['trade_type'] == 'false_positive')
        missed_opportunities = sum(1 for t in all_trades if t['trade_type'] == 'false_negative')
        
        accuracy = (true_positives / total_matches * 100) if total_matches > 0 else 0
        
        matched_trades = [t for t in all_trades if t['matched_criteria']]
        avg_gain = np.mean([t['actual_gain_pct'] for t in matched_trades]) if matched_trades else None
        
        return {
            'total_trades': len(all_trades),
            'total_matches': total_matches,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'missed_opportunities': missed_opportunities,
            'accuracy_pct': round(accuracy, 2),
            'avg_gain_pct': round(avg_gain, 2) if avg_gain else None
        }
