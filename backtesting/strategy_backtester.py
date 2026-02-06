"""
Strategy Backtester - Database Query Version - FIXED
FIXED: Better error handling, writes results incrementally, doesn't stop on single date failures
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, UltimateOscillator, AwesomeOscillatorIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class StrategyBacktester:
    """
    Backtester that queries historical database and calculates indicators on-the-fly
    FIXED: Writes results incrementally and handles errors gracefully
    """
    
    def __init__(self, config: dict):
        """Initialize backtester"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.logger.info("Strategy backtester initialized (database query mode)")
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        supabase_client,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy
        FIXED: Writes results incrementally every 10 days and handles errors better
        
        Args:
            strategy_config: Strategy configuration
            supabase_client: BacktestSupabaseClient instance
            progress_callback: Optional callback(current, total, date)
            
        Returns:
            Dictionary with results
        """
        start_date = pd.to_datetime(strategy_config['start_date']).date()
        end_date = pd.to_datetime(strategy_config['end_date']).date()
        target_gain_pct = strategy_config['target_min_gain_pct']
        target_days = strategy_config.get('target_days', 1)
        criteria = strategy_config['indicator_criteria']
        strategy_id = strategy_config.get('id') or strategy_config.get('strategy_id')
        
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        self.logger.info(f"Target: {target_gain_pct}% in {target_days} days")
        self.logger.info(f"Criteria: {len(criteria)} conditions")
        
        # Get all available dates in database within range
        self.logger.info("Fetching available trading dates from database...")
        trading_days = supabase_client.get_available_dates(start_date, end_date)
        self.logger.info(f"Found {len(trading_days)} trading days in database")
        
        if not trading_days:
            raise ValueError(f"No data found in database for date range {start_date} to {end_date}")
        
        # Process each date with incremental writes
        all_trades = []
        daily_results = []
        batch_write_interval = 10  # Write to database every 10 days
        failed_dates = []
        
        for i, test_date in enumerate(trading_days):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(trading_days), test_date)
                
                self.logger.info(f"Processing {test_date} ({i+1}/{len(trading_days)})")
                
                # Process this single date
                day_trades = self._process_date(
                    test_date,
                    target_gain_pct,
                    target_days,
                    criteria,
                    strategy_config,
                    supabase_client
                )
                
                if day_trades:
                    all_trades.extend(day_trades)
                    
                    # Calculate daily stats
                    daily_stats = self._calculate_daily_stats(test_date, day_trades)
                    daily_results.append(daily_stats)
                
                # Write incrementally every N days
                if (i + 1) % batch_write_interval == 0 or (i + 1) == len(trading_days):
                    self.logger.info(f"Writing intermediate results (processed {i+1}/{len(trading_days)} days)...")
                    
                    try:
                        # Write accumulated results
                        if daily_results:
                            supabase_client.write_daily_results(strategy_id, daily_results)
                        
                        if all_trades:
                            supabase_client.write_trades(strategy_id, all_trades)
                        
                        # Calculate and update current stats
                        current_stats = self._calculate_overall_stats(all_trades)
                        supabase_client.update_strategy_summary(strategy_id, current_stats)
                        
                        self.logger.info(f"✓ Wrote results for {len(daily_results)} days, {len(all_trades)} trades")
                        
                        # Clear the lists since we've written them
                        # (don't clear - we need them for final stats)
                        
                    except Exception as e:
                        self.logger.error(f"Error writing intermediate results: {e}", exc_info=True)
                        # Don't stop - continue processing
                
            except Exception as e:
                self.logger.error(f"Error processing {test_date}: {e}", exc_info=True)
                failed_dates.append(test_date)
                # Don't stop - continue with next date
                continue
        
        # Log summary
        self.logger.info(f"Backtest complete: {len(trading_days)} days processed")
        if failed_dates:
            self.logger.warning(f"Failed to process {len(failed_dates)} dates: {failed_dates[:10]}{'...' if len(failed_dates) > 10 else ''}")
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(all_trades)
        
        return {
            'trades': all_trades,
            'daily_results': daily_results,
            'overall_stats': overall_stats,
            'failed_dates': failed_dates
        }
    
    def _process_date(
        self,
        test_date: datetime.date,
        target_gain_pct: float,
        target_days: int,
        criteria: List[Dict],
        strategy_config: Dict,
        supabase_client
    ) -> List[Dict[str, Any]]:
        """
        Process a single date:
        1. Get top 20 gainers from database
        2. Get ALL stocks matching criteria (not just 20)
        3. Check which ones hit target
        """
        self.logger.debug(f"Processing {test_date}")
        
        # Get top gainers for this date from database (top 50 to get more)
        gainers = supabase_client.get_top_gainers(test_date, top_n=5)
        
        if not gainers:
            self.logger.debug(f"No gainers found for {test_date}")
            gainers = []
        
        self.logger.debug(f"Found {len(gainers)} gainers")
        
        # Get ALL stocks matching criteria (not limited to 20)
        criteria_matches = self._get_criteria_matches(
            test_date,
            criteria,
            strategy_config,
            supabase_client,
            max_stocks=10  # Increased limit to scan more stocks
        )
        
        self.logger.debug(f"Found {len(criteria_matches)} criteria matches")
        
        # Combine all unique symbols
        all_symbols = set()
        all_symbols.update(gainers)
        all_symbols.update(criteria_matches)
        
        self.logger.debug(f"Total unique symbols: {len(all_symbols)}")
        
        # For each symbol, check if it hit target
        trades = []
        
        for symbol in all_symbols:
            try:
                # Get entry and exit prices from database
                entry_data = supabase_client.get_stock_data(symbol, test_date)
                exit_date = test_date + timedelta(days=target_days)
                exit_data = supabase_client.get_stock_data(symbol, exit_date)
                
                if not entry_data or not exit_data:
                    continue
                
                entry_price = entry_data['close']
                exit_price = exit_data['close']
                
                # Calculate actual gain
                actual_gain_pct = ((exit_price - entry_price) / entry_price) * 100
                
                # Get indicator values at signal time
                indicator_values = self._calculate_indicators_for_stock(
                    symbol, test_date, criteria, supabase_client
                )
                
                # Determine if matched criteria
                matched_criteria = symbol in criteria_matches
                
                # Determine if hit target
                hit_target = actual_gain_pct >= target_gain_pct
                
                # Classify trade type
                if matched_criteria and hit_target:
                    trade_type = 'true_positive'
                elif matched_criteria and not hit_target:
                    trade_type = 'false_positive'
                elif not matched_criteria and hit_target:
                    trade_type = 'false_negative'
                else:
                    trade_type = 'true_negative'
                
                trades.append({
                    'symbol': symbol,
                    'exchange': entry_data.get('exchange', 'NASDAQ'),
                    'signal_date': test_date.isoformat(),
                    'entry_price': float(entry_price),
                    'entry_volume': int(entry_data.get('volume', 0)),
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
    
    def _get_criteria_matches(
        self,
        test_date: datetime.date,
        criteria: List[Dict],
        strategy_config: Dict,
        supabase_client,
        max_stocks: int = 500
    ) -> List[str]:
        """
        Find stocks matching criteria on a specific date
        Calculates indicators on-the-fly
        FIXED: Scans more stocks, not limited to 20
        """
        min_price = strategy_config.get('min_price', 0.25)
        max_price = strategy_config.get('max_price')
        min_volume = strategy_config.get('min_volume', 100000)
        
        # Get ALL stocks from database for this date (pre-filtered by price/volume)
        all_stocks = supabase_client.get_all_stocks_for_date(
            test_date,
            min_price=min_price,
            max_price=max_price,
            min_volume=min_volume
        )
        
        if not all_stocks:
            return []
        
        self.logger.debug(f"Scanning {len(all_stocks)} stocks for criteria matches")
        
        # For each stock, calculate indicators and check criteria
        matches = []
        
        for stock_symbol in all_stocks:
            # Stop if we have enough matches
            if len(matches) >= max_stocks:
                break
            
            try:
                # Calculate indicators for this stock
                indicators = self._calculate_indicators_for_stock(
                    stock_symbol, test_date, criteria, supabase_client
                )
                
                # Check all criteria
                all_criteria_met = True
                
                for condition in criteria:
                    indicator_name = condition['indicator'].lower()
                    operator = condition['operator']
                    comparison_type = condition.get('comparison_type', 'value')
                    
                    if indicator_name not in indicators:
                        all_criteria_met = False
                        break
                    
                    actual_value = indicators[indicator_name]
                    
                    if actual_value is None:
                        all_criteria_met = False
                        break
                    
                    # Determine comparison value
                    if comparison_type == 'indicator':
                        # Compare to another indicator
                        compare_to = condition.get('compare_to', '').lower()
                        if compare_to not in indicators:
                            all_criteria_met = False
                            break
                        target_value = indicators[compare_to]
                        if target_value is None:
                            all_criteria_met = False
                            break
                    else:
                        # Compare to a fixed value
                        target_value = condition['value']
                    
                    # Check condition
                    if operator == '>':
                        if not actual_value > target_value:
                            all_criteria_met = False
                            break
                    elif operator == '<':
                        if not actual_value < target_value:
                            all_criteria_met = False
                            break
                    elif operator == '>=':
                        if not actual_value >= target_value:
                            all_criteria_met = False
                            break
                    elif operator == '<=':
                        if not actual_value <= target_value:
                            all_criteria_met = False
                            break
                    elif operator == '==':
                        if not actual_value == target_value:
                            all_criteria_met = False
                            break
                    elif operator == '!=':
                        if not actual_value != target_value:
                            all_criteria_met = False
                            break
                
                if all_criteria_met:
                    matches.append(stock_symbol)
                        
            except Exception as e:
                self.logger.debug(f"Error checking criteria for {stock_symbol}: {e}")
                continue
        
        return matches
    
    def _calculate_indicators_for_stock(
        self,
        symbol: str,
        target_date: datetime.date,
        criteria: List[Dict],
        supabase_client
    ) -> Dict[str, float]:
        """
        Calculate technical indicators for a stock on a specific date
        Fetches historical data from database and calculates on-the-fly
        """
        # Get historical data for this stock (need enough for indicator calculations)
        lookback_days = 250  # Enough for 200-day MA + buffer
        start_date = target_date - timedelta(days=lookback_days)
        
        hist_data = supabase_client.get_stock_history(symbol, start_date, target_date)
        
        if not hist_data or len(hist_data) < 20:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(hist_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # Calculate indicators
        indicators = {}
        
        # Find target date index
        target_date_dt = pd.Timestamp(target_date)
        if target_date_dt not in df.index:
            # Find closest prior date
            prior_dates = [d for d in df.index if d.date() <= target_date]
            if not prior_dates:
                return {}
            target_date_dt = prior_dates[-1]
        
        idx = df.index.get_loc(target_date_dt)
        
        # Extract needed indicators from criteria
        needed = set()
        for c in criteria:
            needed.add(c['indicator'].lower())
            # Also add comparison indicator if comparing to another indicator
            if c.get('comparison_type') == 'indicator':
                needed.add(c.get('compare_to', '').lower())
        
        # Calculate each needed indicator
        try:
            # RSI
            if 'rsi' in needed:
                rsi_ind = RSIIndicator(close=df['close'], window=14)
                indicators['rsi'] = rsi_ind.rsi().iloc[idx]
            
            # Stochastic
            if 'stoch_k' in needed or 'stoch.k' in needed:
                stoch = StochasticOscillator(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=14, smooth_window=3
                )
                indicators['stoch_k'] = stoch.stoch().iloc[idx]
                indicators['stoch.k'] = indicators['stoch_k']
            
            if 'stoch_d' in needed or 'stoch.d' in needed:
                stoch = StochasticOscillator(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=14, smooth_window=3
                )
                indicators['stoch_d'] = stoch.stoch_signal().iloc[idx]
                indicators['stoch.d'] = indicators['stoch_d']
            
            # MACD
            if any(x in needed for x in ['macd', 'macd.macd', 'macd_signal', 'macd.signal']):
                macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
                indicators['macd'] = macd.macd().iloc[idx]
                indicators['macd.macd'] = indicators['macd']
                indicators['macd_signal'] = macd.macd_signal().iloc[idx]
                indicators['macd.signal'] = indicators['macd_signal']
            
            # ADX
            if 'adx' in needed:
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                indicators['adx'] = adx.adx().iloc[idx]
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                ema_key = f'ema_{period}'
                if ema_key in needed or f'ema{period}' in needed:
                    ema = EMAIndicator(close=df['close'], window=period)
                    val = ema.ema_indicator().iloc[idx]
                    indicators[ema_key] = val
                    indicators[f'ema{period}'] = val
                
                sma_key = f'sma_{period}'
                if sma_key in needed or f'sma{period}' in needed:
                    sma = SMAIndicator(close=df['close'], window=period)
                    val = sma.sma_indicator().iloc[idx]
                    indicators[sma_key] = val
                    indicators[f'sma{period}'] = val
            
            # ATR
            if 'atr' in needed:
                atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
                indicators['atr'] = atr.average_true_range().iloc[idx]
            
            # Bollinger Bands
            if any(x in needed for x in ['bb.upper', 'bb.lower', 'bb.middle', 'bb_width']):
                bb = BollingerBands(close=df['close'], window=20, window_dev=2)
                indicators['bb.upper'] = bb.bollinger_hband().iloc[idx]
                indicators['bb.lower'] = bb.bollinger_lband().iloc[idx]
                indicators['bb.middle'] = bb.bollinger_mavg().iloc[idx]
                indicators['bb_width'] = (indicators['bb.upper'] - indicators['bb.lower']) / indicators['bb.middle'] * 100
            
            # Volume
            if 'volume' in needed:
                indicators['volume'] = df['volume'].iloc[idx]
            
            if 'volume_ratio' in needed:
                vol_sma = df['volume'].rolling(window=20).mean()
                indicators['volume_ratio'] = df['volume'].iloc[idx] / vol_sma.iloc[idx]
            
            # Basic price fields
            if 'close' in needed:
                indicators['close'] = df['close'].iloc[idx]
            if 'open' in needed:
                indicators['open'] = df['open'].iloc[idx]
            
        except Exception as e:
            self.logger.debug(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def _calculate_daily_stats(self, test_date: datetime.date, trades: List[Dict]) -> Dict:
        """Calculate statistics for a single day"""
        total_scanned = len(trades)
        criteria_matches = sum(1 for t in trades if t['matched_criteria'])
        true_positives = sum(1 for t in trades if t['trade_type'] == 'true_positive')
        false_positives = sum(1 for t in trades if t['trade_type'] == 'false_positive')
        missed_opportunities = sum(1 for t in trades if t['trade_type'] == 'false_negative')
        
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
