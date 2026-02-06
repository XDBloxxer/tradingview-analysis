"""
Strategy Backtester - FIXED VERSION
Now checks criteria on T-1 (before move) and measures results on T (after move)
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict

# Technical analysis library
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class StrategyBacktester:
    """
    Backtester that checks criteria BEFORE moves happen
    FIXED: Checks indicators on signal_date (T-1), measures gain on test_date (T)
    """
    
    def __init__(self, config: dict):
        """Initialize backtester"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Caches to reduce queries
        self._stock_universe_cache = {}  # date -> list of symbols
        self._price_cache = {}  # (symbol, date) -> price data
        self._history_cache = {}  # symbol -> DataFrame
        self._indicator_cache = {}  # (symbol, date) -> indicators dict
        
        self.logger.info("Strategy backtester initialized (FIXED VERSION)")
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        supabase_client,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy
        FIXED: Checks criteria on T-1, measures gains on T
        """
        start_date = pd.to_datetime(strategy_config['start_date']).date()
        end_date = pd.to_datetime(strategy_config['end_date']).date()
        target_gain_pct = strategy_config['target_min_gain_pct']
        target_days = strategy_config.get('target_days', 1)
        criteria = strategy_config['indicator_criteria']
        strategy_id = strategy_config.get('id') or strategy_config.get('strategy_id')
        
        self.logger.info(f"Backtest: {start_date} to {end_date}, target: {target_gain_pct}% in {target_days}d")
        self.logger.info(f"LOGIC: Check criteria on T-1, measure gain on T")
        
        # Get trading days
        trading_days = supabase_client.get_available_dates(start_date, end_date)
        self.logger.info(f"Trading days: {len(trading_days)}")
        
        if not trading_days:
            raise ValueError(f"No data for {start_date} to {end_date}")
        
        # Process dates with larger batch writes
        all_trades = []
        daily_results = []
        batch_write_interval = 20
        failed_dates = []
        
        trades_written_count = 0
        daily_written_count = 0
        
        for i, test_date in enumerate(trading_days):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(trading_days), test_date)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(trading_days):
                    self.logger.info(f"Progress: {i+1}/{len(trading_days)}")
                
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
                    daily_stats = self._calculate_daily_stats(test_date, day_trades)
                    daily_results.append(daily_stats)
                
                if (i + 1) % batch_write_interval == 0 or (i + 1) == len(trading_days):
                    new_daily = daily_results[daily_written_count:]
                    new_trades = all_trades[trades_written_count:]
                    
                    self.logger.info(f"Writing batch: {len(new_daily)} daily results, {len(new_trades)} trades")
                    
                    if new_daily:
                        supabase_client.write_daily_results(strategy_id, new_daily)
                        daily_written_count = len(daily_results)
                    
                    if new_trades:
                        supabase_client.write_trades(strategy_id, new_trades)
                        trades_written_count = len(all_trades)
                    
                    current_stats = self._calculate_overall_stats(all_trades)
                    self.logger.info(f"Stats: matches={current_stats['total_matches']}, accuracy={current_stats['accuracy_pct']}%")
                    supabase_client.update_strategy_summary(strategy_id, current_stats)
                
            except Exception as e:
                self.logger.error(f"Error on {test_date}: {e}")
                failed_dates.append(test_date)
                continue
        
        self.logger.info(f"Complete: {len(trading_days)} days, {len(failed_dates)} failed")
        
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
        Process a single date
        FIXED: test_date is when stocks MOVED, signal_date is when we CHECK criteria
        """
        
        # KEY FIX: Calculate signal_date (when we check criteria)
        # For target_days=1, signal_date = test_date - 1 day
        signal_date = test_date - timedelta(days=target_days)
        
        self.logger.debug(f"Test date: {test_date}, Signal date: {signal_date}")
        
        # Get top gainers on TEST_DATE (the explosion day)
        if test_date not in self._stock_universe_cache:
            gainers = supabase_client.get_top_gainers(test_date, top_n=5)
            self._stock_universe_cache[test_date] = gainers
        else:
            gainers = self._stock_universe_cache[test_date]
        
        # Get stocks that met criteria on SIGNAL_DATE (before the move)
        criteria_matches = self._get_criteria_matches(
            signal_date,  # ✅ Check criteria BEFORE the move
            criteria,
            strategy_config,
            supabase_client,
            max_stocks=10
        )
        
        # Combine both lists
        all_symbols = set(gainers + criteria_matches)
        
        # Process trades
        trades = []
        
        for symbol in all_symbols:
            try:
                # Entry is on SIGNAL_DATE (when criteria was checked)
                entry_data = self._get_cached_price(symbol, signal_date, supabase_client)
                
                # Exit is on TEST_DATE (target_days later)
                exit_data = self._get_cached_price(symbol, test_date, supabase_client)
                
                if not entry_data or not exit_data:
                    continue
                
                entry_price = entry_data['close']
                exit_price = exit_data['close']
                actual_gain_pct = ((exit_price - entry_price) / entry_price) * 100
                
                # Track high/low for exit analysis
                exit_high = exit_data.get('high', exit_price)
                exit_low = exit_data.get('low', exit_price)
                
                # Calculate max possible gain (if sold at high)
                max_possible_gain_pct = ((exit_high - entry_price) / entry_price) * 100
                
                # Calculate max drawdown (worst point)
                max_drawdown_pct = ((exit_low - entry_price) / entry_price) * 100
                
                # Did the target get hit intraday?
                target_hit_intraday = max_possible_gain_pct >= target_gain_pct
                
                # Calculate indicators on SIGNAL_DATE (before the move)
                indicator_values = self._calculate_indicators_for_stock(
                    symbol, signal_date, criteria, supabase_client
                )
                
                # Did this stock match criteria on signal_date?
                matched_criteria = symbol in criteria_matches
                
                # Did it hit target by test_date?
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
                    'signal_date': signal_date.isoformat(),  # When we found it
                    'entry_price': float(entry_price),
                    'entry_volume': int(entry_data.get('volume', 0)),
                    'indicator_values': indicator_values,
                    'matched_criteria': matched_criteria,
                    'hit_target': hit_target,
                    'actual_gain_pct': float(actual_gain_pct),
                    'exit_price': float(exit_price),
                    'exit_date': test_date.isoformat(),  # When we measured results
                    'trade_type': trade_type,
                    # Exit analysis fields
                    'exit_high': float(exit_high),
                    'exit_low': float(exit_low),
                    'max_possible_gain_pct': float(max_possible_gain_pct),
                    'max_drawdown_pct': float(max_drawdown_pct),
                    'target_hit_intraday': target_hit_intraday
                })
                
            except Exception as e:
                self.logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        return trades
    
    def _get_cached_price(self, symbol: str, target_date: datetime.date, supabase_client):
        """Get price data with caching"""
        cache_key = (symbol, target_date)
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        data = supabase_client.get_stock_data(symbol, target_date)
        self._price_cache[cache_key] = data
        return data
    
    def _get_criteria_matches(
        self,
        signal_date: datetime.date,
        criteria: List[Dict],
        strategy_config: Dict,
        supabase_client,
        max_stocks: int = 10
    ) -> List[str]:
        """
        Find stocks matching criteria on signal_date
        Samples from filtered universe to minimize egress
        """
        min_price = strategy_config.get('min_price', 0.25)
        max_price = strategy_config.get('max_price')
        min_volume = strategy_config.get('min_volume', 100000)
        
        client = supabase_client.client
        
        query = client.table("historical_market_data") \
            .select("symbol") \
            .eq("date", signal_date.isoformat()) \
            .gte("close", min_price) \
            .gte("volume", min_volume)
        
        if max_price:
            query = query.lte("close", max_price)
        
        # Sample random stocks - small sample to minimize egress
        response = query.limit(50).execute()
        
        if not response.data:
            return []
        
        candidate_stocks = [row['symbol'] for row in response.data]
        
        # Check criteria on this sample
        matches = []
        
        for stock_symbol in candidate_stocks:
            if len(matches) >= max_stocks:
                break
            
            try:
                indicators = self._calculate_indicators_for_stock(
                    stock_symbol, signal_date, criteria, supabase_client
                )
                
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
                    
                    if comparison_type == 'indicator':
                        compare_to = condition.get('compare_to', '').lower()
                        if compare_to not in indicators:
                            all_criteria_met = False
                            break
                        target_value = indicators[compare_to]
                        if target_value is None:
                            all_criteria_met = False
                            break
                    else:
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
        """Calculate indicators with caching"""
        
        # Check cache first
        cache_key = (symbol, target_date)
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        # Get historical data
        if symbol in self._history_cache:
            df = self._history_cache[symbol]
        else:
            lookback_days = 250
            start_date = target_date - timedelta(days=lookback_days)
            hist_data = supabase_client.get_stock_history(symbol, start_date, target_date)
            
            if not hist_data or len(hist_data) < 20:
                return {}
            
            df = pd.DataFrame(hist_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            
            # Cache history
            self._history_cache[symbol] = df
        
        # Find target date
        target_date_dt = pd.Timestamp(target_date)
        if target_date_dt not in df.index:
            prior_dates = [d for d in df.index if d.date() <= target_date]
            if not prior_dates:
                return {}
            target_date_dt = prior_dates[-1]
        
        idx = df.index.get_loc(target_date_dt)
        
        # Extract needed indicators
        needed = set()
        for c in criteria:
            needed.add(c['indicator'].lower())
            if c.get('comparison_type') == 'indicator':
                needed.add(c.get('compare_to', '').lower())
        
        indicators = {}
        
        try:
            # Calculate only needed indicators
            if 'rsi' in needed:
                rsi_ind = RSIIndicator(close=df['close'], window=14)
                indicators['rsi'] = rsi_ind.rsi().iloc[idx]
            
            if 'stoch_k' in needed or 'stoch.k' in needed:
                stoch = StochasticOscillator(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=14, smooth_window=3
                )
                val = stoch.stoch().iloc[idx]
                indicators['stoch_k'] = val
                indicators['stoch.k'] = val
            
            if 'stoch_d' in needed or 'stoch.d' in needed:
                stoch = StochasticOscillator(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=14, smooth_window=3
                )
                val = stoch.stoch_signal().iloc[idx]
                indicators['stoch_d'] = val
                indicators['stoch.d'] = val
            
            if any(x in needed for x in ['macd', 'macd.macd', 'macd_signal', 'macd.signal']):
                macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
                indicators['macd'] = macd.macd().iloc[idx]
                indicators['macd.macd'] = indicators['macd']
                indicators['macd_signal'] = macd.macd_signal().iloc[idx]
                indicators['macd.signal'] = indicators['macd_signal']
            
            if 'adx' in needed:
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                indicators['adx'] = adx.adx().iloc[idx]
            
            if 'adx+di' in needed or 'adx_pos' in needed:
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                val = adx.adx_pos().iloc[idx]
                indicators['adx+di'] = val
                indicators['adx_pos'] = val
            
            if 'adx-di' in needed or 'adx_neg' in needed:
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
                val = adx.adx_neg().iloc[idx]
                indicators['adx-di'] = val
                indicators['adx_neg'] = val
            
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
            
            if 'atr' in needed:
                atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
                indicators['atr'] = atr.average_true_range().iloc[idx]
            
            if any(x in needed for x in ['bb.upper', 'bb.lower', 'bb.middle', 'bb_width', 'bbpower']):
                bb = BollingerBands(close=df['close'], window=20, window_dev=2)
                indicators['bb.upper'] = bb.bollinger_hband().iloc[idx]
                indicators['bb.lower'] = bb.bollinger_lband().iloc[idx]
                indicators['bb.middle'] = bb.bollinger_mavg().iloc[idx]
                indicators['bb_width'] = (indicators['bb.upper'] - indicators['bb.lower']) / indicators['bb.middle'] * 100
                
                # BBPower = (close - lower) / (upper - lower) * 100
                close_price = df['close'].iloc[idx]
                indicators['bbpower'] = ((close_price - indicators['bb.lower']) / 
                                        (indicators['bb.upper'] - indicators['bb.lower'])) * 100
            
            if 'volume' in needed:
                indicators['volume'] = df['volume'].iloc[idx]
            
            if 'volume_ratio' in needed:
                vol_sma = df['volume'].rolling(window=20).mean()
                indicators['volume_ratio'] = df['volume'].iloc[idx] / vol_sma.iloc[idx]
            
            if 'close' in needed:
                indicators['close'] = df['close'].iloc[idx]
            if 'open' in needed:
                indicators['open'] = df['open'].iloc[idx]
            if 'high' in needed:
                indicators['high'] = df['high'].iloc[idx]
            if 'low' in needed:
                indicators['low'] = df['low'].iloc[idx]
            
            # Comparison indicators (EMA crossovers, etc.)
            if 'ema50' in indicators and 'ema200' in indicators:
                indicators['ema50_above_ema200'] = 1.0 if indicators['ema50'] > indicators['ema200'] else 0.0
            
            if 'ema20' in indicators and 'ema50' in indicators:
                indicators['ema20_above_ema50'] = 1.0 if indicators['ema20'] > indicators['ema50'] else 0.0
            
        except Exception as e:
            self.logger.debug(f"Error calculating indicators for {symbol}: {e}")
        
        # Cache result
        self._indicator_cache[cache_key] = indicators
        
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
                'avg_gain_pct': None,
                'win_rate_pct': 0,
                'avg_winner_pct': None,
                'avg_loser_pct': None,
                'profit_factor': None,
                'intraday_target_hit_rate': 0
            }
        
        total_matches = sum(1 for t in all_trades if t['matched_criteria'])
        true_positives = sum(1 for t in all_trades if t['trade_type'] == 'true_positive')
        false_positives = sum(1 for t in all_trades if t['trade_type'] == 'false_positive')
        missed_opportunities = sum(1 for t in all_trades if t['trade_type'] == 'false_negative')
        
        accuracy = (true_positives / total_matches * 100) if total_matches > 0 else 0
        
        matched_trades = [t for t in all_trades if t['matched_criteria']]
        avg_gain = np.mean([t['actual_gain_pct'] for t in matched_trades]) if matched_trades else None
        
        # Win rate (% of matched trades that were profitable)
        winners = [t for t in matched_trades if t['actual_gain_pct'] > 0]
        losers = [t for t in matched_trades if t['actual_gain_pct'] < 0]
        win_rate = (len(winners) / len(matched_trades) * 100) if matched_trades else 0
        
        # Average winner vs loser
        avg_winner = np.mean([t['actual_gain_pct'] for t in winners]) if winners else None
        avg_loser = np.mean([t['actual_gain_pct'] for t in losers]) if losers else None
        
        # Profit factor (total gains / total losses)
        total_gains = sum(t['actual_gain_pct'] for t in winners) if winners else 0
        total_losses = abs(sum(t['actual_gain_pct'] for t in losers)) if losers else 0
        profit_factor = (total_gains / total_losses) if total_losses > 0 else None
        
        # Intraday target hit rate
        intraday_hits = sum(1 for t in matched_trades if t.get('target_hit_intraday', False))
        intraday_hit_rate = (intraday_hits / len(matched_trades) * 100) if matched_trades else 0
        
        return {
            'total_trades': len(all_trades),
            'total_matches': total_matches,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'missed_opportunities': missed_opportunities,
            'accuracy_pct': round(accuracy, 2),
            'avg_gain_pct': round(avg_gain, 2) if avg_gain else None,
            'win_rate_pct': round(win_rate, 2),
            'avg_winner_pct': round(avg_winner, 2) if avg_winner else None,
            'avg_loser_pct': round(avg_loser, 2) if avg_loser else None,
            'profit_factor': round(profit_factor, 2) if profit_factor else None,
            'intraday_target_hit_rate': round(intraday_hit_rate, 2)
        }
