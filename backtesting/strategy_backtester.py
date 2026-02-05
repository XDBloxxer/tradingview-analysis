#!/usr/bin/env python3
"""
Strategy Backtester - CORRECTLY ARCHITECTED VERSION

CRITICAL FIXES:
1. Calculates day-over-day gains properly (prev close -> current close)
2. No predefined ticker lists - 100% dynamic from public sources
3. Finds REAL top gainers (100%+ movers)
4. Proper backtesting logic: "Could my criteria have caught this yesterday?"

PROPER LOGIC:
- For each test date:
  * Find stocks with highest gain from (yesterday close -> today close)
  * Check: Did my indicators YESTERDAY predict these winners?
  * Track peak gains during holding period from entry at yesterday's close
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import time

# Technical analysis
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class StrategyBacktester:
    """
    Backtests trading strategies using actual historical market data
    """
    
    # HARDCODED LIMITS - Adjust these
    TOP_WINNERS_PER_DAY = 20  # Top daily gainers to track
    MAX_CRITERIA_MATCHES = 150  # Max stocks matching criteria per day
    UNIVERSE_SIZE = 2000  # Number of stocks to scan (larger = more big movers)
    
    # Parallel processing
    MAX_WORKERS = 10
    
    # Historical data lookback for indicators
    LOOKBACK_DAYS = 120
    
    # Minimum requirements
    MIN_PRICE = 0.50
    MIN_VOLUME = 50000
    
    def __init__(self, config: dict):
        """Initialize backtester"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Cache for historical data
        self.price_cache = {}
        self.indicator_cache = {}
        
        # Stock universe (100% dynamic - NO HARDCODED LISTS)
        self.universe: List[str] = []
        
        self.logger.info("Strategy Backtester initialized (CORRECTLY ARCHITECTED)")
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run complete backtest for a strategy"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING STRATEGY BACKTEST - CORRECT ARCHITECTURE")
        self.logger.info("=" * 80)
        
        # Parse dates
        start_date = pd.to_datetime(strategy_config['start_date']).date()
        end_date = pd.to_datetime(strategy_config['end_date']).date()
        
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Target: {strategy_config['target_min_gain_pct']}% gain in {strategy_config['target_days']} day(s)")
        self.logger.info(f"Tracking PEAK gains during holding period")
        self.logger.info(f"Calculating day-over-day gains (prev close -> current close)")
        self.logger.info(f"Universe size: {self.UNIVERSE_SIZE} stocks (NO PREDEFINED LISTS)")
        
        # Build 100% dynamic universe
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BUILDING 100% DYNAMIC STOCK UNIVERSE")
        self.logger.info("=" * 80)
        self._build_fully_dynamic_universe()
        
        # Pre-download historical data
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PRE-DOWNLOADING HISTORICAL DATA")
        self.logger.info("=" * 80)
        self._preload_historical_data(start_date, end_date)
        
        # Generate trading days
        trading_days = self._get_trading_days(start_date, end_date)
        self.logger.info(f"\nWill test {len(trading_days)} trading days")
        self.logger.info(f"Expected total records: ~{len(trading_days) * (self.TOP_WINNERS_PER_DAY + self.MAX_CRITERIA_MATCHES)}")
        
        # Results storage
        all_trades = []
        daily_results = []
        
        # Process each day
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANALYZING EACH TRADING DAY")
        self.logger.info("=" * 80)
        
        for idx, test_date in enumerate(trading_days):
            if progress_callback:
                progress_callback(idx + 1, len(trading_days), test_date)
            
            self.logger.info(f"\n[{idx + 1}/{len(trading_days)}] Processing {test_date}...")
            
            # Find actual top gainers (prev close -> current close)
            winners = self._get_actual_top_gainers_correct(
                test_date,
                self.TOP_WINNERS_PER_DAY,
                strategy_config.get('min_price', self.MIN_PRICE),
                strategy_config.get('min_volume', self.MIN_VOLUME)
            )
            
            if winners:
                max_gain = max([w['day_gain_pct'] for w in winners])
                self.logger.info(f"  ✓ Found {len(winners)} top gainers (max: {max_gain:.1f}%)")
            else:
                self.logger.info(f"  ✓ Found {len(winners)} top gainers")
            
            # Find stocks matching criteria YESTERDAY (for prediction)
            criteria_matches = self._get_criteria_matches_previous_day(
                test_date,
                strategy_config['indicator_criteria'],
                self.MAX_CRITERIA_MATCHES,
                strategy_config.get('min_price', self.MIN_PRICE),
                strategy_config.get('max_price'),
                strategy_config.get('min_volume', self.MIN_VOLUME)
            )
            self.logger.info(f"  ✓ Found {len(criteria_matches)} stocks matching criteria yesterday")
            
            # Evaluate outcomes
            day_trades = self._evaluate_day_correct(
                test_date,
                winners,
                criteria_matches,
                strategy_config['target_min_gain_pct'],
                strategy_config['target_days']
            )
            
            all_trades.extend(day_trades)
            
            # Aggregate daily stats
            daily_result = self._aggregate_daily_results(
                test_date, day_trades, len(winners), len(criteria_matches)
            )
            daily_results.append(daily_result)
            
            self.logger.info(
                f"  ✓ {daily_result['true_positives']} true positives, "
                f"{daily_result['false_positives']} false positives, "
                f"{daily_result['missed_opportunities']} missed"
            )
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(all_trades, daily_results)
        
        results = {
            'trades': all_trades,
            'daily_results': daily_results,
            'overall_stats': overall_stats
        }
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BACKTEST COMPLETED")
        self.logger.info("=" * 80)
        self._log_summary(overall_stats)
        
        return results
    
    def _build_fully_dynamic_universe(self):
        """
        Build universe 100% dynamically - NO PREDEFINED LISTS
        Fetches ticker lists from public sources only
        """
        self.logger.info("Building 100% dynamic stock universe...")
        
        symbols = set()
        
        # Get S&P 500 (dynamic from Wikipedia)
        try:
            self.logger.info("  Fetching S&P 500 components...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500 = tables[0]
            sp500_symbols = sp500['Symbol'].str.replace('.', '-').tolist()
            symbols.update(sp500_symbols)
            self.logger.info(f"  ✓ Added {len(sp500_symbols)} S&P 500 stocks")
        except Exception as e:
            self.logger.warning(f"  Failed to fetch S&P 500: {e}")
        
        # Get NASDAQ 100 (dynamic from Wikipedia)
        try:
            self.logger.info("  Fetching NASDAQ 100 components...")
            url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
            tables = pd.read_html(url)
            nasdaq100 = tables[4]  # The holdings table
            nasdaq_symbols = nasdaq100['Ticker'].tolist()
            symbols.update(nasdaq_symbols)
            self.logger.info(f"  ✓ Added {len(nasdaq_symbols)} NASDAQ 100 stocks")
        except Exception as e:
            self.logger.warning(f"  Failed to fetch NASDAQ 100: {e}")
        
        # Get Dow Jones Industrial Average (dynamic)
        try:
            self.logger.info("  Fetching Dow Jones components...")
            url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
            tables = pd.read_html(url)
            dow = tables[1]
            dow_symbols = dow['Symbol'].tolist()
            symbols.update(dow_symbols)
            self.logger.info(f"  ✓ Added {len(dow_symbols)} Dow Jones stocks")
        except Exception as e:
            self.logger.warning(f"  Failed to fetch Dow Jones: {e}")
        
        # Get S&P 400 MidCap
        try:
            self.logger.info("  Fetching S&P 400 MidCap components...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
            tables = pd.read_html(url)
            sp400 = tables[0]
            sp400_symbols = sp400['Symbol'].str.replace('.', '-').tolist()
            symbols.update(sp400_symbols)
            self.logger.info(f"  ✓ Added {len(sp400_symbols)} S&P 400 stocks")
        except Exception as e:
            self.logger.warning(f"  Failed to fetch S&P 400: {e}")
        
        # Get S&P 600 SmallCap (more big movers here)
        try:
            self.logger.info("  Fetching S&P 600 SmallCap components...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
            tables = pd.read_html(url)
            sp600 = tables[0]
            sp600_symbols = sp600['Symbol'].str.replace('.', '-').tolist()
            symbols.update(sp600_symbols)
            self.logger.info(f"  ✓ Added {len(sp600_symbols)} S&P 600 stocks")
        except Exception as e:
            self.logger.warning(f"  Failed to fetch S&P 600: {e}")
        
        # Convert to list and limit
        self.universe = list(symbols)[:self.UNIVERSE_SIZE]
        
        self.logger.info(f"\n✓ Built 100% dynamic universe of {len(self.universe)} stocks")
        self.logger.info(f"  NO PREDEFINED LISTS - All fetched dynamically")
        self.logger.info(f"  Sample: {self.universe[:10]}")
    
    def _preload_historical_data(self, start_date: datetime.date, end_date: datetime.date):
        """Pre-download historical data for entire universe"""
        # Need extra days for indicators and future price checking
        fetch_start = start_date - timedelta(days=self.LOOKBACK_DAYS)
        fetch_end = end_date + timedelta(days=30)
        
        self.logger.info(f"Downloading data from {fetch_start} to {fetch_end}...")
        self.logger.info(f"This will take a few minutes for {len(self.universe)} stocks...")
        
        successful = 0
        failed = 0
        
        # Download in batches
        batch_size = 50
        
        for i in range(0, len(self.universe), batch_size):
            batch = self.universe[i:i + batch_size]
            
            try:
                # Download batch
                data = yf.download(
                    batch,
                    start=fetch_start,
                    end=fetch_end,
                    group_by='ticker',
                    threads=True,
                    progress=False
                )
                
                # Store in cache
                for symbol in batch:
                    try:
                        if len(batch) == 1:
                            df = data
                        else:
                            df = data[symbol]
                        
                        if isinstance(df, pd.DataFrame) and not df.empty and len(df) > 50:
                            self.price_cache[symbol] = df
                            self.indicator_cache[symbol] = self._calculate_indicators(df)
                            successful += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1
                        continue
                
                # Rate limiting
                time.sleep(0.5)
                
                if (i // batch_size) % 5 == 0:
                    self.logger.info(f"  Progress: {i + len(batch)}/{len(self.universe)} stocks...")
                    
            except Exception as e:
                self.logger.warning(f"  Batch download failed: {e}")
                failed += len(batch)
                continue
        
        self.logger.info(f"\n✓ Downloaded data for {successful} stocks ({failed} failed)")
    
    def _get_actual_top_gainers_correct(
        self,
        date: datetime.date,
        count: int,
        min_price: float,
        min_volume: int
    ) -> List[Dict[str, Any]]:
        """
        Find ACTUAL top gainers using CORRECT calculation:
        (today's close - yesterday's close) / yesterday's close
        
        This is what "top gainers" actually means in the market
        """
        gainers = []
        
        for symbol in self.universe:
            if symbol not in self.price_cache:
                continue
            
            df = self.price_cache[symbol]
            
            try:
                df_dates = pd.to_datetime(df.index).date
                
                # Find today's date in data
                available_dates = [d for d in df_dates if d <= date]
                if not available_dates:
                    continue
                
                today = available_dates[-1]
                today_idx = list(df_dates).index(today)
                
                # Need yesterday for calculation
                if today_idx == 0:
                    continue
                
                yesterday_idx = today_idx - 1
                
                # Get prices
                yesterday_close = df.iloc[yesterday_idx]['Close']
                today_close = df.iloc[today_idx]['Close']
                today_volume = df.iloc[today_idx]['Volume']
                
                # Filter by price and volume
                if today_close < min_price or today_volume < min_volume:
                    continue
                
                # Calculate CORRECT day-over-day gain
                day_gain_pct = ((today_close - yesterday_close) / yesterday_close) * 100
                
                if day_gain_pct > 0:  # Only gainers
                    gainers.append({
                        'symbol': symbol,
                        'exchange': 'US',
                        'date': today,
                        'prev_close': float(yesterday_close),
                        'close': float(today_close),
                        'volume': int(today_volume),
                        'day_gain_pct': float(day_gain_pct)
                    })
                    
            except Exception as e:
                continue
        
        # Sort by gain percentage and return top N
        gainers.sort(key=lambda x: x['day_gain_pct'], reverse=True)
        return gainers[:count]
    
    def _get_criteria_matches_previous_day(
        self,
        date: datetime.date,
        indicator_criteria: List[Dict[str, Any]],
        max_matches: int,
        min_price: float,
        max_price: Optional[float],
        min_volume: int
    ) -> List[Dict[str, Any]]:
        """
        Find stocks matching criteria on the PREVIOUS day
        
        This is the correct logic: "What would my criteria have flagged yesterday
        that I could have bought at yesterday's close?"
        """
        matches = []
        
        for symbol in self.universe:
            if symbol not in self.indicator_cache:
                continue
            
            try:
                indicators_df = self.indicator_cache[symbol]
                df_dates = indicators_df.index
                
                # Find yesterday (one day before test date)
                available_dates = [d for d in df_dates if d < date]
                if not available_dates:
                    continue
                
                yesterday = available_dates[-1]
                indicators = indicators_df.loc[yesterday]
                
                # Filter by price and volume
                close_price = indicators['close']
                volume = indicators['volume']
                
                if close_price < min_price or volume < min_volume:
                    continue
                
                if max_price and close_price > max_price:
                    continue
                
                # Check if criteria match YESTERDAY
                if self._check_criteria(indicators, indicator_criteria):
                    matches.append({
                        'symbol': symbol,
                        'exchange': 'US',
                        'signal_date': yesterday,  # When we got the signal
                        'entry_date': date,  # When we would enter (next day)
                        'entry_price': float(close_price),  # Enter at yesterday's close
                        'volume': int(volume),
                        'indicators': self._extract_indicator_values(indicators)
                    })
                    
                    if len(matches) >= max_matches:
                        break
                        
            except Exception as e:
                continue
        
        return matches
    
    def _evaluate_day_correct(
        self,
        date: datetime.date,
        winners: List[Dict[str, Any]],
        criteria_matches: List[Dict[str, Any]],
        target_gain_pct: float,
        target_days: int
    ) -> List[Dict[str, Any]]:
        """
        Evaluate outcomes with CORRECT logic
        Entry is at yesterday's close (when criteria matched)
        """
        trades = []
        
        match_symbols = {m['symbol'] for m in criteria_matches}
        
        # Evaluate all criteria matches
        for match in criteria_matches:
            symbol = match['symbol']
            entry_date = match['entry_date']
            entry_price = match['entry_price']
            
            # Calculate outcome from entry
            peak_gain, exit_price, exit_gain = self._calculate_peak_outcome_from_entry(
                symbol, entry_date, entry_price, target_days
            )
            
            # Check if it hit target
            hit_target = peak_gain >= target_gain_pct if peak_gain is not None else False
            
            trade_type = 'true_positive' if hit_target else 'false_positive'
            
            trades.append({
                'symbol': symbol,
                'exchange': match['exchange'],
                'signal_date': match['signal_date'].isoformat(),
                'entry_price': entry_price,
                'entry_volume': match['volume'],
                'indicator_values': match['indicators'],
                'matched_criteria': True,
                'hit_target': hit_target,
                'peak_gain_pct': peak_gain,
                'actual_gain_pct': exit_gain,
                'exit_price': exit_price,
                'trade_type': trade_type
            })
        
        # Missed opportunities
        for winner in winners:
            symbol = winner['symbol']
            
            if symbol in match_symbols:
                continue
            
            # Would have entered at yesterday's close
            entry_price = winner['prev_close']
            
            # Calculate outcome
            peak_gain, exit_price, exit_gain = self._calculate_peak_outcome_from_entry(
                symbol, date, entry_price, target_days
            )
            
            hit_target = peak_gain >= target_gain_pct if peak_gain is not None else False
            
            if hit_target:
                trades.append({
                    'symbol': symbol,
                    'exchange': winner['exchange'],
                    'signal_date': date.isoformat(),
                    'entry_price': entry_price,
                    'entry_volume': winner['volume'],
                    'indicator_values': {},
                    'matched_criteria': False,
                    'hit_target': True,
                    'peak_gain_pct': peak_gain,
                    'actual_gain_pct': exit_gain,
                    'exit_price': exit_price,
                    'trade_type': 'false_negative'
                })
        
        return trades
    
    def _calculate_peak_outcome_from_entry(
        self,
        symbol: str,
        entry_date: datetime.date,
        entry_price: float,
        hold_days: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate peak gain from a specific entry price and date
        """
        if symbol not in self.price_cache:
            return None, None, None
        
        try:
            df = self.price_cache[symbol]
            df_dates = pd.to_datetime(df.index).date
            
            # Find entry date index
            available_dates = [d for d in df_dates if d >= entry_date]
            if not available_dates:
                return None, None, None
            
            entry_idx = list(df_dates).index(available_dates[0])
            
            # Get future prices during holding period
            future_indices = list(range(entry_idx, min(entry_idx + hold_days, len(df))))
            
            if len(future_indices) < 2:  # Need at least entry + 1 day
                return None, None, None
            
            # Use High prices for peak detection
            future_highs = df.iloc[future_indices[1:]]['High'].values
            
            # Calculate peak gain
            peak_price = np.max(future_highs)
            peak_gain_pct = ((peak_price - entry_price) / entry_price) * 100
            
            # Calculate exit gain
            exit_idx = future_indices[-1]
            exit_price = df.iloc[exit_idx]['Close']
            exit_gain_pct = ((exit_price - entry_price) / entry_price) * 100
            
            return float(peak_gain_pct), float(exit_price), float(exit_gain_pct)
            
        except Exception as e:
            self.logger.debug(f"Error calculating outcome for {symbol}: {e}")
            return None, None, None
    
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
        
        # Convert index to date
        result.index = pd.to_datetime(result.index).date
        
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
    
    def _extract_indicator_values(self, indicators: pd.Series) -> Dict[str, Any]:
        """Extract indicator values to dict"""
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
    
    def _get_trading_days(
        self,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> List[datetime.date]:
        """Get list of trading days"""
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')
        trading_days = [d.date() for d in all_days]
        holidays = self._get_us_holidays(start_date.year, end_date.year)
        trading_days = [d for d in trading_days if d not in holidays]
        return trading_days
    
    def _get_us_holidays(self, start_year: int, end_year: int) -> set:
        """Get US market holidays"""
        holidays = set()
        for year in range(start_year, end_year + 1):
            holidays.add(datetime(year, 1, 1).date())
            holidays.add(datetime(year, 7, 4).date())
            holidays.add(datetime(year, 12, 25).date())
        return holidays
    
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
        
        match_gains = [t['peak_gain_pct'] for t in matches if t['peak_gain_pct'] is not None]
        miss_gains = [t['peak_gain_pct'] for t in misses if t['peak_gain_pct'] is not None]
        all_gains = [t['peak_gain_pct'] for t in trades if t['peak_gain_pct'] is not None]
        
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
        
        match_gains = [t['peak_gain_pct'] for t in matches if t['peak_gain_pct'] is not None]
        all_gains = [t['peak_gain_pct'] for t in trades if t['peak_gain_pct'] is not None]
        
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
        self.logger.info("\nOVERALL RESULTS:")
        self.logger.info(f"  Total Trades: {stats['total_trades']}")
        self.logger.info(f"  Criteria Matches: {stats['total_matches']}")
        self.logger.info(f"  True Positives: {stats['true_positives']}")
        self.logger.info(f"  False Positives: {stats['false_positives']}")
        self.logger.info(f"  Missed Opportunities: {stats['missed_opportunities']}")
        self.logger.info(f"  Accuracy: {stats['accuracy_pct']}%")
        if stats['avg_gain_pct']:
            self.logger.info(f"  Average Peak Gain: {stats['avg_gain_pct']}%")
        if stats['max_gain_pct']:
            self.logger.info(f"  Max Peak Gain: {stats['max_gain_pct']}%")
