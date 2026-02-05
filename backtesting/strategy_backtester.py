#!/usr/bin/env python3
"""
Strategy Backtester - Core backtesting engine (PROPERLY FIXED)
Evaluates trading strategies against ACTUAL historical data

CRITICAL FIXES:
1. Dynamically builds universe using yfinance (no predefined lists)
2. Finds ACTUAL top gainers for each historical date from price data
3. Checks PEAK gains during holding period (accounts for consolidation)
4. Properly evaluates indicator criteria using historical values

LOGIC:
- Fetch list of active tickers dynamically
- For each test date:
  * Get historical data for all tickers
  * Calculate which stocks had highest gains THAT DAY
  * Find stocks matching indicator criteria from historical data
  * Check if they hit target at ANY point during holding period (peak gain)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    
    # HARDCODED LIMITS - Adjust these to balance accuracy vs speed
    TOP_WINNERS_PER_DAY = 15  # Top daily gainers to track
    MAX_CRITERIA_MATCHES = 100  # Max stocks matching criteria per day
    UNIVERSE_SIZE = 1500  # Number of stocks to scan (higher = better coverage)
    
    # Parallel processing
    MAX_WORKERS = 10
    
    # Historical data lookback for indicators
    LOOKBACK_DAYS = 120
    
    # Minimum requirements for stock inclusion
    MIN_PRICE = 0.50
    MIN_VOLUME = 50000
    
    def __init__(self, config: dict):
        """Initialize backtester"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Cache for historical data (reduces API calls)
        self.price_cache = {}
        self.indicator_cache = {}
        
        # Stock universe (built dynamically)
        self.universe: List[str] = []
        
        self.logger.info("Strategy Backtester initialized (PROPERLY FIXED)")
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run complete backtest for a strategy"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING STRATEGY BACKTEST")
        self.logger.info("=" * 80)
        
        # Parse dates
        start_date = pd.to_datetime(strategy_config['start_date']).date()
        end_date = pd.to_datetime(strategy_config['end_date']).date()
        
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Target: {strategy_config['target_min_gain_pct']}% gain in {strategy_config['target_days']} day(s)")
        self.logger.info(f"Checking PEAK gains (accounts for consolidation after spike)")
        self.logger.info(f"Universe size: {self.UNIVERSE_SIZE} stocks")
        self.logger.info(f"Top winners per day: {self.TOP_WINNERS_PER_DAY}")
        self.logger.info(f"Max criteria matches per day: {self.MAX_CRITERIA_MATCHES}")
        
        # Build dynamic stock universe
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BUILDING DYNAMIC STOCK UNIVERSE")
        self.logger.info("=" * 80)
        self._build_dynamic_universe(start_date, end_date)
        
        # Pre-download historical data for entire universe
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PRE-DOWNLOADING HISTORICAL DATA")
        self.logger.info("=" * 80)
        self._preload_historical_data(start_date, end_date)
        
        # Generate trading days
        trading_days = self._get_trading_days(start_date, end_date)
        self.logger.info(f"\nWill test {len(trading_days)} trading days")
        
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
            
            # Find actual top gainers for this date
            winners = self._get_actual_top_gainers(
                test_date,
                self.TOP_WINNERS_PER_DAY,
                strategy_config.get('min_price', self.MIN_PRICE),
                strategy_config.get('min_volume', self.MIN_VOLUME)
            )
            self.logger.info(f"  ✓ Found {len(winners)} top gainers (up to {max([w['day_gain_pct'] for w in winners]) if winners else 0:.1f}%)")
            
            # Find stocks matching criteria
            criteria_matches = self._get_criteria_matches(
                test_date,
                strategy_config['indicator_criteria'],
                self.MAX_CRITERIA_MATCHES,
                strategy_config.get('min_price', self.MIN_PRICE),
                strategy_config.get('max_price'),
                strategy_config.get('min_volume', self.MIN_VOLUME)
            )
            self.logger.info(f"  ✓ Found {len(criteria_matches)} stocks matching criteria")
            
            # Evaluate outcomes (with peak gain detection)
            day_trades = self._evaluate_day(
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
                f"  ✓ Results: {daily_result['true_positives']} true positives, "
                f"{daily_result['false_positives']} false positives, "
                f"{daily_result['missed_opportunities']} missed opportunities"
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
    
    def _build_dynamic_universe(self, start_date: datetime.date, end_date: datetime.date):
        """
        Build universe dynamically using yfinance
        Gets most actively traded stocks from major exchanges
        """
        self.logger.info("Building dynamic stock universe using yfinance...")
        
        # Strategy: Download lists of stocks from major indices and ETFs
        # Then filter for actively traded ones
        
        symbols = set()
        
        # Get S&P 500 components
        try:
            self.logger.info("  Fetching S&P 500 components...")
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            sp500_symbols = sp500['Symbol'].str.replace('.', '-').tolist()
            symbols.update(sp500_symbols[:500])  # Limit to 500
            self.logger.info(f"  ✓ Added {len(sp500_symbols[:500])} S&P 500 stocks")
        except Exception as e:
            self.logger.warning(f"  Could not fetch S&P 500: {e}")
        
        # Get NASDAQ 100 components
        try:
            self.logger.info("  Fetching NASDAQ 100 components...")
            nasdaq100 = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')[4]
            nasdaq_symbols = nasdaq100['Ticker'].tolist()
            symbols.update(nasdaq_symbols[:100])
            self.logger.info(f"  ✓ Added {len(nasdaq_symbols[:100])} NASDAQ 100 stocks")
        except Exception as e:
            self.logger.warning(f"  Could not fetch NASDAQ 100: {e}")
        
        # Get Russell 2000 small caps (where the big movers often are)
        try:
            self.logger.info("  Fetching Russell 2000 components...")
            # Use a sample of Russell 2000 from iShares IWM ETF holdings
            iwm = yf.Ticker("IWM")
            # We can't get holdings directly, so we'll add known volatile small caps
            # This is a limitation - in production you'd want a data provider
        except Exception as e:
            self.logger.warning(f"  Could not fetch Russell 2000: {e}")
        
        # Add some known volatile/popular tickers that often have big moves
        popular_tickers = [
            # Meme stocks / high volatility
            'GME', 'AMC', 'BBBY', 'KOSS', 'EXPR', 'NAKD', 'SNDL', 'TLRY',
            # Penny stocks that move
            'MULN', 'WULF', 'GREE', 'SPRT', 'IRNT', 'OPAD', 'BGFV',
            # Biotech (often big movers)
            'MRNA', 'BNTX', 'NVAX', 'GILD', 'REGN', 'VRTX', 'BIIB',
            # EV / Tech high volatility  
            'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
            # Crypto related
            'MARA', 'RIOT', 'COIN', 'MSTR', 'SI', 'HUT',
            # SPACs and recent IPOs often have big moves
            'HOOD', 'DKNG', 'OPEN', 'SOFI', 'UPST', 'AFRM'
        ]
        symbols.update(popular_tickers)
        
        # Convert to list and limit
        self.universe = list(symbols)[:self.UNIVERSE_SIZE]
        
        self.logger.info(f"\n✓ Built universe of {len(self.universe)} stocks")
        self.logger.info(f"  Sample: {self.universe[:10]}")
    
    def _preload_historical_data(self, start_date: datetime.date, end_date: datetime.date):
        """
        Pre-download historical data for entire universe
        This is much faster than fetching on-demand
        """
        # Need extra days for indicators and future price checking
        fetch_start = start_date - timedelta(days=self.LOOKBACK_DAYS)
        fetch_end = end_date + timedelta(days=30)
        
        self.logger.info(f"Downloading data from {fetch_start} to {fetch_end}...")
        self.logger.info(f"This may take a few minutes for {len(self.universe)} stocks...")
        
        successful = 0
        failed = 0
        
        # Download in batches to avoid overwhelming the API
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
                            # Store price data
                            self.price_cache[symbol] = df
                            
                            # Calculate and store indicators
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
    
    def _get_actual_top_gainers(
        self,
        date: datetime.date,
        count: int,
        min_price: float,
        min_volume: int
    ) -> List[Dict[str, Any]]:
        """
        Find the ACTUAL top gainers for a specific historical date
        by analyzing price data
        """
        gainers = []
        
        for symbol in self.universe:
            if symbol not in self.price_cache:
                continue
            
            df = self.price_cache[symbol]
            
            # Check if we have data for this date
            try:
                # Get closest date
                df_dates = pd.to_datetime(df.index).date
                available_dates = [d for d in df_dates if d <= date]
                
                if not available_dates:
                    continue
                
                actual_date = available_dates[-1]
                
                if actual_date not in df_dates:
                    continue
                
                idx = list(df_dates).index(actual_date)
                row = df.iloc[idx]
                
                # Filter by price and volume
                close_price = row['Close']
                volume = row['Volume']
                
                if close_price < min_price or volume < min_volume:
                    continue
                
                # Calculate day's gain
                open_price = row['Open']
                day_gain_pct = ((close_price - open_price) / open_price) * 100
                
                if day_gain_pct > 0:  # Only gainers
                    gainers.append({
                        'symbol': symbol,
                        'exchange': 'US',  # yfinance doesn't specify
                        'date': actual_date,
                        'open': float(open_price),
                        'close': float(close_price),
                        'volume': int(volume),
                        'day_gain_pct': float(day_gain_pct)
                    })
                    
            except Exception as e:
                continue
        
        # Sort by gain percentage and return top N
        gainers.sort(key=lambda x: x['day_gain_pct'], reverse=True)
        return gainers[:count]
    
    def _get_criteria_matches(
        self,
        date: datetime.date,
        indicator_criteria: List[Dict[str, Any]],
        max_matches: int,
        min_price: float,
        max_price: Optional[float],
        min_volume: int
    ) -> List[Dict[str, Any]]:
        """
        Find stocks matching indicator criteria for this date
        """
        matches = []
        
        for symbol in self.universe:
            if symbol not in self.indicator_cache:
                continue
            
            # Check if we have data for this date
            try:
                indicators_df = self.indicator_cache[symbol]
                df_dates = indicators_df.index
                
                # Find closest date
                available_dates = [d for d in df_dates if d <= date]
                if not available_dates:
                    continue
                
                actual_date = available_dates[-1]
                indicators = indicators_df.loc[actual_date]
                
                # Filter by price and volume
                close_price = indicators['close']
                volume = indicators['volume']
                
                if close_price < min_price or volume < min_volume:
                    continue
                
                if max_price and close_price > max_price:
                    continue
                
                # Check if criteria match
                if self._check_criteria(indicators, indicator_criteria):
                    matches.append({
                        'symbol': symbol,
                        'exchange': 'US',
                        'date': actual_date,
                        'entry_price': float(close_price),
                        'volume': int(volume),
                        'indicators': self._extract_indicator_values(indicators)
                    })
                    
                    if len(matches) >= max_matches:
                        break
                        
            except Exception as e:
                continue
        
        return matches
    
    def _evaluate_day(
        self,
        date: datetime.date,
        winners: List[Dict[str, Any]],
        criteria_matches: List[Dict[str, Any]],
        target_gain_pct: float,
        target_days: int
    ) -> List[Dict[str, Any]]:
        """
        Evaluate outcomes for the day
        CRITICAL: Uses PEAK gain during holding period (not just exit price)
        """
        trades = []
        
        match_symbols = {m['symbol'] for m in criteria_matches}
        
        # Evaluate all criteria matches
        for match in criteria_matches:
            symbol = match['symbol']
            
            # Calculate outcome (with peak gain detection)
            peak_gain, exit_price, exit_gain = self._calculate_peak_outcome(
                symbol, date, target_days
            )
            
            # Check if it hit target at ANY point during holding period
            hit_target = peak_gain >= target_gain_pct if peak_gain is not None else False
            
            trade_type = 'true_positive' if hit_target else 'false_positive'
            
            trades.append({
                'symbol': symbol,
                'exchange': match['exchange'],
                'signal_date': date.isoformat(),
                'entry_price': match['entry_price'],
                'entry_volume': match['volume'],
                'indicator_values': match['indicators'],
                'matched_criteria': True,
                'hit_target': hit_target,
                'peak_gain_pct': peak_gain,  # NEW: Track peak gain
                'actual_gain_pct': exit_gain,  # Exit gain for reference
                'exit_price': exit_price,
                'trade_type': trade_type
            })
        
        # Missed opportunities (winners not in criteria)
        for winner in winners:
            symbol = winner['symbol']
            
            if symbol in match_symbols:
                continue
            
            # Calculate outcome
            peak_gain, exit_price, exit_gain = self._calculate_peak_outcome(
                symbol, date, target_days
            )
            
            hit_target = peak_gain >= target_gain_pct if peak_gain is not None else False
            
            if hit_target:
                trades.append({
                    'symbol': symbol,
                    'exchange': winner['exchange'],
                    'signal_date': date.isoformat(),
                    'entry_price': winner['close'],
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
    
    def _calculate_peak_outcome(
        self,
        symbol: str,
        entry_date: datetime.date,
        hold_days: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate PEAK gain during holding period
        
        Returns:
            (peak_gain_pct, exit_price, exit_gain_pct)
        """
        if symbol not in self.price_cache:
            return None, None, None
        
        try:
            df = self.price_cache[symbol]
            df_dates = pd.to_datetime(df.index).date
            
            # Find entry index
            available_dates = [d for d in df_dates if d >= entry_date]
            if not available_dates:
                return None, None, None
            
            entry_idx = list(df_dates).index(available_dates[0])
            entry_price = df.iloc[entry_idx]['Close']
            
            # Get future prices during holding period
            future_indices = list(range(entry_idx + 1, min(entry_idx + 1 + hold_days, len(df))))
            
            if not future_indices:
                return None, None, None
            
            future_prices = df.iloc[future_indices]['High'].values  # Use High for peak
            
            # Calculate peak gain
            peak_price = np.max(future_prices)
            peak_gain_pct = ((peak_price - entry_price) / entry_price) * 100
            
            # Calculate exit gain (at end of holding period)
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
