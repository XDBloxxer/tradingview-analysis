"""
Daily Non-Winners Detector
Finds stocks that did NOT explode - critical negative examples for ML training
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tradingview_scraper.symbols.screener import Screener
    SCREENER_AVAILABLE = True
except ImportError:
    SCREENER_AVAILABLE = False
    logging.warning("tradingview-scraper not available - will use yfinance only")

from .rate_limiter import RateLimiter


class DailyNonWinnersDetector:
    """
    Detects stocks that did NOT explode (negative examples)
    
    Strategy:
    1. Get list of stocks that WERE screened/predicted
    2. Get list of actual winners for the day
    3. Non-winners = screened stocks - winners
    4. Sample diverse non-winners (different changes: flat, slight up, slight down, down)
    """
    
    # Pattern exclusions (same as winners detector)
    EXCLUDED_PATTERNS = ['OTC', '.PK', '.OB', '-']
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        detection_config = config.get("detection", {})
        
        self.min_price = detection_config.get("min_price", 0.25)
        self.min_volume = detection_config.get("min_volume", 10000)
        
        self.rate_limiter = RateLimiter(config)
        
        if SCREENER_AVAILABLE:
            self.screener = Screener()
        else:
            self.screener = None
        
        self.logger.info(
            f"Non-Winners detector initialized: "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_non_winners(
        self, 
        top_n: int = 15, 
        target_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Detect non-winners (negative examples)
        
        Strategy:
        1. Get diverse sample of stocks with various performance levels
        2. Exclude any that became winners (>20% gain)
        3. Sample from different categories:
           - Flat: -2% to +2%
           - Slight gainers: +2% to +10%
           - Slight losers: -2% to -10%
           - Big losers: < -10%
        
        Args:
            top_n: Number of non-winners to collect
            target_date: Target date
            
        Returns:
            List of non-winner dictionaries
        """
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        self.logger.info(f"Detecting non-winners for {target_date_str}...")
        self.logger.info(f"Strategy: Sample {top_n} diverse stocks that did NOT explode")
        
        # Get actual winners to exclude
        winners_symbols = self._get_winners_symbols(target_date)
        self.logger.info(f"Found {len(winners_symbols)} winners to exclude")
        
        # Collect diverse non-winners
        non_winners = []
        
        # Category 1: Flat stocks (-2% to +2%)
        flat_count = int(top_n * 0.3)  # 30%
        self.logger.info(f"Collecting {flat_count} flat stocks (-2% to +2%)...")
        flat_stocks = self._get_stocks_by_change_range(
            target_date, -2.0, 2.0, flat_count * 2, winners_symbols
        )
        non_winners.extend(flat_stocks[:flat_count])
        
        # Category 2: Slight gainers (+2% to +10%)
        slight_gain_count = int(top_n * 0.3)  # 30%
        self.logger.info(f"Collecting {slight_gain_count} slight gainers (+2% to +10%)...")
        slight_gainers = self._get_stocks_by_change_range(
            target_date, 2.0, 10.0, slight_gain_count * 2, winners_symbols
        )
        non_winners.extend(slight_gainers[:slight_gain_count])
        
        # Category 3: Slight losers (-2% to -10%)
        slight_loss_count = int(top_n * 0.2)  # 20%
        self.logger.info(f"Collecting {slight_loss_count} slight losers (-2% to -10%)...")
        slight_losers = self._get_stocks_by_change_range(
            target_date, -10.0, -2.0, slight_loss_count * 2, winners_symbols
        )
        non_winners.extend(slight_losers[:slight_loss_count])
        
        # Category 4: Bigger losers (< -10%)
        big_loss_count = top_n - len(non_winners)  # Fill remaining
        self.logger.info(f"Collecting {big_loss_count} bigger losers (< -10%)...")
        big_losers = self._get_stocks_by_change_range(
            target_date, -50.0, -10.0, big_loss_count * 2, winners_symbols
        )
        non_winners.extend(big_losers[:big_loss_count])
        
        # Verify we have enough
        if len(non_winners) < top_n:
            shortage = top_n - len(non_winners)
            self.logger.warning(f"Only found {len(non_winners)}/{top_n} non-winners")
            self.logger.info(f"Filling shortage with random liquid stocks...")
            
            additional = self._get_random_liquid_stocks(
                target_date, shortage * 2, winners_symbols, set(nw['symbol'] for nw in non_winners)
            )
            non_winners.extend(additional[:shortage])
        
        # Add metadata
        for nw in non_winners:
            nw['detection_date'] = target_date_str
            nw['detection_time'] = '16:00:00'
        
        self.logger.info(f"✓ Collected {len(non_winners)} diverse non-winners:")
        
        # Log distribution
        flat = sum(1 for nw in non_winners if -2 <= nw['change_pct'] <= 2)
        slight_gain = sum(1 for nw in non_winners if 2 < nw['change_pct'] <= 10)
        slight_loss = sum(1 for nw in non_winners if -10 <= nw['change_pct'] < -2)
        big_loss = sum(1 for nw in non_winners if nw['change_pct'] < -10)
        
        self.logger.info(f"  Distribution:")
        self.logger.info(f"    Flat (-2% to +2%): {flat}")
        self.logger.info(f"    Slight gainers (+2% to +10%): {slight_gain}")
        self.logger.info(f"    Slight losers (-2% to -10%): {slight_loss}")
        self.logger.info(f"    Bigger losers (< -10%): {big_loss}")
        
        return non_winners[:top_n]
    
    def _get_winners_symbols(self, target_date: datetime) -> set:
        """Get set of symbols that were winners on this date"""
        try:
            # Import here to avoid circular dependency
            from .daily_winners_supabase_client import DailyWinnersSupabaseClient
            
            client = DailyWinnersSupabaseClient(self.config)
            winners_df = client.read_winners(
                start_date=target_date.date().isoformat(),
                end_date=target_date.date().isoformat()
            )
            
            if winners_df.empty:
                return set()
            
            return set(winners_df['symbol'].tolist())
            
        except Exception as e:
            self.logger.warning(f"Could not fetch winners: {e}")
            return set()
    
    def _get_stocks_by_change_range(
        self,
        target_date: datetime,
        min_change: float,
        max_change: float,
        limit: int,
        exclude_symbols: set
    ) -> List[Dict[str, Any]]:
        """
        Get stocks within a specific change percentage range
        
        Args:
            target_date: Target date
            min_change: Minimum change %
            max_change: Maximum change %
            limit: Number of stocks to get
            exclude_symbols: Symbols to exclude (winners)
            
        Returns:
            List of stock dictionaries
        """
        try:
            if SCREENER_AVAILABLE and self.screener:
                return self._screen_by_change_range(
                    min_change, max_change, limit, exclude_symbols
                )
            else:
                return self._get_from_liquid_stocks(
                    target_date, min_change, max_change, limit, exclude_symbols
                )
        except Exception as e:
            self.logger.error(f"Error getting stocks in range {min_change}% to {max_change}%: {e}")
            return []
    
    def _screen_by_change_range(
        self,
        min_change: float,
        max_change: float,
        limit: int,
        exclude_symbols: set
    ) -> List[Dict[str, Any]]:
        """Use TradingView screener to find stocks in change range"""
        try:
            filters = [
                {'left': 'close', 'operation': 'greater', 'right': self.min_price},
                {'left': 'volume', 'operation': 'greater', 'right': self.min_volume},
                {'left': 'change', 'operation': 'greater', 'right': min_change},
                {'left': 'change', 'operation': 'less', 'right': max_change}
            ]
            
            result = self.screener.screen(
                market='america',
                filters=filters,
                sort_by='volume',
                sort_order='desc',
                limit=limit * 2
            )
            
            if result['status'] != 'success' or not result.get('data'):
                return []
            
            candidates = []
            for item in result['data']:
                try:
                    symbol_full = item.get('symbol', '')
                    if ':' in symbol_full:
                        exchange_prefix, symbol = symbol_full.split(':', 1)
                    else:
                        symbol = symbol_full
                        exchange_prefix = 'NASDAQ'
                    
                    if not symbol or symbol in exclude_symbols:
                        continue
                    
                    if self._is_excluded_symbol(symbol, exchange_prefix):
                        continue
                    
                    price = float(item.get('close', 0))
                    change_pct = float(item.get('change', 0))
                    volume = int(item.get('volume', 0))
                    
                    if price < self.min_price or volume < self.min_volume:
                        continue
                    
                    candidates.append({
                        'symbol': symbol.strip().upper(),
                        'exchange': exchange_prefix,
                        'price': float(price),
                        'change_pct': float(change_pct),
                        'volume': int(volume),
                        'high': float(item.get('high', price)),
                        'low': float(item.get('low', price)),
                        'open': float(item.get('open', price)),
                        'close': float(price)
                    })
                    
                    if len(candidates) >= limit:
                        break
                        
                except Exception as e:
                    continue
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"TradingView screener error: {e}")
            return []
    
    def _get_from_liquid_stocks(
        self,
        target_date: datetime,
        min_change: float,
        max_change: float,
        limit: int,
        exclude_symbols: set
    ) -> List[Dict[str, Any]]:
        """Get stocks from liquid stock list using yfinance"""
        liquid_stocks = self._get_liquid_stocks_list()
        liquid_stocks = [s for s in liquid_stocks if s not in exclude_symbols]
        
        candidates = []
        
        for symbol in liquid_stocks:
            if len(candidates) >= limit:
                break
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d', interval='1d')
                
                if hist.empty or len(hist) < 2:
                    continue
                
                latest = hist.iloc[-1]
                previous = hist.iloc[-2]
                
                close = latest['Close']
                prev_close = previous['Close']
                volume = latest['Volume']
                
                change_pct = ((close - prev_close) / prev_close) * 100
                
                if min_change <= change_pct <= max_change:
                    if close >= self.min_price and volume >= self.min_volume:
                        candidates.append({
                            'symbol': symbol,
                            'exchange': 'NASDAQ',
                            'price': float(close),
                            'change_pct': float(change_pct),
                            'volume': int(volume),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'open': float(latest['Open']),
                            'close': float(close)
                        })
                
            except Exception as e:
                continue
            
            time.sleep(0.1)
        
        return candidates
    
    def _get_random_liquid_stocks(
        self,
        target_date: datetime,
        limit: int,
        exclude_winners: set,
        exclude_existing: set
    ) -> List[Dict[str, Any]]:
        """Get random stocks from liquid list as fallback"""
        liquid_stocks = self._get_liquid_stocks_list()
        liquid_stocks = [
            s for s in liquid_stocks 
            if s not in exclude_winners and s not in exclude_existing
        ]
        
        candidates = []
        
        for symbol in liquid_stocks[:limit * 2]:
            if len(candidates) >= limit:
                break
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d', interval='1d')
                
                if hist.empty or len(hist) < 2:
                    continue
                
                latest = hist.iloc[-1]
                previous = hist.iloc[-2]
                
                close = latest['Close']
                prev_close = previous['Close']
                volume = latest['Volume']
                
                change_pct = ((close - prev_close) / prev_close) * 100
                
                # Exclude big gainers (>20%)
                if change_pct > 20:
                    continue
                
                if close >= self.min_price and volume >= self.min_volume:
                    candidates.append({
                        'symbol': symbol,
                        'exchange': 'NASDAQ',
                        'price': float(close),
                        'change_pct': float(change_pct),
                        'volume': int(volume),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'open': float(latest['Open']),
                        'close': float(close)
                    })
                
            except Exception as e:
                continue
            
            time.sleep(0.1)
        
        return candidates
    
    def _is_excluded_symbol(self, symbol: str, exchange: str) -> bool:
        """Check if symbol should be excluded"""
        if exchange == 'OTC':
            return True
        
        symbol_upper = symbol.upper()
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern in symbol_upper:
                return True
        
        if len(symbol) > 5:
            return True
        
        return False
    
    def _get_liquid_stocks_list(self) -> List[str]:
        """Get list of liquid stocks for sampling"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'JNJ',
            'V', 'PG', 'JPM', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'COST',
            'AVGO', 'KO', 'LLY', 'ADBE', 'WMT', 'MCD', 'CSCO', 'ACN', 'TMO', 'DIS',
            'ABT', 'DHR', 'VZ', 'NKE', 'NFLX', 'CRM', 'CMCSA', 'TXN', 'INTC', 'ORCL',
            'AMD', 'QCOM', 'HON', 'UNP', 'PM', 'NEE', 'RTX', 'UPS', 'LOW', 'INTU',
            'IBM', 'BA', 'AMGN', 'SPGI', 'GS', 'BLK', 'CAT', 'ELV', 'SBUX', 'DE',
            'AXP', 'ISRG', 'BKNG', 'GILD', 'ADI', 'TJX', 'MMC', 'MDLZ', 'VRTX', 'ADP',
            'CI', 'SYK', 'REGN', 'ZTS', 'PLD', 'AMT', 'DUK', 'SO', 'PGR', 'BDX',
            'MO', 'TGT', 'CL', 'USB', 'BMY', 'SCHW', 'CVS', 'CB', 'BSX', 'LRCX',
            'SLB', 'EOG', 'ITW', 'NOC', 'EQIX', 'MMM', 'C', 'PNC', 'EMR', 'AMAT',
            'F', 'GM', 'T', 'BAC', 'WFC', 'MS', 'AIG', 'GE', 'XOM', 'CHTR',
            'DAL', 'AAL', 'UAL', 'CCL', 'NCLH', 'MAR', 'HLT', 'MGM', 'WYNN', 'LVS'
        ]
