"""
Daily Winners Detector - FIXED VERSION
Uses TradingView MarketMovers API with yfinance fallback
FIXES:
1. Smarter freshness validation that doesn't call yfinance for every symbol
2. Batch verification to reduce API calls
3. Better handling of current-day data from TradingView
4. Improved OTC filtering
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import time

try:
    from tradingview_scraper.symbols.market_movers import MarketMovers
except ImportError:
    raise ImportError(
        "tradingview-scraper is required. Install with: pip install tradingview-scraper"
    )

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top daily winners using TradingView MarketMovers API with yfinance fallback
    """
    
    # Known problematic exchanges/suffixes to exclude
    EXCLUDED_PATTERNS = [
        'OTC',  # Over-the-counter
        '.PK',  # Pink sheets
        '.OB',  # OTC Bulletin Board
        '-',    # Often delisted stocks
    ]
    
    def __init__(self, config: dict):
        """
        Initialize daily winners detector
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        detection_config = config.get("detection", {})
        
        # Stock universe filters
        self.min_price = detection_config.get("min_price", 0.25)
        self.min_volume = detection_config.get("min_volume", 10000)
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        
        # Initialize MarketMovers
        self.market_movers = MarketMovers(export_result=False)
        
        # Cache for freshness checks to avoid redundant API calls
        self.freshness_cache = {}
        
        self.logger.info(
            f"Daily Winners detector initialized: "
            f"min_price={self.min_price}, min_volume={self.min_volume}, "
            f"excludes OTC/delisted stocks with improved validation"
        )
    
    def detect_top_winners(self, top_n: int = 15, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners using TradingView MarketMovers API with yfinance fallback
        
        Args:
            top_n: Number of top winners to return (default 15)
            target_date: Date to detect winners for (defaults to today)
            
        Returns:
            List of winner dictionaries with symbol, price, change, volume
        """
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        # Note: TradingView MarketMovers shows current day data
        is_today = target_date.date() == datetime.now().date()
        
        if not is_today:
            self.logger.warning(
                f"TradingView MarketMovers only shows current day data. "
                f"Requested date {target_date_str} is not today. "
                f"Will fetch current day's gainers instead."
            )
        
        self.logger.info(f"Fetching top {top_n} day gainers from TradingView MarketMovers...")
        
        # Try TradingView first
        candidates = self._fetch_from_tradingview(target_date, top_n)
        
        # If we don't have enough, supplement with yfinance
        if len(candidates) < top_n:
            self.logger.warning(
                f"Only found {len(candidates)} valid winners from TradingView. "
                f"Fetching additional candidates from yfinance..."
            )
            
            # Get symbols we already have to avoid duplicates
            existing_symbols = {c['symbol'] for c in candidates}
            
            # Fetch from yfinance
            yf_candidates = self._fetch_from_yfinance(target_date, top_n * 2, existing_symbols)
            
            # Combine and sort
            candidates.extend(yf_candidates)
            self.logger.info(
                f"Combined total: {len(candidates)} candidates "
                f"({len(candidates) - len(yf_candidates)} from TradingView, "
                f"{len(yf_candidates)} from yfinance)"
            )
        
        if not candidates:
            self.logger.warning("⚠️ No winners found from either source!")
            return []
        
        # FINAL BATCH VALIDATION: Verify freshness for top candidates only
        self.logger.info(f"Performing final batch freshness validation on top {min(len(candidates), top_n * 2)} candidates...")
        validated_candidates = self._batch_verify_freshness(candidates[:top_n * 2], target_date)
        
        if not validated_candidates:
            self.logger.warning("⚠️ All candidates failed freshness validation!")
            return []
        
        # Sort by change percentage and take top N
        df_results = pd.DataFrame(validated_candidates)
        df_results = df_results.sort_values('change_pct', ascending=False)
        top_winners = df_results.head(top_n).to_dict('records')
        
        # Add detection date and time
        for winner in top_winners:
            winner['detection_date'] = target_date_str
            winner['detection_time'] = '16:00:00'
        
        self.logger.info(f"✅ Found top {len(top_winners)} daily winners:")
        for i, winner in enumerate(top_winners, 1):
            source = winner.pop('source', 'unknown')
            self.logger.info(
                f"  #{i}: {winner['exchange']}:{winner['symbol']} "
                f"(+{winner['change_pct']:.2f}%) @ ${winner['price']:.2f}, "
                f"vol={winner['volume']:,} [{source}]"
            )
        
        return top_winners
    
    def _is_excluded_symbol(self, symbol: str, exchange: str) -> bool:
        """
        Check if a symbol should be excluded based on pattern matching
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name
            
        Returns:
            True if symbol should be excluded
        """
        # Check exchange
        if exchange == 'OTC':
            return True
        
        # Check symbol patterns
        symbol_upper = symbol.upper()
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern in symbol_upper:
                return True
        
        # Check for unusual symbol patterns that indicate delisted/problematic stocks
        # Examples: symbols with too many letters, unusual characters
        if len(symbol) > 5:  # Most valid symbols are 1-5 characters
            return True
        
        return False
    
    def _fetch_from_tradingview(self, target_date: datetime, top_n: int) -> List[Dict[str, Any]]:
        """
        Fetch candidates from TradingView MarketMovers API
        
        Args:
            target_date: Target date for validation
            top_n: Number of winners needed
            
        Returns:
            List of valid candidate dictionaries
        """
        try:
            # Fetch more than we need to allow for filtering
            fetch_limit = max(top_n * 10, 500)
            
            # Get gainers from TradingView
            result = self.market_movers.scrape(
                market='stocks-usa',
                category='gainers',
                limit=fetch_limit
            )
            
            if result['status'] != 'success':
                self.logger.error(f"TradingView API returned non-success status: {result.get('status')}")
                return []
            
            data = result.get('data', [])
            
            if not data:
                self.logger.warning("No data returned from TradingView MarketMovers")
                return []
            
            self.logger.info(f"Received {len(data)} gainers from TradingView")
            
            # Process and filter results
            all_candidates = []
            filtered_counts = {
                'excluded_pattern': 0,
                'low_price': 0,
                'low_volume': 0,
                'no_change': 0,
                'parse_error': 0
            }
            
            for item in data:
                try:
                    # Extract symbol from format "NASDAQ:AAPL" -> "AAPL"
                    symbol_full = item.get('symbol', '')
                    if ':' in symbol_full:
                        exchange_prefix, symbol = symbol_full.split(':', 1)
                    else:
                        symbol = symbol_full
                        exchange_prefix = 'NASDAQ'  # Default
                    
                    if not symbol:
                        continue
                    
                    # Map TradingView exchange prefixes to standard names
                    exchange_map = {
                        'NASDAQ': 'NASDAQ',
                        'NYSE': 'NYSE',
                        'AMEX': 'AMEX',
                        'BATS': 'NASDAQ'
                    }
                    exchange = exchange_map.get(exchange_prefix, exchange_prefix)
                    
                    # Check for excluded patterns (OTC, delisted, etc.)
                    if self._is_excluded_symbol(symbol, exchange):
                        filtered_counts['excluded_pattern'] += 1
                        self.logger.debug(f"Filtered excluded symbol: {exchange}:{symbol}")
                        continue
                    
                    # Get price
                    price = float(item.get('close', 0))
                    
                    # Get change percentage - this is already a percentage value
                    change_pct = float(item.get('change', 0))
                    
                    # Get volume
                    volume = int(item.get('volume', 0))
                    
                    # Apply filters
                    if price < self.min_price:
                        filtered_counts['low_price'] += 1
                        self.logger.debug(f"Filtered {symbol}: price ${price:.2f} < ${self.min_price}")
                        continue
                    
                    if volume < self.min_volume:
                        filtered_counts['low_volume'] += 1
                        self.logger.debug(f"Filtered {symbol}: volume {volume:,} < {self.min_volume:,}")
                        continue
                    
                    if change_pct <= 0:
                        filtered_counts['no_change'] += 1
                        self.logger.debug(f"Filtered {symbol}: change {change_pct:.2f}% <= 0")
                        continue
                    
                    # Stock passed all filters (freshness check will be done in batch later)
                    all_candidates.append({
                        'symbol': symbol.strip().upper(),
                        'exchange': exchange,
                        'price': float(price),
                        'change_pct': float(change_pct),
                        'volume': int(volume),
                        'source': 'tradingview'
                    })
                    
                    self.logger.debug(
                        f"✓ Added {exchange}:{symbol}: +{change_pct:.2f}% @ ${price:.2f}, vol={volume:,}"
                    )
                    
                except Exception as e:
                    filtered_counts['parse_error'] += 1
                    self.logger.debug(f"Error processing item: {e}")
                    continue
            
            # Log filtering statistics
            self.logger.info(f"TradingView filtering results:")
            self.logger.info(f"  - Total fetched: {len(data)}")
            self.logger.info(f"  - Excluded patterns (OTC, delisted, etc.): {filtered_counts['excluded_pattern']}")
            self.logger.info(f"  - Low price (< ${self.min_price}): {filtered_counts['low_price']}")
            self.logger.info(f"  - Low volume (< {self.min_volume:,}): {filtered_counts['low_volume']}")
            self.logger.info(f"  - No/negative change: {filtered_counts['no_change']}")
            self.logger.info(f"  - Parse errors: {filtered_counts['parse_error']}")
            self.logger.info(f"  - ✅ Passed filters (before freshness check): {len(all_candidates)}")
            
            return all_candidates
            
        except Exception as e:
            self.logger.error(f"Error fetching from TradingView: {e}", exc_info=True)
            return []
    
    def _batch_verify_freshness(
        self, 
        candidates: List[Dict[str, Any]], 
        target_date: datetime,
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Verify freshness for a batch of candidates using yfinance
        More efficient than checking one by one
        
        Args:
            candidates: List of candidate dictionaries
            target_date: Target date for validation
            batch_size: Number of symbols to check at once
            
        Returns:
            List of candidates that passed freshness check
        """
        if not candidates:
            return []
        
        valid_candidates = []
        target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
        
        # Process in batches to reduce API calls
        symbols = [c['symbol'] for c in candidates]
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            batch_candidates = candidates[i:i+batch_size]
            
            self.logger.debug(f"Verifying freshness for batch {i//batch_size + 1}: {len(batch_symbols)} symbols")
            
            try:
                # Download batch data (last 5 days)
                data = yf.download(
                    batch_symbols,
                    period='5d',
                    interval='1d',
                    group_by='ticker',
                    progress=False,
                    threads=True,
                    show_errors=False  # Suppress yfinance errors
                )
                
                if data.empty:
                    self.logger.warning(f"No data returned for batch {i//batch_size + 1}")
                    continue
                
                # Check each symbol in the batch
                for idx, (symbol, candidate) in enumerate(zip(batch_symbols, batch_candidates)):
                    try:
                        # Handle both single and multi-ticker responses
                        if len(batch_symbols) == 1:
                            symbol_data = data
                        else:
                            if symbol not in data.columns.levels[0]:
                                self.logger.debug(f"{symbol}: No data in response")
                                continue
                            symbol_data = data[symbol]
                        
                        if symbol_data.empty or len(symbol_data) < 1:
                            self.logger.debug(f"{symbol}: Empty data")
                            continue
                        
                        # Get the most recent trading date
                        last_date = symbol_data.index[-1].date()
                        
                        # Calculate days difference
                        days_diff = (target_date_obj - last_date).days
                        
                        # Allow up to 5 days difference (accounts for weekends, holidays, and delays)
                        if days_diff > 5:
                            self.logger.debug(
                                f"{symbol}: Stale data (last update: {last_date}, target: {target_date_obj}, "
                                f"diff: {days_diff} days) - REJECTED"
                            )
                            continue
                        
                        # Verify volume is reasonable (not zero)
                        last_volume = symbol_data['Volume'].iloc[-1]
                        if last_volume < self.min_volume:
                            self.logger.debug(f"{symbol}: Last volume {last_volume:,} too low - REJECTED")
                            continue
                        
                        # Stock passed freshness check
                        self.logger.debug(f"{symbol}: Fresh data (last update: {last_date}) - PASSED")
                        valid_candidates.append(candidate)
                        
                    except Exception as e:
                        self.logger.debug(f"{symbol}: Error verifying freshness: {e}")
                        continue
                
                # Small delay between batches to be nice to the API
                if i + batch_size < len(symbols):
                    time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in batch freshness verification: {e}")
                continue
        
        self.logger.info(
            f"Freshness validation: {len(valid_candidates)}/{len(candidates)} candidates passed "
            f"({len(candidates) - len(valid_candidates)} rejected as stale/delisted)"
        )
        
        return valid_candidates
    
    def _fetch_from_yfinance(
        self, 
        target_date: datetime, 
        limit: int,
        exclude_symbols: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch candidates from yfinance as fallback
        
        Args:
            target_date: Target date for validation
            limit: Maximum number of candidates to fetch
            exclude_symbols: Set of symbols to exclude (already have from TradingView)
            
        Returns:
            List of valid candidate dictionaries
        """
        try:
            self.logger.info(f"Fetching gainers from yfinance (excluding {len(exclude_symbols)} existing symbols)...")
            
            candidates = []
            
            # Get a broader list of active stocks
            liquid_stocks = self._get_liquid_stocks_list()
            
            # Remove excluded symbols
            liquid_stocks = [s for s in liquid_stocks if s not in exclude_symbols]
            
            self.logger.info(f"Scanning {len(liquid_stocks)} liquid stocks for gainers...")
            
            # Fetch data for multiple stocks at once
            batch_size = 50
            for i in range(0, len(liquid_stocks), batch_size):
                batch = liquid_stocks[i:i+batch_size]
                
                try:
                    # Download batch data
                    data = yf.download(
                        batch,
                        period='2d',
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        threads=True,
                        show_errors=False
                    )
                    
                    # Process each stock in batch
                    for symbol in batch:
                        if symbol in exclude_symbols:
                            continue
                        
                        try:
                            if len(batch) == 1:
                                stock_data = data
                            else:
                                if symbol not in data.columns.levels[0]:
                                    continue
                                stock_data = data[symbol]
                            
                            if stock_data.empty or len(stock_data) < 2:
                                continue
                            
                            # Get latest and previous close
                            latest = stock_data.iloc[-1]
                            previous = stock_data.iloc[-2]
                            
                            close = latest['Close']
                            prev_close = previous['Close']
                            volume = latest['Volume']
                            
                            # Calculate change
                            change_pct = ((close - prev_close) / prev_close) * 100
                            
                            # Apply filters
                            if close < self.min_price:
                                continue
                            if volume < self.min_volume:
                                continue
                            if change_pct <= 0:
                                continue
                            
                            # Check for excluded patterns
                            if self._is_excluded_symbol(symbol, 'NASDAQ'):
                                continue
                            
                            # Add candidate (already validated by yfinance having recent data)
                            candidates.append({
                                'symbol': symbol,
                                'exchange': 'NASDAQ',  # Default
                                'price': float(close),
                                'change_pct': float(change_pct),
                                'volume': int(volume),
                                'source': 'yfinance'
                            })
                            
                            self.logger.debug(f"✓ Added from yfinance: {symbol} (+{change_pct:.2f}%)")
                            
                        except Exception as e:
                            self.logger.debug(f"Error processing {symbol} from yfinance: {e}")
                            continue
                
                except Exception as e:
                    self.logger.debug(f"Error downloading batch: {e}")
                    continue
                
                # Stop if we have enough
                if len(candidates) >= limit:
                    break
                
                # Small delay between batches
                time.sleep(0.5)
            
            # Sort by change percentage
            if candidates:
                candidates = sorted(candidates, key=lambda x: x['change_pct'], reverse=True)
                candidates = candidates[:limit]
            
            self.logger.info(f"✅ Found {len(candidates)} valid gainers from yfinance")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error fetching from yfinance: {e}", exc_info=True)
            return []
    
    def _get_liquid_stocks_list(self) -> List[str]:
        """
        Get a list of liquid stocks to scan
        """
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
        ]
