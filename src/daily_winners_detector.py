"""
Daily Winners Detector - Uses TradingView MarketMovers API with yfinance fallback
Gets ACTUAL top daily gainers dynamically from market
FIXED: Excludes OTC stocks, properly handles change field, validates data freshness, falls back to yfinance
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

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
        
        self.logger.info(
            f"Daily Winners detector initialized (TradingView + yfinance fallback): "
            f"min_price={self.min_price}, min_volume={self.min_volume}, "
            f"excludes OTC stocks, validates data freshness"
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
        
        # Sort by change percentage and take top N
        df_results = pd.DataFrame(candidates)
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
                'otc_excluded': 0,
                'low_price': 0,
                'low_volume': 0,
                'no_change': 0,
                'stale_data': 0,
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
                    
                    # EXCLUDE OTC STOCKS ENTIRELY
                    if exchange_prefix == 'OTC':
                        filtered_counts['otc_excluded'] += 1
                        self.logger.debug(f"Filtered OTC stock: {symbol}")
                        continue
                    
                    # Map TradingView exchange prefixes to standard names
                    exchange_map = {
                        'NASDAQ': 'NASDAQ',
                        'NYSE': 'NYSE',
                        'AMEX': 'AMEX',
                        'BATS': 'NASDAQ'
                    }
                    exchange = exchange_map.get(exchange_prefix, exchange_prefix)
                    
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
                    
                    # NEW: Verify data freshness
                    if not self._verify_fresh_data(symbol, target_date):
                        filtered_counts['stale_data'] += 1
                        self.logger.debug(f"Filtered {symbol}: stale data (last update not recent)")
                        continue
                    
                    # Stock passed all filters
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
            self.logger.info(f"  - OTC excluded: {filtered_counts['otc_excluded']}")
            self.logger.info(f"  - Low price (< ${self.min_price}): {filtered_counts['low_price']}")
            self.logger.info(f"  - Low volume (< {self.min_volume:,}): {filtered_counts['low_volume']}")
            self.logger.info(f"  - No/negative change: {filtered_counts['no_change']}")
            self.logger.info(f"  - Stale data: {filtered_counts['stale_data']}")
            self.logger.info(f"  - Parse errors: {filtered_counts['parse_error']}")
            self.logger.info(f"  - ✅ Passed all filters: {len(all_candidates)}")
            
            return all_candidates
            
        except Exception as e:
            self.logger.error(f"Error fetching from TradingView: {e}", exc_info=True)
            return []
    
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
            
            # Get gainers from yfinance screener
            # yfinance doesn't have a direct API for this, so we'll use a workaround
            # Fetch popular gainers using yfinance's Ticker.history
            
            # Try to get S&P 500 stocks and find today's gainers
            candidates = []
            
            # Download major indices to find gainers
            try:
                # Get a broader list of active stocks from major ETFs
                etf_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']  # Major ETFs
                all_stocks = set()
                
                for etf in etf_symbols:
                    try:
                        etf_ticker = yf.Ticker(etf)
                        # Try to get holdings (not always available)
                        # Fallback: we'll use a predefined list of liquid stocks
                        pass
                    except:
                        pass
                
                # Fallback: Use a predefined list of liquid stocks
                # In production, you'd want to maintain a more comprehensive list
                liquid_stocks = self._get_liquid_stocks_list()
                
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
                            threads=True
                        )
                        
                        # Process each stock in batch
                        for symbol in batch:
                            if symbol in exclude_symbols:
                                continue
                            
                            try:
                                if len(batch) == 1:
                                    stock_data = data
                                else:
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
                                
                                # Verify freshness
                                if not self._verify_fresh_data(symbol, target_date):
                                    continue
                                
                                # Add candidate
                                candidates.append({
                                    'symbol': symbol,
                                    'exchange': 'NASDAQ',  # Default, could be improved
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
                
            except Exception as e:
                self.logger.error(f"Error in yfinance scan: {e}")
            
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
        In production, you'd want to maintain this from a more comprehensive source
        """
        # This is a subset of liquid stocks - you may want to expand this
        # or fetch from a more comprehensive source
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
            # Add more as needed
        ]
    
    def _verify_fresh_data(self, symbol: str, target_date: datetime) -> bool:
        """
        Verify that a stock has fresh data from the target date
        Filters out delisted/halted stocks with stale data
        
        Args:
            symbol: Stock symbol
            target_date: Expected date of latest data
            
        Returns:
            True if data is fresh, False if stale
        """
        try:
            # Allow a grace period for market holidays
            # If target is today but it's a holiday, accept yesterday's data
            target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
            
            # Fetch recent 1-day data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')  # Get last 5 days to be safe
            
            if hist.empty:
                self.logger.debug(f"{symbol}: No recent data available")
                return False
            
            # Get the most recent trading date
            last_date = hist.index[-1].date()
            
            # Calculate days difference
            days_diff = (target_date_obj - last_date).days
            
            # Allow up to 3 days difference (accounts for weekends and holidays)
            if days_diff > 3:
                self.logger.debug(
                    f"{symbol}: Stale data (last update: {last_date}, target: {target_date_obj}, "
                    f"diff: {days_diff} days)"
                )
                return False
            
            # If it's Monday and last data is Friday, that's acceptable
            # If today and data is from today or yesterday, that's acceptable
            self.logger.debug(f"{symbol}: Fresh data (last update: {last_date})")
            return True
            
        except Exception as e:
            self.logger.debug(f"{symbol}: Error verifying freshness: {e}")
            # On error, be conservative and reject
            return False
