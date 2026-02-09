"""
Daily Winners Detector - Uses TradingView MarketMovers API
Gets ACTUAL top daily gainers dynamically from market
FIXED: Excludes OTC stocks and properly handles change field (basis points)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

try:
    from tradingview_scraper.symbols.market_movers import MarketMovers
except ImportError:
    raise ImportError(
        "tradingview-scraper is required. Install with: pip install tradingview-scraper"
    )

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top daily winners using TradingView MarketMovers API
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
            f"Daily Winners detector initialized (TradingView MarketMovers): "
            f"min_price={self.min_price}, min_volume={self.min_volume}, "
            f"excludes OTC stocks"
        )
    
    def detect_top_winners(self, top_n: int = 15, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners using TradingView MarketMovers API
        
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
        
        try:
            # Fetch more than we need to allow for filtering out OTC
            fetch_limit = max(top_n * 10, 500)  # Fetch lots since we'll filter OTC
            
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
                    
                    # Get change - CRITICAL: TradingView returns basis points * 100
                    # So 10% gain shows as 1000, not 10
                    # We need to divide by 100 to get actual percentage
                    change_raw = float(item.get('change', 0))
                    change_pct = change_raw / 100.0  # Convert to actual percentage
                    
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
                    
                    # Stock passed all filters
                    all_candidates.append({
                        'symbol': symbol.strip().upper(),
                        'exchange': exchange,
                        'price': float(price),
                        'change_pct': float(change_pct),
                        'volume': int(volume),
                        'name': item.get('description', ''),
                        'market_cap': item.get('market_cap_basic', 0)
                    })
                    
                    self.logger.debug(
                        f"✓ Added {exchange}:{symbol}: +{change_pct:.2f}% @ ${price:.2f}, vol={volume:,}"
                    )
                    
                except Exception as e:
                    filtered_counts['parse_error'] += 1
                    self.logger.debug(f"Error processing item: {e}")
                    continue
            
            # Log filtering statistics
            self.logger.info(f"Filtering results:")
            self.logger.info(f"  - Total fetched: {len(data)}")
            self.logger.info(f"  - OTC excluded: {filtered_counts['otc_excluded']}")
            self.logger.info(f"  - Low price (< ${self.min_price}): {filtered_counts['low_price']}")
            self.logger.info(f"  - Low volume (< {self.min_volume:,}): {filtered_counts['low_volume']}")
            self.logger.info(f"  - No/negative change: {filtered_counts['no_change']}")
            self.logger.info(f"  - Parse errors: {filtered_counts['parse_error']}")
            self.logger.info(f"  - ✅ Passed all filters: {len(all_candidates)}")
            
            if not all_candidates:
                self.logger.warning("⚠️ No winners found after filtering!")
                return []
            
            # Convert to DataFrame and sort by change percentage
            df_results = pd.DataFrame(all_candidates)
            df_results = df_results.sort_values('change_pct', ascending=False)
            
            # Take top N
            top_winners = df_results.head(top_n).to_dict('records')
            
            # Add detection date and time, remove unnecessary fields
            for winner in top_winners:
                winner['detection_date'] = target_date_str
                winner['detection_time'] = '16:00:00'
                winner.pop('name', None)
                winner.pop('market_cap', None)
            
            self.logger.info(f"✅ Found top {len(top_winners)} daily winners (OTC excluded):")
            for i, winner in enumerate(top_winners, 1):
                self.logger.info(
                    f"  #{i}: {winner['exchange']}:{winner['symbol']} "
                    f"(+{winner['change_pct']:.2f}%) @ ${winner['price']:.2f}, "
                    f"vol={winner['volume']:,}"
                )
            
            return top_winners
            
        except Exception as e:
            self.logger.error(f"Error fetching day gainers from TradingView: {e}", exc_info=True)
            return []
