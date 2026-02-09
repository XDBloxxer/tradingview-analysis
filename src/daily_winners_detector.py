"""
Daily Winners Detector - Uses TradingView MarketMovers API
Gets ACTUAL top daily gainers dynamically from market
FIXED: Proper field names and realistic filtering thresholds
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
        
        # Stock universe filters - REALISTIC values for daily winners
        self.min_price = detection_config.get("min_price", 0.50)
        self.min_volume = detection_config.get("min_volume", 100000)
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        
        # Initialize MarketMovers
        self.market_movers = MarketMovers(export_result=False)
        
        self.logger.info(
            f"Daily Winners detector initialized (TradingView MarketMovers): "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_top_winners(self, top_n: int = 10, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners using TradingView MarketMovers API
        
        Args:
            top_n: Number of top winners to return
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
        
        self.logger.info(f"Fetching ACTUAL day gainers from TradingView MarketMovers...")
        
        try:
            # Fetch more than we need to allow for filtering
            fetch_limit = max(top_n * 3, 50)
            
            # Get gainers from TradingView - SIMPLIFIED, let API handle fields
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
            
            # Debug: Log first item to see actual structure
            if data:
                self.logger.info(f"Sample data structure: {list(data[0].keys())}")
                self.logger.info(f"Sample item: {data[0]}")
            
            # Process and filter results
            all_candidates = []
            
            for item in data:
                try:
                    # Extract symbol from format "NASDAQ:AAPL" -> "AAPL"
                    symbol_full = item.get('symbol', '')
                    if ':' in symbol_full:
                        exchange_prefix, symbol = symbol_full.split(':', 1)
                        # Map TradingView exchange prefixes to standard names
                        exchange_map = {
                            'NASDAQ': 'NASDAQ',
                            'NYSE': 'NYSE',
                            'AMEX': 'AMEX',
                            'OTC': 'OTC',
                            'BATS': 'NASDAQ'  # BATS often used for NASDAQ stocks
                        }
                        exchange = exchange_map.get(exchange_prefix, exchange_prefix)
                    else:
                        symbol = symbol_full
                        exchange = 'NASDAQ'  # Default
                    
                    if not symbol:
                        continue
                    
                    # Get price and change - check multiple possible field names
                    price = item.get('close') or item.get('price') or item.get('last') or 0
                    
                    # Try multiple field names for change percentage
                    change_pct = (
                        item.get('change') or 
                        item.get('change_percentage') or 
                        item.get('change_percent') or 
                        item.get('perf') or
                        0
                    )
                    
                    volume = item.get('volume') or item.get('Volume') or 0
                    
                    # Convert to proper types
                    try:
                        price = float(price)
                        change_pct = float(change_pct)
                        volume = int(volume)
                    except (ValueError, TypeError):
                        self.logger.debug(f"Type conversion failed for {symbol}")
                        continue
                    
                    # Apply REALISTIC filters - ANY positive gainer above min requirements
                    if price >= self.min_price and volume >= self.min_volume and change_pct > 0:
                        all_candidates.append({
                            'symbol': symbol.strip().upper(),
                            'exchange': exchange,
                            'price': float(price),
                            'change_pct': float(change_pct),
                            'volume': int(volume),
                            'name': item.get('name', ''),
                            'market_cap': item.get('market_cap_basic', 0)
                        })
                        self.logger.debug(f"Added {symbol}: +{change_pct:.2f}% @ ${price:.2f}, vol={volume:,}")
                    else:
                        self.logger.debug(
                            f"Filtered out {symbol}: price=${price:.2f}, "
                            f"volume={volume:,}, change={change_pct:.2f}%"
                        )
                    
                except Exception as e:
                    self.logger.debug(f"Error processing item: {e}")
                    continue
            
            if not all_candidates:
                self.logger.warning("No winners found after filtering")
                self.logger.warning(f"Original data count: {len(data)}")
                if data:
                    self.logger.warning(f"Sample raw item for debugging: {data[0]}")
                return []
            
            # Convert to DataFrame and sort
            df_results = pd.DataFrame(all_candidates)
            df_results = df_results.sort_values('change_pct', ascending=False)
            
            # Take top N
            top_winners = df_results.head(top_n).to_dict('records')
            
            # Add detection date and time
            for winner in top_winners:
                winner['detection_date'] = target_date_str
                winner['detection_time'] = '16:00:00'
                # Remove fields we don't want to store
                winner.pop('name', None)
                winner.pop('market_cap', None)
            
            self.logger.info(f"✓ Found top {len(top_winners)} ACTUAL daily winners:")
            if top_winners:
                for i, winner in enumerate(top_winners[:5], 1):
                    self.logger.info(f"  #{i}: {winner['symbol']} (+{winner['change_pct']:.2f}%) @ ${winner['price']:.2f}")
                if len(top_winners) > 5:
                    self.logger.info(f"  ... and {len(top_winners) - 5} more")
            
            return top_winners
            
        except Exception as e:
            self.logger.error(f"Error fetching day gainers from TradingView: {e}", exc_info=True)
            return []
