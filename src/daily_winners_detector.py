"""
Daily Winners Detector - Uses TradingView MarketMovers for ACTUAL daily gainers
Gets ACTUAL top daily gainers based on today's performance
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from tradingview_scraper.symbols.market_movers import MarketMovers

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top daily winners using TradingView MarketMovers
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
        self.min_price = detection_config.get("min_price", 0.50)
        self.min_volume = detection_config.get("min_volume", 100000)
        self.exchanges = detection_config.get("exchanges", ["NASDAQ", "NYSE", "AMEX"])
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        self.market_movers = MarketMovers()
        
        self.logger.info(
            f"Daily Winners detector initialized (using TradingView MarketMovers): "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_top_winners(self, top_n: int = 10, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners using TradingView MarketMovers
        
        Args:
            top_n: Number of top winners to return
            target_date: Date to detect winners for (defaults to today)
            
        Returns:
            List of winner dictionaries with symbol, price, change, volume
        """
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        self.logger.info(f"Detecting top {top_n} ACTUAL daily winners for {target_date_str}...")
        self.logger.info("Using TradingView MarketMovers gainers screener")
        
        all_candidates = []
        
        try:
            self.rate_limiter.wait()
            
            # Get actual day gainers from TradingView
            gainers_result = self.market_movers.scrape(
                market='stocks-usa',
                category='gainers',
                limit=100  # Get more to filter
            )
            
            if gainers_result and gainers_result.get('status') == 'success':
                gainers = gainers_result.get('data', [])
                
                self.logger.info(f"Got {len(gainers)} gainers from TradingView MarketMovers")
                
                for item in gainers:
                    try:
                        # Extract symbol and exchange
                        symbol_full = item.get('symbol', '')
                        if ':' in symbol_full:
                            item_exchange, symbol = symbol_full.split(':', 1)
                        else:
                            symbol = symbol_full
                            item_exchange = 'NASDAQ'
                        
                        # Map exchange names
                        exchange = item_exchange.upper()
                        if exchange not in self.exchanges:
                            continue
                        
                        price = item.get('close', 0)
                        change_pct = item.get('change', 0)
                        volume = item.get('volume', 0)
                        
                        # Apply filters
                        if price >= self.min_price and volume >= self.min_volume and change_pct > 0:
                            all_candidates.append({
                                'symbol': symbol,
                                'exchange': exchange,
                                'price': float(price),
                                'change_pct': float(change_pct),
                                'volume': int(volume)
                            })
                    except Exception as e:
                        self.logger.debug(f"Error processing gainer: {e}")
                        continue
            else:
                self.logger.warning(f"MarketMovers failed: {gainers_result.get('status', 'unknown')}")
        
        except Exception as e:
            self.logger.error(f"Error fetching from MarketMovers: {e}")
        
        if not all_candidates:
            self.logger.warning("No winners found")
            return []
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(all_candidates)
        df = df.sort_values('change_pct', ascending=False)
        
        # Take top N
        top_winners = df.head(top_n).to_dict('records')
        
        # Add detection date and time
        for winner in top_winners:
            winner['detection_date'] = target_date_str
            winner['detection_time'] = '16:00:00'
        
        self.logger.info(f"✓ Found top {len(top_winners)} ACTUAL daily winners for {target_date_str}:")
        if top_winners:
            for i, winner in enumerate(top_winners[:5], 1):
                self.logger.info(f"  #{i}: {winner['symbol']} (+{winner['change_pct']:.2f}%) @ ${winner['price']:.2f}")
            if len(top_winners) > 5:
                self.logger.info(f"  ... and {len(top_winners) - 5} more")
        
        return top_winners
