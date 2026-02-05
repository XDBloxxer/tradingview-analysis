"""
Daily Winners Detector - Uses yfinance day_gainers screener
Gets ACTUAL top daily gainers dynamically from market
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import yfinance as yf

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top daily winners using yfinance day_gainers screener
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
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        
        self.logger.info(
            f"Daily Winners detector initialized (using yfinance day_gainers): "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_top_winners(self, top_n: int = 10, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners using yfinance day_gainers screener
        
        Args:
            top_n: Number of top winners to return
            target_date: Date to detect winners for (defaults to today)
            
        Returns:
            List of winner dictionaries with symbol, price, change, volume
        """
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        # Note: yfinance screeners only work for current day
        is_today = target_date.date() == datetime.now().date()
        
        if not is_today:
            self.logger.warning(
                f"yfinance day_gainers only works for current day. "
                f"Requested date {target_date_str} is not today. "
                f"Will attempt to get current day's gainers instead."
            )
        
        self.logger.info(f"Fetching ACTUAL day gainers from yfinance screener...")
        
        try:
            # Get day gainers screener
            screener = yf.Screener()
            
            # Fetch day gainers (this is the actual screener that gets real-time top gainers)
            result = screener.get_screeners(['day_gainers'], count=100)
            
            if not result or 'day_gainers' not in result:
                self.logger.error("Failed to get day_gainers from yfinance")
                return []
            
            quotes = result['day_gainers'].get('quotes', [])
            
            if not quotes:
                self.logger.warning("No quotes returned from day_gainers screener")
                return []
            
            self.logger.info(f"Got {len(quotes)} stocks from day_gainers screener")
            
            # Process and filter results
            all_candidates = []
            
            for quote in quotes:
                try:
                    symbol = quote.get('symbol', '')
                    
                    # Get exchange
                    exchange = quote.get('exchange', 'NASDAQ')
                    
                    # Map yfinance exchanges to our format
                    if exchange in ['NMS', 'NGM', 'NCM']:
                        exchange = 'NASDAQ'
                    elif exchange == 'NYQ':
                        exchange = 'NYSE'
                    elif exchange in ['NYE', 'ASE']:
                        exchange = 'AMEX'
                    
                    # Skip OTC and foreign exchanges
                    if exchange not in ['NASDAQ', 'NYSE', 'AMEX']:
                        self.logger.debug(f"Skipping {symbol} - exchange {exchange} not supported")
                        continue
                    
                    price = quote.get('regularMarketPrice', 0)
                    change_pct = quote.get('regularMarketChangePercent', 0)
                    volume = quote.get('regularMarketVolume', 0)
                    
                    # Apply filters
                    if price >= self.min_price and volume >= self.min_volume and change_pct > 0:
                        all_candidates.append({
                            'symbol': symbol,
                            'exchange': exchange,
                            'price': float(price),
                            'change_pct': float(change_pct),
                            'volume': int(volume)
                        })
                        self.logger.debug(f"Added {symbol} ({exchange}): +{change_pct:.2f}%")
                    
                except Exception as e:
                    self.logger.debug(f"Error processing quote: {e}")
                    continue
            
            if not all_candidates:
                self.logger.warning("No winners found after filtering")
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
            
        except Exception as e:
            self.logger.error(f"Error fetching day gainers: {e}", exc_info=True)
            return []
