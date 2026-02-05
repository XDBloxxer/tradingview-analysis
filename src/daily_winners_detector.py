"""
Daily Winners Detector - Scrapes Yahoo Finance day gainers page
Gets ACTUAL top daily gainers dynamically from market
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top daily winners by scraping Yahoo Finance day gainers
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
            f"Daily Winners detector initialized (scraping Yahoo Finance): "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_top_winners(self, top_n: int = 10, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners by scraping Yahoo Finance day gainers
        
        Args:
            top_n: Number of top winners to return
            target_date: Date to detect winners for (defaults to today)
            
        Returns:
            List of winner dictionaries with symbol, price, change, volume
        """
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        # Note: Yahoo Finance screeners only work for current day
        is_today = target_date.date() == datetime.now().date()
        
        if not is_today:
            self.logger.warning(
                f"Yahoo Finance day gainers only shows current day. "
                f"Requested date {target_date_str} is not today. "
                f"Will fetch current day's gainers instead."
            )
        
        self.logger.info(f"Fetching ACTUAL day gainers from Yahoo Finance...")
        
        try:
            # Scrape Yahoo Finance day gainers page
            url = "https://finance.yahoo.com/screener/predefined/day_gainers"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table with gainers data
            # Yahoo Finance uses different structure, try to find the data
            table = soup.find('table')
            
            if not table:
                self.logger.error("Could not find data table on Yahoo Finance page")
                return []
            
            # Parse table into DataFrame using StringIO to avoid warning
            from io import StringIO
            df = pd.read_html(StringIO(str(table)))[0]
            
            self.logger.info(f"Found {len(df)} stocks from Yahoo Finance day gainers")
            
            # Process and filter results
            all_candidates = []
            
            for idx, row in df.iterrows():
                try:
                    # Extract data from row
                    # Column names vary, try different possibilities
                    symbol_raw = row.get('Symbol', row.get('Ticker', ''))
                    
                    if not symbol_raw:
                        continue
                    
                    # Clean up symbol - sometimes it has extra characters or spaces
                    # If it's like "S SLAB", take the part after the space
                    if isinstance(symbol_raw, str):
                        symbol = symbol_raw.strip()
                        # If there's a space, take the second part (the actual symbol)
                        if ' ' in symbol:
                            parts = symbol.split()
                            symbol = parts[-1]  # Take the last part which is the actual symbol
                    else:
                        symbol = str(symbol_raw).strip()
                    
                    if not symbol:
                        continue
                    
                    # Get price and change
                    price_str = row.get('Price (Intraday)', row.get('Price', '0'))
                    change_str = row.get('% Change', row.get('Change %', '0'))
                    volume_str = row.get('Volume', row.get('Avg Vol (3 month)', '0'))
                    
                    # Clean up and convert price (remove commas, dollar signs)
                    if isinstance(price_str, str):
                        price = float(price_str.replace('$', '').replace(',', ''))
                    else:
                        price = float(price_str)
                    
                    # Clean up change percentage (remove % sign)
                    if isinstance(change_str, str):
                        change_pct = float(change_str.replace('%', '').replace('+', ''))
                    else:
                        change_pct = float(change_str)
                    
                    # Clean up volume (remove commas, handle M/B suffixes)
                    if isinstance(volume_str, str):
                        volume_str = volume_str.replace(',', '')
                        if 'M' in volume_str:
                            volume = float(volume_str.replace('M', '')) * 1000000
                        elif 'B' in volume_str:
                            volume = float(volume_str.replace('B', '')) * 1000000000
                        elif 'K' in volume_str:
                            volume = float(volume_str.replace('K', '')) * 1000
                        else:
                            volume = float(volume_str)
                    else:
                        volume = float(volume_str)
                    
                    # Determine exchange (assume NASDAQ by default for US stocks)
                    exchange = 'NASDAQ'
                    
                    # Apply filters
                    if price >= self.min_price and volume >= self.min_volume and change_pct > 0:
                        all_candidates.append({
                            'symbol': symbol,
                            'exchange': exchange,
                            'price': float(price),
                            'change_pct': float(change_pct),
                            'volume': int(volume)
                        })
                        self.logger.debug(f"Added {symbol}: +{change_pct:.2f}%")
                    
                except Exception as e:
                    self.logger.debug(f"Error processing row: {e}")
                    continue
            
            if not all_candidates:
                self.logger.warning("No winners found after filtering")
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
            
            self.logger.info(f"✓ Found top {len(top_winners)} ACTUAL daily winners:")
            if top_winners:
                for i, winner in enumerate(top_winners[:5], 1):
                    self.logger.info(f"  #{i}: {winner['symbol']} (+{winner['change_pct']:.2f}%) @ ${winner['price']:.2f}")
                if len(top_winners) > 5:
                    self.logger.info(f"  ... and {len(top_winners) - 5} more")
            
            return top_winners
            
        except Exception as e:
            self.logger.error(f"Error fetching day gainers: {e}", exc_info=True)
            return []
