"""
Daily Winners Detector - Finds top 10 performing stocks end of day (4pm NYC)
Completely separate from the spike/grinder detection system
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from tradingview_scraper.symbols.screener import Screener
from tqdm import tqdm

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top 10 daily winners at market close (4pm NYC)
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
        
        # Initialize Screener
        self.screener = Screener()
        
        self.logger.info(
            f"Daily Winners detector initialized: "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_top_winners(self, top_n: int = 10, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners at market close
        
        Args:
            top_n: Number of top winners to return
            target_date: Date to detect winners for (defaults to today)
            
        Returns:
            List of winner dictionaries with symbol, price, change, volume
        """
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        self.logger.info(f"Detecting top {top_n} winners for {target_date_str}...")
        
        all_winners = []
        
        for exchange in self.exchanges:
            self.logger.info(f"Scanning {exchange}...")
            
            try:
                # Get all symbols with price/volume data
                symbols_data = self._get_symbols_for_exchange(exchange)
                
                if not symbols_data:
                    self.logger.warning(f"No data retrieved for {exchange}")
                    continue
                
                self.logger.info(f"Retrieved {len(symbols_data)} symbols from {exchange}")
                
                # Filter by minimum requirements
                filtered_symbols = self._filter_symbols(symbols_data)
                self.logger.info(f"Filtered to {len(filtered_symbols)} symbols meeting criteria")
                
                # Add to all winners
                all_winners.extend(filtered_symbols)
                
            except Exception as e:
                self.logger.error(f"Error scanning {exchange}: {str(e)}", exc_info=True)
                continue
        
        if not all_winners:
            self.logger.warning("No winners found")
            return []
        
        # Convert to DataFrame for easier sorting
        df = pd.DataFrame(all_winners)
        
        # Sort by change percentage descending
        df = df.sort_values('change_pct', ascending=False)
        
        # Take top N
        top_winners = df.head(top_n).to_dict('records')
        
        # Add detection date and time
        for winner in top_winners:
            winner['detection_date'] = target_date_str
            winner['detection_time'] = '16:00:00'  # 4pm NYC
        
        self.logger.info(f"Found top {len(top_winners)} winners")
        if top_winners:
            self.logger.info(f"Top winner: {top_winners[0]['symbol']} (+{top_winners[0]['change_pct']:.2f}%)")
        
        return top_winners
    
    def _get_symbols_for_exchange(self, exchange: str) -> List[Dict]:
        """
        Get symbols with current data for an exchange using TradingView Screener
        
        Args:
            exchange: Exchange name (NASDAQ, NYSE, AMEX)
            
        Returns:
            List of symbol data dictionaries with indicators
        """
        try:
            self.rate_limiter.wait()
            
            # Map exchange to market format for screener
            if exchange in ["NASDAQ", "NYSE", "AMEX", "NYSEAMERICAN"]:
                market = "america"
            else:
                market = exchange.lower()
            
            self.logger.debug(f"Screening {market} market (exchange: {exchange})...")
            
            # Get stocks sorted by performance - focusing on actual daily gainers
            results = self.screener.screen(
                market=market,
                filters=[
                    {'left': 'close', 'operation': 'greater', 'right': self.min_price},
                    {'left': 'volume', 'operation': 'greater', 'right': self.min_volume},
                    {'left': 'change', 'operation': 'greater', 'right': 1.0},  # At least 1% gain to be a "winner"
                ],
                limit=500,  # Get top 500 to have a good pool
                sort_by='change',  # Sort by percentage change
                sort_order='desc'
            )
            
            # Check status
            if not results or results.get('status') != 'success':
                self.logger.warning(f"Screener failed for {exchange}: {results.get('status', 'unknown')}")
                return []
            
            # Get data
            data = results.get('data', [])
            
            if not data:
                self.logger.warning(f"No data returned for {exchange}")
                return []
            
            # Filter by exchange if we got mixed results
            filtered_data = []
            for item in data:
                symbol_full = item.get('symbol', '')
                
                # Extract exchange from symbol
                if ':' in symbol_full:
                    item_exchange, item_symbol = symbol_full.split(':', 1)
                    
                    # Check if this matches our target exchange
                    if item_exchange.upper() == exchange.upper():
                        item['clean_symbol'] = item_symbol
                        item['exchange'] = exchange
                        filtered_data.append(item)
                else:
                    item['clean_symbol'] = symbol_full
                    item['exchange'] = exchange
                    filtered_data.append(item)
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error screening {exchange}: {str(e)}", exc_info=True)
            return []
    
    def _filter_symbols(self, symbols_data: List[Dict]) -> List[Dict]:
        """
        Filter symbols by minimum price and volume requirements
        Convert to standardized format
        
        Args:
            symbols_data: List of symbol data dictionaries
            
        Returns:
            Filtered and standardized list
        """
        filtered = []
        
        for symbol_data in symbols_data:
            try:
                # Get symbol - use clean_symbol if available
                symbol = symbol_data.get('clean_symbol', symbol_data.get('symbol', symbol_data.get('name', '')))
                if ':' in symbol:
                    _, symbol = symbol.split(':', 1)
                
                price = symbol_data.get('close', 0)
                change_pct = symbol_data.get('change', symbol_data.get('change_abs', 0))
                volume = symbol_data.get('volume', 0)
                exchange = symbol_data.get('exchange', 'NASDAQ')
                
                # Apply filters
                if price >= self.min_price and volume >= self.min_volume and change_pct > 0:
                    filtered.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'price': price,
                        'change_pct': change_pct,
                        'volume': volume
                    })
                    
            except (KeyError, TypeError, ValueError) as e:
                # Skip symbols with invalid data
                continue
        
        return filtered
