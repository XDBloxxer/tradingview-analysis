"""
Event detection for identifying Spikers and Grinders
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from tradingview_scraper.symbols.screener import Screener
from tqdm import tqdm

from .rate_limiter import RateLimiter
from .sheets_writer import SheetsWriter
from .utils import get_indicator_mapping


class EventDetector:
    """
    Detects stock price events (spikers and grinders) from TradingView data
    """
    
    def __init__(self, config: dict):
        """
        Initialize event detector
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        detection_config = config.get("detection", {})
        
        # Event thresholds
        self.spiker_threshold = detection_config.get("spiker_threshold", 15.0)
        self.grinder_days = detection_config.get("grinder_days", 3)
        self.grinder_threshold = detection_config.get("grinder_threshold", 0.5)
        
        # Stock universe filters
        self.min_price = detection_config.get("min_price", 0.50)
        self.min_volume = detection_config.get("min_volume", 100000)
        self.lookback_days = detection_config.get("lookback_days", 7)
        self.exchanges = detection_config.get("exchanges", ["NASDAQ", "NYSE", "AMEX"])
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        self.sheets_writer = SheetsWriter(config)
        
        # Indicator mapping
        self.indicator_mapping = get_indicator_mapping(config)
        
        # Initialize Screener
        self.screener = Screener()
        
        self.logger.info(
            f"Event detector initialized: "
            f"Spiker≥{self.spiker_threshold}%, "
            f"Grinder≥{self.grinder_threshold}% for {self.grinder_days} days"
        )
    
    def detect_events(self) -> List[Dict[str, Any]]:
        """
        Detect all events (spikers and grinders) from TradingView
        
        Returns:
            List of event dictionaries
        """
        self.logger.info("Starting event detection...")
        
        all_events = []
        
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
                
                # Detect events in filtered symbols
                events = self._detect_events_in_symbols(filtered_symbols, exchange)
                all_events.extend(events)
                
                self.logger.info(f"Found {len(events)} events in {exchange}")
                
            except Exception as e:
                self.logger.error(f"Error scanning {exchange}: {str(e)}", exc_info=True)
                continue
        
        self.logger.info(f"Total events detected: {len(all_events)}")
        return all_events
    
    def _get_symbols_for_exchange(self, exchange: str) -> List[Dict]:
        """
        Get symbols with current data for an exchange using TradingView Screener
        
        Uses the Screener class to scan for stocks matching our criteria.
        The screener returns up to 1000 symbols per request.
        
        Args:
            exchange: Exchange name (NASDAQ, NYSE, AMEX)
            
        Returns:
            List of symbol data dictionaries with indicators
        """
        try:
            self.rate_limiter.wait()
            
            # Map exchange to market format for screener
            # The screener uses 'america' for US markets
            if exchange in ["NASDAQ", "NYSE", "AMEX", "NYSEAMERICAN"]:
                market = "america"
            else:
                market = exchange.lower()
            
            self.logger.debug(f"Screening {market} market (exchange: {exchange})...")
            
            # Use the screener to get stocks
            # Get more stocks and lower the filter to catch more events
            results = self.screener.screen(
                market=market,
                filters=[
                    {'left': 'close', 'operation': 'greater', 'right': self.min_price},
                    {'left': 'volume', 'operation': 'greater', 'right': self.min_volume},
                    # Also filter for stocks with some movement
                    {'left': 'change_abs', 'operation': 'nempty'}
                ],
                limit=5000,  # Increased from 1000 to get more stocks, Stock limit
                sort_by='change_abs',  # Sort by absolute change to catch movers
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
            # Symbols come in format "NASDAQ:AAPL"
            filtered_data = []
            for item in data:
                symbol_full = item.get('symbol', '')
                
                # Extract exchange from symbol
                if ':' in symbol_full:
                    item_exchange, item_symbol = symbol_full.split(':', 1)
                    
                    # Check if this matches our target exchange
                    if item_exchange.upper() == exchange.upper():
                        # Add clean symbol name
                        item['clean_symbol'] = item_symbol
                        item['exchange'] = exchange
                        filtered_data.append(item)
                else:
                    # No exchange prefix, include it
                    item['clean_symbol'] = symbol_full
                    item['exchange'] = exchange
                    filtered_data.append(item)
            
            self.logger.info(f"Retrieved {len(filtered_data)} symbols from {exchange}")
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error screening {exchange}: {str(e)}", exc_info=True)
            return []
    
    def _filter_symbols(self, symbols_data: List[Dict]) -> List[Dict]:
        """
        Filter symbols by minimum price and volume requirements
        
        Args:
            symbols_data: List of symbol data dictionaries
            
        Returns:
            Filtered list
        """
        filtered = []
        
        for symbol_data in symbols_data:
            try:
                price = symbol_data.get('close', 0)
                volume = symbol_data.get('volume', 0)
                
                # Apply filters
                if price >= self.min_price and volume >= self.min_volume:
                    filtered.append(symbol_data)
                    
            except (KeyError, TypeError, ValueError) as e:
                # Skip symbols with invalid data
                continue
        
        return filtered
    
    def _detect_events_in_symbols(self, symbols_data: List[Dict], exchange: str) -> List[Dict[str, Any]]:
        """
        Detect spikers and grinders in symbol data
        
        Args:
            symbols_data: List of symbol data dictionaries
            exchange: Exchange name
            
        Returns:
            List of detected events
        """
        events = []
        today = datetime.now().date()
        
        for symbol_data in tqdm(symbols_data, desc=f"Detecting events in {exchange}"):
            try:
                # Get symbol - use clean_symbol if available (from screener), otherwise parse
                symbol = symbol_data.get('clean_symbol', symbol_data.get('symbol', symbol_data.get('name', '')))
                if ':' in symbol:
                    _, symbol = symbol.split(':', 1)
                
                price = symbol_data.get('close', 0)
                # Try different field names for change percentage
                change_pct = symbol_data.get('change', symbol_data.get('change_abs', 0))
                volume = symbol_data.get('volume', 0)
                
                # Check for Spiker (single-day move > threshold)
                if change_pct >= self.spiker_threshold:
                    events.append({
                        'Date': today.isoformat(),
                        'Symbol': symbol,
                        'Exchange': exchange,
                        'Event_Type': 'Spiker',
                        'Price': price,
                        'Change_%': change_pct,
                        'Volume': volume
                    })
                
                # Check for Grinder
                # Use relative volume and moderate gains as proxy
                # Try multiple field names for relative volume
                rel_volume = symbol_data.get('Rec.Vol.M', 
                             symbol_data.get('relative_volume_10d_calc',
                             symbol_data.get('Relative.Volume.10d.Calc', 1.0)))
                
                # Potential grinder: moderate positive gain with decent volume
                # Lower threshold to catch more grinders
                if (self.grinder_threshold <= change_pct < self.spiker_threshold 
                    and rel_volume >= 0.8):  # Lowered from 1.2 to catch more
                    
                    # Note: Proper grinder detection requires 3 consecutive days
                    # This is a simplified version using current data only
                    events.append({
                        'Date': today.isoformat(),
                        'Symbol': symbol,
                        'Exchange': exchange,
                        'Event_Type': 'Grinder',
                        'Price': price,
                        'Change_%': change_pct,
                        'Volume': volume
                    })
                
            except Exception as e:
                self.logger.debug(f"Error processing symbol {symbol_data}: {str(e)}")
                continue
        
        return events
    
    def write_to_sheets(self, events: List[Dict[str, Any]]):
        """
        Write detected events to Google Sheets
        
        Args:
            events: List of event dictionaries
        """
        self.sheets_writer.write_candidates(events)
