"""
Daily Winners Detector - FIXED VERSION using yfinance for reliable data
Gets ACTUAL top daily gainers based on today's performance
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top 10 daily winners using yfinance for reliable data
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
        
        self.logger.info(
            f"Daily Winners detector initialized (using yfinance): "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )
    
    def detect_top_winners(self, top_n: int = 10, target_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Detect top N daily winners using yfinance screener
        
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
        self.logger.info("Using yfinance to get reliable daily gainers")
        
        all_candidates = []
        
        # Get most active stocks using yfinance screener
        try:
            # Get day gainers from yfinance
            self.logger.info("Fetching day gainers from yfinance...")
            gainers = yf.Screener()
            
            # Get day gainers screener
            # This returns actual stocks with highest % change today
            gainers_data = gainers.get_screeners(['day_gainers'], count=100)
            
            if gainers_data and 'day_gainers' in gainers_data:
                quotes = gainers_data['day_gainers'].get('quotes', [])
                
                self.logger.info(f"Got {len(quotes)} gainers from yfinance")
                
                for quote in quotes:
                    try:
                        symbol = quote.get('symbol', '')
                        
                        # Get exchange - yfinance uses different format
                        exchange = quote.get('exchange', 'NASDAQ')
                        # Map yfinance exchanges to our format
                        if exchange in ['NMS', 'NGM', 'NCM']:
                            exchange = 'NASDAQ'
                        elif exchange == 'NYQ':
                            exchange = 'NYSE'
                        elif exchange == 'NYE':
                            exchange = 'AMEX'
                        
                        # Skip if not in our target exchanges
                        if exchange not in self.exchanges:
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
                    except Exception as e:
                        self.logger.debug(f"Error processing quote: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error fetching from yfinance screener: {e}")
            self.logger.info("Falling back to manual stock list scan...")
            all_candidates = self._fallback_scan(target_date)
        
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
    
    def _fallback_scan(self, target_date: datetime) -> List[Dict[str, Any]]:
        """
        Fallback method: scan a predefined list of active stocks
        This is used if yfinance screener fails
        """
        self.logger.info("Using fallback method - scanning active stocks...")
        
        # List of highly traded stocks to scan
        # These are common stocks that frequently appear in daily movers
        scan_symbols = [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'NFLX',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
            # Healthcare
            'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'UNH',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            # Consumer
            'WMT', 'HD', 'DIS', 'NKE', 'SBUX', 'MCD',
            # Volatile/Penny stocks that often move
            'SOXL', 'TQQQ', 'SPXL', 'UVXY', 'SQQQ',
        ]
        
        # Add more volatile small caps
        # These are stocks under $50 that tend to have big moves
        volatile_small_caps = [
            'AMC', 'GME', 'BBBY', 'PLUG', 'RIOT', 'MARA', 'LCID', 'RIVN',
            'NIO', 'PLTR', 'SOFI', 'CLOV', 'WISH', 'BB', 'NOK', 'SNDL'
        ]
        
        scan_symbols.extend(volatile_small_caps)
        
        candidates = []
        
        for symbol in tqdm(scan_symbols, desc="Scanning stocks"):
            try:
                ticker = yf.Ticker(symbol)
                
                # Get today's data
                hist = ticker.history(period='2d')
                
                if len(hist) < 2:
                    continue
                
                today = hist.iloc[-1]
                yesterday = hist.iloc[-2]
                
                price = today['Close']
                change_pct = ((price - yesterday['Close']) / yesterday['Close']) * 100
                volume = today['Volume']
                
                # Determine exchange (approximate)
                info = ticker.info
                exchange = info.get('exchange', 'NASDAQ')
                if exchange in ['NMS', 'NGM', 'NCM']:
                    exchange = 'NASDAQ'
                elif exchange == 'NYQ':
                    exchange = 'NYSE'
                
                if price >= self.min_price and volume >= self.min_volume and change_pct > 0:
                    candidates.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'price': float(price),
                        'change_pct': float(change_pct),
                        'volume': int(volume)
                    })
                
            except Exception as e:
                self.logger.debug(f"Error scanning {symbol}: {e}")
                continue
        
        return candidates
