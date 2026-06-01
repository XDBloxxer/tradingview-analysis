"""
Daily Non-Winners Detector
Finds stocks that did NOT explode - critical negative examples for ML training
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tradingview_scraper.symbols.screener import Screener
    SCREENER_AVAILABLE = True
except ImportError:
    SCREENER_AVAILABLE = False
    logging.warning("tradingview-scraper not available - will use yfinance only")

from .rate_limiter import RateLimiter


class DailyNonWinnersDetector:
    """
    Detects stocks that did NOT explode (negative examples)
    
    Strategy:
    1. Get list of stocks that WERE screened/predicted
    2. Get list of actual winners for the day
    3. Non-winners = screened stocks - winners
    4. Sample diverse non-winners (different changes: flat, slight up, slight down, down)
    """
    
    # Pattern exclusions (same as winners detector)
    EXCLUDED_PATTERNS = ['OTC', '.PK', '.OB', '-']

    # Default filters used when learned_filters.json is absent or a key is missing
    DEFAULT_FILTERS = {
        "min_price": 0.25,
        "max_price": None,
        "min_volume": 10000,
        "min_hv10": None,
        "max_hv10": None,
        "min_hv20": None,
        "max_hv20": None,
        "min_atr14": None,
        "min_relative_volume": None,
        "min_volume_ratio": None,
    }
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        detection_config = config.get("detection", {})
        
        self.min_price = detection_config.get("min_price", 0.25)
        self.min_volume = detection_config.get("min_volume", 10000)
        
        self.rate_limiter = RateLimiter(config)
        
        if SCREENER_AVAILABLE:
            self.screener = Screener()
        else:
            self.screener = None

        # Load learned filters from ml_models/learned_filters.json (if available)
        self.learned_filters = self._load_learned_filters()
        
        self.logger.info(
            f"Non-Winners detector initialized: "
            f"min_price={self.min_price}, min_volume={self.min_volume}"
        )

    def _load_learned_filters(self) -> dict:
        """Load learned filters from ml_models/learned_filters.json.
        
        Falls back to DEFAULT_FILTERS if the file is missing or unreadable.
        """
        defaults = dict(self.DEFAULT_FILTERS)
        try:
            filter_path = Path("ml_models/learned_filters.json")
            if filter_path.exists():
                with open(filter_path, "r") as f:
                    learned = json.load(f)

                applied = []
                for key, value in learned.items():
                    if key.startswith("_"):   # skip metadata keys
                        continue
                    if value is None:
                        continue
                    defaults[key] = value
                    applied.append(f"{key}={value}")

                if applied:
                    self.logger.info(f"Loaded learned filters for non-winners: {', '.join(applied)}")
                else:
                    self.logger.info("learned_filters.json found but contained no usable keys — using defaults")
            else:
                self.logger.info("No learned_filters.json found — using permissive defaults for non-winners")
        except Exception as e:
            self.logger.warning(f"Could not load learned_filters.json: {e} — using defaults")
        return defaults
    
    def detect_non_winners(
        self, 
        top_n: int = 15, 
        target_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Detect non-winners (negative examples)
        
        Strategy:
        1. Apply learned_filters to find a pool of stocks that *look like* they
           could have been winners — but didn't gain ≥20 % intraday.
        2. From that filtered pool, sample diverse categories the same way as
           before (flat / slight-gain / slight-loss / big-loss).
        3. If the filtered pool is too small to fill top_n, fall back to the
           original random-liquid-stock approach for the remaining slots.
        
        Args:
            top_n: Number of non-winners to collect
            target_date: Target date
            
        Returns:
            List of non-winner dictionaries
        """
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        self.logger.info(f"Detecting non-winners for {target_date_str}...")
        self.logger.info(f"Strategy: First apply learned filters, then sample {top_n} diverse non-winners")
        
        # Get actual winners to exclude
        winners_symbols = self._get_winners_symbols(target_date)
        self.logger.info(f"Found {len(winners_symbols)} winners to exclude")
        
        # ── Phase 1: learned-filter candidates ──────────────────────────────
        self.logger.info("Phase 1: Fetching candidates via learned filters...")
        filtered_candidates = self._get_filtered_candidates(winners_symbols)
        self.logger.info(f"  Learned-filter pool: {len(filtered_candidates)} stocks")

        non_winners = []

        if filtered_candidates:
            # Build a quick lookup by symbol for the filtered pool
            filtered_map: Dict[str, Dict] = {c['symbol']: c for c in filtered_candidates}
            filtered_syms = set(filtered_map.keys())

            def _pick_from_filtered(min_chg, max_chg, n):
                picked = []
                already = set(nw['symbol'] for nw in non_winners)
                for sym, data in filtered_map.items():
                    if sym in already:
                        continue
                    if min_chg <= data['change_pct'] <= max_chg:
                        picked.append(data)
                    if len(picked) >= n:
                        break
                return picked

            # Category 1: Flat stocks (-2% to +2%)
            flat_count = int(top_n * 0.3)
            flat_stocks = _pick_from_filtered(-2.0, 2.0, flat_count)
            non_winners.extend(flat_stocks)
            self.logger.info(f"  Flat  (-2% to +2%):  {len(flat_stocks)}/{flat_count} from filtered pool")

            # Category 2: Slight gainers (+2% to +10%)
            slight_gain_count = int(top_n * 0.3)
            slight_gainers = _pick_from_filtered(2.0, 10.0, slight_gain_count)
            non_winners.extend(slight_gainers)
            self.logger.info(f"  Slight gain (+2% to +10%): {len(slight_gainers)}/{slight_gain_count} from filtered pool")

            # Category 3: Slight losers (-2% to -10%)
            slight_loss_count = int(top_n * 0.2)
            slight_losers = _pick_from_filtered(-10.0, -2.0, slight_loss_count)
            non_winners.extend(slight_losers)
            self.logger.info(f"  Slight loss (-2% to -10%): {len(slight_losers)}/{slight_loss_count} from filtered pool")

            # Category 4: Bigger losers (< -10%)
            big_loss_count = top_n - len(non_winners)
            big_losers = _pick_from_filtered(-50.0, -10.0, big_loss_count)
            non_winners.extend(big_losers)
            self.logger.info(f"  Big loss (< -10%): {len(big_losers)}/{big_loss_count} from filtered pool")
        else:
            self.logger.info("  No filtered candidates returned — skipping Phase 1")
            filtered_syms = set()

        # ── Phase 2: fallback for any shortage ──────────────────────────────
        if len(non_winners) < top_n:
            shortage = top_n - len(non_winners)
            self.logger.info(
                f"Phase 2: Only {len(non_winners)}/{top_n} from filtered pool. "
                f"Filling {shortage} slots via original random-liquid-stock method..."
            )

            already_syms = set(nw['symbol'] for nw in non_winners)
            # Try category-by-category fallback first (mirrors original logic)
            per_cat_needed = shortage // 4 or 1

            fallback = []
            for min_chg, max_chg in [(-2.0, 2.0), (2.0, 10.0), (-10.0, -2.0), (-50.0, -10.0)]:
                if len(fallback) >= shortage:
                    break
                batch = self._get_stocks_by_change_range(
                    target_date, min_chg, max_chg,
                    per_cat_needed * 2,
                    winners_symbols | already_syms | filtered_syms
                )
                for stock in batch:
                    if stock['symbol'] not in already_syms and stock['symbol'] not in filtered_syms:
                        fallback.append(stock)
                        already_syms.add(stock['symbol'])
                    if len(fallback) >= shortage:
                        break

            # If still short, use the fully random fallback
            if len(fallback) < shortage:
                remaining = shortage - len(fallback)
                self.logger.info(f"  Still {remaining} short — using random liquid stocks as final fallback")
                random_stocks = self._get_random_liquid_stocks(
                    target_date, remaining * 2,
                    winners_symbols,
                    already_syms | filtered_syms
                )
                fallback.extend(random_stocks[:remaining])

            non_winners.extend(fallback[:shortage])
            self.logger.info(f"  Fallback added {len(fallback[:shortage])} stocks")

        # Add metadata
        for nw in non_winners:
            nw['detection_date'] = target_date_str
            nw['detection_time'] = '16:00:00'
        
        self.logger.info(f"✓ Collected {len(non_winners)} diverse non-winners:")
        
        # Log distribution
        flat = sum(1 for nw in non_winners if -2 <= nw['change_pct'] <= 2)
        slight_gain = sum(1 for nw in non_winners if 2 < nw['change_pct'] <= 10)
        slight_loss = sum(1 for nw in non_winners if -10 <= nw['change_pct'] < -2)
        big_loss = sum(1 for nw in non_winners if nw['change_pct'] < -10)
        
        self.logger.info(f"  Distribution:")
        self.logger.info(f"    Flat (-2% to +2%): {flat}")
        self.logger.info(f"    Slight gainers (+2% to +10%): {slight_gain}")
        self.logger.info(f"    Slight losers (-2% to -10%): {slight_loss}")
        self.logger.info(f"    Bigger losers (< -10%): {big_loss}")
        
        return non_winners[:top_n]
    
    def _get_filtered_candidates(self, exclude_symbols: set) -> List[Dict[str, Any]]:
        """
        Fetch a pool of stocks that pass the learned filters but did NOT gain ≥20 % intraday.

        Uses the TradingView screener when available (applying learned filter values as
        screener constraints), otherwise falls back to yfinance on the liquid-stocks list.
        In both cases only stocks with change_pct < 20 % are returned.
        """
        lf = self.learned_filters

        # Map from learned_filters keys → (tv_column, operation)
        TV_FILTER_MAP = {
            "min_price":           ("close",                    "greater"),
            "max_price":           ("close",                    "less"),
            "min_volume":          ("volume",                   "greater"),
            "min_hv10":            ("historical_volatility_10", "greater"),
            "max_hv10":            ("historical_volatility_10", "less"),
            "min_hv20":            ("historical_volatility_20", "greater"),
            "max_hv20":            ("historical_volatility_20", "less"),
            "min_relative_volume": ("relative_volume_10d_calc", "greater"),
            "min_volume_ratio":    ("relative_volume_10d_calc", "greater"),
        }

        if SCREENER_AVAILABLE and self.screener:
            # Build TradingView filter list from learned values
            tv_col_bounds: Dict[str, Dict] = {}
            for filter_key, (tv_col, operation) in TV_FILTER_MAP.items():
                value = lf.get(filter_key)
                if value is None:
                    continue
                if tv_col not in tv_col_bounds:
                    tv_col_bounds[tv_col] = {}
                if operation == "greater":
                    existing = tv_col_bounds[tv_col].get("min")
                    if existing is None or value > existing:
                        tv_col_bounds[tv_col]["min"] = value
                elif operation == "less":
                    existing = tv_col_bounds[tv_col].get("max")
                    if existing is None or value < existing:
                        tv_col_bounds[tv_col]["max"] = value

            tv_filters = []
            for tv_col, bounds in tv_col_bounds.items():
                if "min" in bounds:
                    tv_filters.append({"left": tv_col, "operation": "greater", "right": bounds["min"]})
                if "max" in bounds:
                    tv_filters.append({"left": tv_col, "operation": "less",    "right": bounds["max"]})

            # Also hard-exclude winners (change >= 20 %)
            tv_filters.append({"left": "change", "operation": "less", "right": 20.0})

            try:
                result = self.screener.screen(
                    market="america",
                    filters=tv_filters,
                    sort_by="volume",
                    sort_order="desc",
                    limit=500,
                )

                if result.get("status") != "success" or not result.get("data"):
                    self.logger.warning("Learned-filter screener returned no results")
                    return []

                candidates = []
                for item in result["data"]:
                    try:
                        symbol_full = item.get("symbol", "")
                        if ":" in symbol_full:
                            exchange_prefix, symbol = symbol_full.split(":", 1)
                        else:
                            symbol = symbol_full
                            exchange_prefix = "NASDAQ"

                        if not symbol or symbol in exclude_symbols:
                            continue
                        if self._is_excluded_symbol(symbol, exchange_prefix):
                            continue

                        price      = float(item.get("close",  0))
                        change_pct = float(item.get("change", 0))
                        volume     = int(item.get("volume",   0))

                        if price < self.min_price or volume < self.min_volume:
                            continue
                        if change_pct >= 20.0:
                            continue

                        candidates.append({
                            "symbol":     symbol.strip().upper(),
                            "exchange":   exchange_prefix,
                            "price":      float(price),
                            "change_pct": float(change_pct),
                            "volume":     int(volume),
                            "high":       float(item.get("high", price)),
                            "low":        float(item.get("low",  price)),
                            "open":       float(item.get("open", price)),
                            "close":      float(price),
                        })
                    except Exception:
                        continue

                return candidates

            except Exception as e:
                self.logger.warning(f"Learned-filter screener error: {e} — falling back to yfinance")

        # ── yfinance fallback ────────────────────────────────────────────────
        min_price_filter  = lf.get("min_price",  self.min_price)
        max_price_filter  = lf.get("max_price",  None)
        min_volume_filter = lf.get("min_volume", self.min_volume)

        liquid_stocks = [s for s in self._get_liquid_stocks_list() if s not in exclude_symbols]
        candidates = []

        for symbol in liquid_stocks:
            try:
                ticker = yf.Ticker(symbol)
                hist   = ticker.history(period="2d", interval="1d")

                if hist.empty or len(hist) < 2:
                    continue

                latest   = hist.iloc[-1]
                previous = hist.iloc[-2]

                close      = latest["Close"]
                prev_close = previous["Close"]
                volume     = latest["Volume"]

                change_pct = ((close - prev_close) / prev_close) * 100

                if change_pct >= 20.0:
                    continue
                if close < min_price_filter:
                    continue
                if max_price_filter is not None and close > max_price_filter:
                    continue
                if volume < min_volume_filter:
                    continue

                candidates.append({
                    "symbol":     symbol,
                    "exchange":   "NASDAQ",
                    "price":      float(close),
                    "change_pct": float(change_pct),
                    "volume":     int(volume),
                    "high":       float(latest["High"]),
                    "low":        float(latest["Low"]),
                    "open":       float(latest["Open"]),
                    "close":      float(close),
                })
            except Exception:
                continue

            time.sleep(0.1)

        return candidates

    def _get_winners_symbols(self, target_date: datetime) -> set:
        """Get set of symbols that were winners on this date"""
        try:
            # Import here to avoid circular dependency
            from .daily_winners_supabase_client import DailyWinnersSupabaseClient
            
            client = DailyWinnersSupabaseClient(self.config)
            winners_df = client.read_winners(
                start_date=target_date.date().isoformat(),
                end_date=target_date.date().isoformat()
            )
            
            if winners_df.empty:
                return set()
            
            return set(winners_df['symbol'].tolist())
            
        except Exception as e:
            self.logger.warning(f"Could not fetch winners: {e}")
            return set()
    
    def _get_stocks_by_change_range(
        self,
        target_date: datetime,
        min_change: float,
        max_change: float,
        limit: int,
        exclude_symbols: set
    ) -> List[Dict[str, Any]]:
        """
        Get stocks within a specific change percentage range
        
        Args:
            target_date: Target date
            min_change: Minimum change %
            max_change: Maximum change %
            limit: Number of stocks to get
            exclude_symbols: Symbols to exclude (winners)
            
        Returns:
            List of stock dictionaries
        """
        try:
            if SCREENER_AVAILABLE and self.screener:
                return self._screen_by_change_range(
                    min_change, max_change, limit, exclude_symbols
                )
            else:
                return self._get_from_liquid_stocks(
                    target_date, min_change, max_change, limit, exclude_symbols
                )
        except Exception as e:
            self.logger.error(f"Error getting stocks in range {min_change}% to {max_change}%: {e}")
            return []
    
    def _screen_by_change_range(
        self,
        min_change: float,
        max_change: float,
        limit: int,
        exclude_symbols: set
    ) -> List[Dict[str, Any]]:
        """Use TradingView screener to find stocks in change range"""
        try:
            filters = [
                {'left': 'close', 'operation': 'greater', 'right': self.min_price},
                {'left': 'volume', 'operation': 'greater', 'right': self.min_volume},
                {'left': 'change', 'operation': 'greater', 'right': min_change},
                {'left': 'change', 'operation': 'less', 'right': max_change}
            ]
            
            result = self.screener.screen(
                market='america',
                filters=filters,
                sort_by='volume',
                sort_order='desc',
                limit=limit * 2
            )
            
            if result['status'] != 'success' or not result.get('data'):
                return []
            
            candidates = []
            for item in result['data']:
                try:
                    symbol_full = item.get('symbol', '')
                    if ':' in symbol_full:
                        exchange_prefix, symbol = symbol_full.split(':', 1)
                    else:
                        symbol = symbol_full
                        exchange_prefix = 'NASDAQ'
                    
                    if not symbol or symbol in exclude_symbols:
                        continue
                    
                    if self._is_excluded_symbol(symbol, exchange_prefix):
                        continue
                    
                    price = float(item.get('close', 0))
                    change_pct = float(item.get('change', 0))
                    volume = int(item.get('volume', 0))
                    
                    if price < self.min_price or volume < self.min_volume:
                        continue
                    
                    candidates.append({
                        'symbol': symbol.strip().upper(),
                        'exchange': exchange_prefix,
                        'price': float(price),
                        'change_pct': float(change_pct),
                        'volume': int(volume),
                        'high': float(item.get('high', price)),
                        'low': float(item.get('low', price)),
                        'open': float(item.get('open', price)),
                        'close': float(price)
                    })
                    
                    if len(candidates) >= limit:
                        break
                        
                except Exception as e:
                    continue
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"TradingView screener error: {e}")
            return []
    
    def _get_from_liquid_stocks(
        self,
        target_date: datetime,
        min_change: float,
        max_change: float,
        limit: int,
        exclude_symbols: set
    ) -> List[Dict[str, Any]]:
        """Get stocks from liquid stock list using yfinance"""
        liquid_stocks = self._get_liquid_stocks_list()
        liquid_stocks = [s for s in liquid_stocks if s not in exclude_symbols]
        
        candidates = []
        
        for symbol in liquid_stocks:
            if len(candidates) >= limit:
                break
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d', interval='1d')
                
                if hist.empty or len(hist) < 2:
                    continue
                
                latest = hist.iloc[-1]
                previous = hist.iloc[-2]
                
                close = latest['Close']
                prev_close = previous['Close']
                volume = latest['Volume']
                
                change_pct = ((close - prev_close) / prev_close) * 100
                
                if min_change <= change_pct <= max_change:
                    if close >= self.min_price and volume >= self.min_volume:
                        candidates.append({
                            'symbol': symbol,
                            'exchange': 'NASDAQ',
                            'price': float(close),
                            'change_pct': float(change_pct),
                            'volume': int(volume),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'open': float(latest['Open']),
                            'close': float(close)
                        })
                
            except Exception as e:
                continue
            
            time.sleep(0.1)
        
        return candidates
    
    def _get_random_liquid_stocks(
        self,
        target_date: datetime,
        limit: int,
        exclude_winners: set,
        exclude_existing: set
    ) -> List[Dict[str, Any]]:
        """Get random stocks from liquid list as fallback"""
        liquid_stocks = self._get_liquid_stocks_list()
        liquid_stocks = [
            s for s in liquid_stocks 
            if s not in exclude_winners and s not in exclude_existing
        ]
        
        candidates = []
        
        for symbol in liquid_stocks[:limit * 2]:
            if len(candidates) >= limit:
                break
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d', interval='1d')
                
                if hist.empty or len(hist) < 2:
                    continue
                
                latest = hist.iloc[-1]
                previous = hist.iloc[-2]
                
                close = latest['Close']
                prev_close = previous['Close']
                volume = latest['Volume']
                
                change_pct = ((close - prev_close) / prev_close) * 100
                
                # Exclude big gainers (>20%)
                if change_pct > 20:
                    continue
                
                if close >= self.min_price and volume >= self.min_volume:
                    candidates.append({
                        'symbol': symbol,
                        'exchange': 'NASDAQ',
                        'price': float(close),
                        'change_pct': float(change_pct),
                        'volume': int(volume),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'open': float(latest['Open']),
                        'close': float(close)
                    })
                
            except Exception as e:
                continue
            
            time.sleep(0.1)
        
        return candidates
    
    def _is_excluded_symbol(self, symbol: str, exchange: str) -> bool:
        """Check if symbol should be excluded"""
        if exchange == 'OTC':
            return True
        
        symbol_upper = symbol.upper()
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern in symbol_upper:
                return True
        
        if len(symbol) > 5:
            return True
        
        return False
    
    def _get_liquid_stocks_list(self) -> List[str]:
        """Get list of liquid stocks for sampling"""
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
            'F', 'GM', 'T', 'BAC', 'WFC', 'MS', 'AIG', 'GE', 'XOM', 'CHTR',
            'DAL', 'AAL', 'UAL', 'CCL', 'NCLH', 'MAR', 'HLT', 'MGM', 'WYNN', 'LVS'
        ]
