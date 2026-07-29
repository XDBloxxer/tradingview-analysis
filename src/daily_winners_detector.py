"""
Daily Winners Detector - DEBUG VERSION
This version has extensive logging to diagnose why stale stocks pass validation

FIX: All three fetch paths now populate high, low, open, close so the
     daily_winners table is fully populated after the schema migration.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf
import time

try:
    from tradingview_scraper.symbols.market_movers import MarketMovers
except ImportError:
    raise ImportError(
        "tradingview-scraper is required. Install with: pip install tradingview-scraper"
    )

from .rate_limiter import RateLimiter


class DailyWinnersDetector:
    """
    Detects top daily winners - DEBUG VERSION with extensive logging
    """
    
    EXCLUDED_PATTERNS = [
        'OTC',  # Over-the-counter
        '.PK',  # Pink sheets
        '.OB',  # OTC Bulletin Board
        '-',    # Often delisted stocks
    ]

    # ── POINT-IN-TIME UNIVERSE FIX (mirrors daily_non_winners_detector.py) ──
    # Same hardcoded ~100-megacap fallback bug, same fix: a broad universe
    # file with anti-repetition tracking, kept in a SEPARATE counts file from
    # the non-winners side since winners/non-winners are different quotas —
    # a symbol legitimately can and should appear as a candidate check on
    # both sides across a backfill without either quota starving the other.
    UNIVERSE_PATH = Path("data/universe_symbols.csv")
    SELECTION_COUNTS_PATH = Path("data/winner_selection_counts.json")
    MAX_USES_PER_SYMBOL = 10

    # ── WINNER PROFILE FIX ───────────────────────────────────────────────
    # Every candidate path (TradingView, yfinance screener, liquid-stocks
    # live, liquid-stocks backfill) was only checking price > min_price,
    # volume > min_volume, and change_pct > 0 — i.e. "any stock that went up
    # at all, at any price." That's not what "winner" means anywhere else in
    # this pipeline:
    #   - learned_filters.json (derived from 1311 real historical winners)
    #     caps max_price at 50.0 — winners are the small-cap explosive
    #     universe, not megacaps that drifted up 0.3%.
    #   - the non-winners screener hard-excludes change >= 20% specifically
    #     because that's the boundary that separates "explosive winner" from
    #     "ordinary mover" (see _screen_by_change_range's `change < 20.0`
    #     filter in daily_non_winners_detector.py).
    # Without both of those enforced here, "winners" backfilled from a broad
    # point-in-time universe (which, unlike the old 120-megacap list, now
    # legitimately contains high-priced blue chips) can end up being the
    # least-bad stock in a small random sample rather than an actual
    # explosive mover — which is exactly the "under 20% gain, priced over
    # $50" symptom being fixed here.
    DEFAULT_MAX_PRICE = 50.0
    DEFAULT_MIN_CHANGE_PCT = 20.0

    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        detection_config = config.get("detection", {})
        
        self.min_price = detection_config.get("min_price", 0.25)
        self.min_volume = detection_config.get("min_volume", 10000)

        winners_config = config.get("daily_winners", {})
        learned = self._load_learned_filters()
        # learned_filters.json wins if present; config.yaml is next; hardcoded
        # default is last resort. min_change_pct isn't part of learned_filters.json
        # (it only carries price/volume/market-cap/volatility bounds), so it comes
        # from config.yaml's daily_winners section or the class default.
        self.max_price = learned.get("max_price") or winners_config.get(
            "max_price", self.DEFAULT_MAX_PRICE
        )
        self.min_change_pct = float(
            winners_config.get("min_change_pct", self.DEFAULT_MIN_CHANGE_PCT)
        )

        self._universe_cache: Optional[List[str]] = None
        self._selection_counts: Dict[str, int] = self._load_selection_counts()
        
        self.rate_limiter = RateLimiter(config)
        self.market_movers = MarketMovers(export_result=False)
        self.freshness_cache = {}
        
        self.logger.info(
            f"Daily Winners detector initialized (DEBUG MODE): "
            f"min_price={self.min_price}, min_volume={self.min_volume}, "
            f"max_price={self.max_price}, min_change_pct={self.min_change_pct}"
        )

    def _load_learned_filters(self) -> dict:
        """Load max_price (and nothing else, for now) from
        ml_models/learned_filters.json if present. Mirrors
        DailyNonWinnersDetector._load_learned_filters so both sides of the
        pipeline agree on what a 'winner-shaped' stock looks like — same
        file, same max_price, so the winners table and the non-winners
        exclusion boundary stay consistent with each other."""
        try:
            filter_path = Path("ml_models/learned_filters.json")
            if not filter_path.exists():
                self.logger.info(
                    "No learned_filters.json found — using default max_price="
                    f"{self.DEFAULT_MAX_PRICE} for winners."
                )
                return {}
            with open(filter_path) as f:
                data = json.load(f)
            return {"max_price": data.get("max_price")}
        except Exception as e:
            self.logger.warning(f"Could not load learned_filters.json: {e} — using defaults")
            return {}

    def _is_live(self, target_date: datetime) -> bool:
        """
        True if `target_date` is today (a live run), False if it's a
        historical backfill. Only live runs may fall back to "most recent
        bar" style fetches (period='2d'/'5d' + iloc[-1]); backfills must use
        a fetch pinned to the exact target_date, or that date's row can get
        silently overwritten with today's bar (the same class of bug fixed
        in _backfill_ohlc / _fetch_yf_bar_for_date below).
        """
        if target_date is None:
            return True
        target = target_date.date() if isinstance(target_date, datetime) else target_date
        return target == datetime.now().date()

    def _fetch_yf_bar_for_date(self, symbol: str, target_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch the daily OHLCV bar for `symbol` on `target_date` specifically
        (not "whatever's most recent"). Returns None if that exact date isn't
        present (non-trading day, delisted, no data yet) rather than
        substituting a different day's bar.
        """
        target_date_only = target_date.date() if isinstance(target_date, datetime) else target_date
        start = target_date_only - timedelta(days=10)
        end = target_date_only + timedelta(days=1)

        try:
            hist = self.rate_limiter.call_with_backoff(
                yf.download,
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                interval='1d',
                progress=False,
                auto_adjust=True,
                label=f"{symbol} OHLC bar",
            )
        except Exception as e:
            self.logger.debug(f"{symbol}: yfinance fetch failed: {e}")
            return None

        if hist is None or hist.empty:
            return None

        idx_dates = hist.index.date
        target_mask = idx_dates == target_date_only
        if not target_mask.any():
            return None

        target_pos = int(target_mask.nonzero()[0][-1])
        if target_pos == 0:
            return None

        window = hist.iloc[:target_pos + 1]
        latest = window.iloc[-1]
        previous = window.iloc[-2]

        try:
            close = float(latest['Close'])
            prev_close = float(previous['Close'])
            volume = int(latest['Volume'])
        except (TypeError, ValueError):
            return None

        if prev_close == 0:
            return None

        return {
            'close': close,
            'prev_close': prev_close,
            'change_pct': ((close - prev_close) / prev_close) * 100,
            'volume': volume,
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'open': float(latest['Open']),
        }
    
    def detect_top_winners(self, top_n: int = 15, target_date: datetime = None) -> List[Dict[str, Any]]:
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.date().isoformat()
        
        self.logger.info(f"Fetching top {top_n} day gainers from TradingView MarketMovers...")
        
        # Fetch MORE than we need initially to account for validation rejections
        initial_fetch_size = top_n * 2
        
        candidates = self._fetch_from_tradingview(target_date, initial_fetch_size)
        
        # Supplement with yfinance if needed
        needed = initial_fetch_size - len(candidates)
        if needed > 0:
            self.logger.info(
                f"TradingView returned {len(candidates)}/{initial_fetch_size}. "
                f"Fetching {needed} more from yfinance..."
            )
            existing_symbols = {c['symbol'] for c in candidates}
            yf_candidates = self._fetch_from_yfinance_screener(target_date, needed * 2, existing_symbols)
            candidates.extend(yf_candidates[:needed])
        
        if not candidates:
            self.logger.warning("⚠️ No winners found!")
            return []
        
        # ── Backfill OHLC for any candidate missing it (TradingView path) ──
        candidates = self._backfill_ohlc(candidates, target_date)
        
        self.logger.info(f"Total candidates before validation: {len(candidates)}")
        
        skip_validation = self.config.get('daily_winners', {}).get('skip_freshness_validation', False)
        self.logger.info(f"🔍 skip_freshness_validation config: {skip_validation}")
        
        if skip_validation:
            self.logger.warning("⚠️ SKIPPING freshness validation per config!")
            validated_candidates = candidates
        else:
            self.logger.info(f"🔍 Running batch freshness validation on {len(candidates)} candidates...")
            validated_candidates = self._batch_verify_freshness(candidates, target_date)
        
        if not validated_candidates:
            self.logger.warning("⚠️ All candidates failed freshness validation!")
            return []
        
        # Sort by change percentage
        df_results = pd.DataFrame(validated_candidates)
        df_results = df_results.sort_values('change_pct', ascending=False)
        
        # Backfill shortage
        if len(df_results) < top_n:
            shortage = top_n - len(df_results)
            self.logger.warning(
                f"⚠️ Only {len(df_results)} stocks passed validation (need {top_n}). "
                f"Shortage: {shortage} stocks"
            )
            self.logger.info("💡 Trying to backfill with additional candidates...")
            
            existing_symbols = {c['symbol'] for c in candidates}
            additional_candidates = self._fetch_from_yfinance_screener(
                target_date, shortage * 3, existing_symbols
            )
            
            if additional_candidates:
                additional_candidates = self._backfill_ohlc(additional_candidates, target_date)
                self.logger.info(f"Fetched {len(additional_candidates)} additional candidates")
                
                if not skip_validation:
                    additional_validated = self._batch_verify_freshness(additional_candidates, target_date)
                else:
                    additional_validated = additional_candidates
                
                if additional_validated:
                    self.logger.info(f"✅ {len(additional_validated)} additional candidates passed validation")
                    validated_candidates.extend(additional_validated)
                    df_results = pd.DataFrame(validated_candidates)
                    df_results = df_results.sort_values('change_pct', ascending=False)
        
        top_winners = df_results.head(top_n).to_dict('records')
        
        for winner in top_winners:
            winner['detection_date'] = target_date_str
            winner['detection_time'] = '16:00:00'
        
        self.logger.info(f"✅ Found top {len(top_winners)} daily winners (target: {top_n}):")
        for i, winner in enumerate(top_winners, 1):
            source = winner.pop('source', 'unknown')
            self.logger.info(
                f"  #{i}: {winner['exchange']}:{winner['symbol']} "
                f"(+{winner['change_pct']:.2f}%) @ ${winner['price']:.2f}, "
                f"vol={winner['volume']:,} [{source}]"
            )
        
        if len(top_winners) < top_n:
            self.logger.warning(
                f"⚠️ WARNING: Only found {len(top_winners)}/{top_n} winners after validation."
            )

        # Record EVERY selected winner here, regardless of which path found
        # them (TradingView / yfinance screener / liquid-stocks fallback).
        # Previously this only happened inside _fetch_from_liquid_stocks_for_date,
        # so on any day where TradingView/yfinance successfully returned
        # candidates (the normal case) nothing was ever recorded and
        # data/winner_selection_counts.json never got created.
        self._record_selections([w['symbol'] for w in top_winners])

        return top_winners
    
    # ─────────────────────────────────────────────────────────────────────
    # OHLC backfill — fetches daily bar OHLC for any candidate that came
    # in without it (mainly the TradingView MarketMovers path).
    # Uses a single yf.download() batch call for efficiency.
    # ─────────────────────────────────────────────────────────────────────

    def _backfill_ohlc(
        self,
        candidates: List[Dict[str, Any]],
        target_date: datetime,
    ) -> List[Dict[str, Any]]:
        """
        For any candidate missing high/low/open/close, fetch from yfinance
        daily bars (1d interval) and fill them in — for `target_date`
        specifically.

        TradingView MarketMovers only returns close + change + volume.
        yfinance Screener returns regularMarketPrice but not the full OHLC bar.
        This method patches both.

        FIX: This previously called yf.download(period='2d') and always took
        the LAST row (iloc[-1]) regardless of what date was being backfilled.
        When this ran as part of a bulk catch-up job across many historical
        detection_dates, every candidate — no matter which historical date it
        belonged to — got patched with whatever the most-recent trading bar
        happened to be at the moment the script ran. That silently wrote the
        same OHLC snapshot into the table under dozens of different
        detection_date rows (confirmed via duplicate symbol+OHLC combinations
        spanning ~38 distinct dates with identical values).

        The fix fetches an explicit [target_date, target_date + 1 day) window
        and selects the row matching target_date, rather than "whatever's
        last." If the market was closed that day (weekend/holiday) or data
        isn't available for that exact date, we skip the backfill for that
        candidate rather than silently substituting a different day's bar.
        """
        missing = [c for c in candidates
                   if any(c.get(f) is None for f in ('high', 'low', 'open', 'close'))]

        if not missing:
            return candidates

        symbols = list({c['symbol'] for c in missing})
        target_date_only = target_date.date()
        # yfinance's `end` is exclusive, so request a small window and filter
        # down to the exact date rather than relying on "last row returned."
        start = target_date_only
        end = target_date_only + timedelta(days=1)

        self.logger.info(
            f"Backfilling OHLC for {len(symbols)} candidates via yfinance "
            f"daily bars for {target_date_only.isoformat()}..."
        )

        try:
            data = self.rate_limiter.call_with_backoff(
                yf.download,
                symbols,
                start=start.isoformat(),
                end=end.isoformat(),
                interval='1d',
                group_by='ticker',
                progress=False,
                threads=True,
                auto_adjust=True,
                label=f"bulk OHLC backfill ({len(symbols)} symbols) {target_date_only.isoformat()}",
            )

            if data.empty:
                self.logger.warning(
                    f"yfinance returned no OHLC data for {target_date_only.isoformat()} "
                    "(likely a non-trading day, or data not yet available) — "
                    "skipping backfill rather than substituting another day's bar."
                )
                return candidates

            # Build a quick lookup: symbol → OHLC bar for target_date_only.
            ohlc_lookup: Dict[str, Dict[str, float]] = {}

            def _row_for_target_date(frame: pd.DataFrame):
                """Return the row whose index date matches target_date_only,
                or None if that exact date isn't present in the response."""
                if frame.empty:
                    return None
                idx_dates = frame.index.date
                matches = frame.loc[idx_dates == target_date_only]
                if matches.empty:
                    return None
                return matches.iloc[0]

            if len(symbols) == 1:
                # Single-symbol download: columns are Open/High/Low/Close/Volume (no ticker level)
                row = _row_for_target_date(data)
                if row is not None:
                    ohlc_lookup[symbols[0]] = {
                        'open':  float(row.get('Open',  row.get('open',  0))),
                        'high':  float(row.get('High',  row.get('high',  0))),
                        'low':   float(row.get('Low',   row.get('low',   0))),
                        'close': float(row.get('Close', row.get('close', 0))),
                    }
            else:
                # Multi-symbol: columns are MultiIndex (ticker, field)
                for sym in symbols:
                    try:
                        if sym not in data.columns.get_level_values(0):
                            continue
                        sym_data = data[sym].dropna(how='all')
                        row = _row_for_target_date(sym_data)
                        if row is None:
                            continue
                        ohlc_lookup[sym] = {
                            'open':  float(row.get('Open',  row.get('open',  0))),
                            'high':  float(row.get('High',  row.get('high',  0))),
                            'low':   float(row.get('Low',   row.get('low',   0))),
                            'close': float(row.get('Close', row.get('close', 0))),
                        }
                    except Exception:
                        continue

        except Exception as e:
            self.logger.warning(f"OHLC backfill failed: {e}")
            return candidates

        # Patch candidates in-place
        patched = 0
        skipped_no_match = 0
        for c in candidates:
            sym = c.get('symbol')
            if sym in ohlc_lookup:
                bar = ohlc_lookup[sym]
                for field in ('open', 'high', 'low', 'close'):
                    if c.get(field) is None:
                        c[field] = bar[field]
                # Also align price with close if it was 0 / None
                if not c.get('price') and bar['close']:
                    c['price'] = bar['close']
                patched += 1
            elif sym in {m['symbol'] for m in missing}:
                skipped_no_match += 1

        self.logger.info(
            f"✓ OHLC backfilled for {patched}/{len(missing)} candidates "
            f"for {target_date_only.isoformat()}"
            + (f" ({skipped_no_match} skipped — no bar for that exact date, "
               "left unpatched rather than guessing)" if skipped_no_match else "")
        )
        return candidates

    # ─────────────────────────────────────────────────────────────────────
    # Existing helpers (unchanged except _fetch_from_tradingview which now
    # includes high/low/open when the scraper returns them)
    # ─────────────────────────────────────────────────────────────────────

    def _is_excluded_symbol(self, symbol: str, exchange: str) -> bool:
        if exchange == 'OTC':
            return True
        symbol_upper = symbol.upper()
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern in symbol_upper:
                return True
        if len(symbol) > 5:
            return True
        return False
    
    def _fetch_from_tradingview(self, target_date: datetime, top_n: int) -> List[Dict[str, Any]]:
        # TradingView's MarketMovers "gainers" scrape has no historical/point-in-time
        # mode — it always returns *today's* real-time gainers regardless of what
        # target_date is passed in. Calling it for a backfill date silently mislabels
        # today's actual winners as winners for target_date (the exact bug reported:
        # backfilling 02-04 came back with today's winners). Gate it to live runs only.
        if not self._is_live(target_date):
            self.logger.info(
                f"⏭️  Skipping TradingView MarketMovers for {target_date.date().isoformat()} "
                f"(not live — TradingView gainers has no point-in-time mode, only 'today')."
            )
            return []

        try:
            fetch_limit = max(top_n * 10, 500)
            
            result = self.market_movers.scrape(
                market='stocks-usa',
                category='gainers',
                limit=fetch_limit
            )
            
            if result['status'] != 'success':
                self.logger.error(f"TradingView API error: {result.get('status')}")
                return []
            
            data = result.get('data', [])
            if not data:
                self.logger.warning("No data from TradingView")
                return []
            
            self.logger.info(f"Received {len(data)} gainers from TradingView")
            
            all_candidates = []
            filtered_counts = {
                'excluded_pattern': 0,
                'low_price': 0,
                'high_price': 0,
                'low_volume': 0,
                'no_change': 0,
                'parse_error': 0
            }
            
            for item in data:
                try:
                    symbol_full = item.get('symbol', '')
                    if ':' in symbol_full:
                        exchange_prefix, symbol = symbol_full.split(':', 1)
                    else:
                        symbol = symbol_full
                        exchange_prefix = 'NASDAQ'
                    
                    if not symbol:
                        continue
                    
                    exchange_map = {
                        'NASDAQ': 'NASDAQ',
                        'NYSE':   'NYSE',
                        'AMEX':   'AMEX',
                        'BATS':   'NASDAQ'
                    }
                    exchange = exchange_map.get(exchange_prefix, exchange_prefix)
                    
                    if self._is_excluded_symbol(symbol, exchange):
                        filtered_counts['excluded_pattern'] += 1
                        self.logger.debug(f"🚫 Filtered (pattern): {exchange}:{symbol}")
                        continue
                    
                    price      = float(item.get('close', 0))
                    change_pct = float(item.get('change', 0))
                    volume     = int(item.get('volume', 0))
                    
                    if price < self.min_price:
                        filtered_counts['low_price'] += 1
                        self.logger.debug(f"🚫 Filtered (price): {symbol} ${price:.2f}")
                        continue
                    
                    if price > self.max_price:
                        filtered_counts['high_price'] += 1
                        self.logger.debug(f"🚫 Filtered (price > max): {symbol} ${price:.2f}")
                        continue
                    
                    if volume < self.min_volume:
                        filtered_counts['low_volume'] += 1
                        self.logger.debug(f"🚫 Filtered (volume): {symbol} {volume:,}")
                        continue
                    
                    if change_pct < self.min_change_pct:
                        filtered_counts['no_change'] += 1
                        continue
                    
                    # Capture OHLC from TradingView if available; None otherwise
                    # (None fields are backfilled by _backfill_ohlc)
                    all_candidates.append({
                        'symbol':     symbol.strip().upper(),
                        'exchange':   exchange,
                        'price':      float(price),
                        'change_pct': float(change_pct),
                        'volume':     int(volume),
                        'high':       float(item['high'])  if item.get('high')  else None,
                        'low':        float(item['low'])   if item.get('low')   else None,
                        'open':       float(item['open'])  if item.get('open')  else None,
                        'close':      float(item['close']) if item.get('close') else float(price),
                        'source':     'tradingview',
                    })
                    
                    self.logger.debug(f"✅ Passed filters: {exchange}:{symbol} +{change_pct:.2f}%")
                    
                except Exception:
                    filtered_counts['parse_error'] += 1
                    continue
            
            self.logger.info(
                f"TradingView filtering: total={len(data)}, "
                f"excluded={filtered_counts['excluded_pattern']}, "
                f"low_price={filtered_counts['low_price']}, "
                f"high_price={filtered_counts['high_price']}, "
                f"low_volume={filtered_counts['low_volume']}, "
                f"no_change={filtered_counts['no_change']}, "
                f"passed={len(all_candidates)}"
            )
            
            return all_candidates
            
        except Exception as e:
            self.logger.error(f"Error fetching from TradingView: {e}", exc_info=True)
            return []
    
    def _fetch_from_yfinance_screener(
        self, 
        target_date: datetime, 
        limit: int,
        exclude_symbols: Set[str]
    ) -> List[Dict[str, Any]]:
        # Same problem as TradingView: yfinance's `Screener().get_screeners(['day_gainers'])`
        # is a live "today's movers" endpoint with no target_date parameter — there is
        # no way to ask it for gainers on a past date. For a backfill date it would
        # silently hand back today's real winners again. Route straight to the
        # point-in-time universe scan instead.
        if not self._is_live(target_date):
            self.logger.info(
                f"⏭️  Skipping yfinance day_gainers Screener for {target_date.date().isoformat()} "
                f"(not live — it only returns today's movers, no historical mode)."
            )
            return self._fetch_from_liquid_stocks(target_date, limit, exclude_symbols)

        try:
            self.logger.info("Fetching from yfinance screener...")
            
            try:
                from yfinance import Screener
                
                screener = Screener()
                gainers_data = screener.get_screeners(['day_gainers'], count=limit * 2)
                
                if not gainers_data or 'day_gainers' not in gainers_data:
                    return self._fetch_from_liquid_stocks(target_date, limit, exclude_symbols)
                
                quotes = gainers_data['day_gainers'].get('quotes', [])
                candidates = []
                
                for quote in quotes:
                    symbol = quote.get('symbol', '')
                    if not symbol or symbol in exclude_symbols:
                        continue
                    if self._is_excluded_symbol(symbol, 'NASDAQ'):
                        continue
                    
                    price      = quote.get('regularMarketPrice', 0)
                    change_pct = quote.get('regularMarketChangePercent', 0)
                    volume     = quote.get('regularMarketVolume', 0)
                    
                    if (price < self.min_price or price > self.max_price
                            or volume < self.min_volume or change_pct < self.min_change_pct):
                        continue
                    candidates.append({
                        'symbol':     symbol.upper(),
                        'exchange':   quote.get('exchange', 'NASDAQ'),
                        'price':      float(price),
                        'change_pct': float(change_pct),
                        'volume':     int(volume),
                        'high':       float(quote['regularMarketDayHigh'])  if quote.get('regularMarketDayHigh')  else None,
                        'low':        float(quote['regularMarketDayLow'])   if quote.get('regularMarketDayLow')   else None,
                        'open':       float(quote['regularMarketOpen'])     if quote.get('regularMarketOpen')     else None,
                        'close':      float(price),
                        'source':     'yfinance_screener',
                    })
                    
                    if len(candidates) >= limit:
                        break
                
                self.logger.info(f"✅ Found {len(candidates)} from yfinance Screener")
                return candidates
                
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Screener unavailable: {e}")
                return self._fetch_from_liquid_stocks(target_date, limit, exclude_symbols)
                
        except Exception as e:
            self.logger.error(f"Error in yfinance screener: {e}", exc_info=True)
            return []
    
    def _fetch_from_liquid_stocks(
        self,
        target_date: datetime,
        limit: int,
        exclude_symbols: Set[str]
    ) -> List[Dict[str, Any]]:
        if not self._is_live(target_date):
            return self._fetch_from_liquid_stocks_for_date(target_date, limit, exclude_symbols)

        try:
            self.logger.info("Scanning liquid stocks...")
            
            candidates = []
            liquid_stocks = self._get_liquid_stocks_list()
            liquid_stocks = [s for s in liquid_stocks if s not in exclude_symbols]
            
            batch_size = 50
            for i in range(0, len(liquid_stocks), batch_size):
                batch = liquid_stocks[i:i + batch_size]
                
                try:
                    data = self.rate_limiter.call_with_backoff(
                        yf.download,
                        batch,
                        period='2d',
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        threads=True,
                        auto_adjust=True,
                        label=f"live liquid batch ({len(batch)} symbols)",
                    )
                    
                    if data.empty:
                        continue
                    
                    for symbol in batch:
                        if symbol in exclude_symbols:
                            continue
                        
                        try:
                            if len(batch) == 1:
                                stock_data = data
                            else:
                                if symbol not in data.columns.get_level_values(0):
                                    continue
                                stock_data = data[symbol]
                            
                            if stock_data.empty or len(stock_data) < 2:
                                continue
                            
                            latest   = stock_data.iloc[-1]
                            previous = stock_data.iloc[-2]
                            
                            close      = float(latest['Close'])
                            prev_close = float(previous['Close'])
                            volume     = int(latest['Volume'])
                            change_pct = ((close - prev_close) / prev_close) * 100
                            
                            if (close < self.min_price or close > self.max_price
                                    or volume < self.min_volume or change_pct < self.min_change_pct):
                                continue
                            if self._is_excluded_symbol(symbol, 'NASDAQ'):
                                continue
                            
                            candidates.append({
                                'symbol':     symbol.upper(),
                                'exchange':   'NASDAQ',
                                'price':      close,
                                'change_pct': float(change_pct),
                                'volume':     volume,
                                # Full OHLC available from daily bar
                                'high':       float(latest['High']),
                                'low':        float(latest['Low']),
                                'open':       float(latest['Open']),
                                'close':      close,
                                'source':     'yfinance_liquid',
                            })
                            
                        except Exception:
                            continue
                
                except Exception:
                    continue
                
                if len(candidates) >= limit:
                    break
                
                time.sleep(0.3)
            
            if candidates:
                candidates = sorted(candidates, key=lambda x: x['change_pct'], reverse=True)
                candidates = candidates[:limit]
            
            self.logger.info(f"✅ Found {len(candidates)} from liquid stocks")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error in liquid stocks: {e}", exc_info=True)
            return []

    # Scans batches of symbols in parallel rather than one `yf.download` call
    # at a time. This mattered less before min_change_pct was enforced (any
    # positive mover passed, so few symbols needed checking) — with a real
    # 20% floor, most symbols get rejected and the scan has to get much
    # deeper into the universe to gather enough qualifying candidates.
    # Serial fetches at that depth is what turned a 15-stock backfill into a
    # 15-minute wait; concurrency is the fix, not a smaller universe.
    SCAN_WORKERS = 20
    SCAN_BATCH_SIZE = 60

    def _fetch_from_liquid_stocks_for_date(
        self,
        target_date: datetime,
        limit: int,
        exclude_symbols: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Backfill-safe counterpart to _fetch_from_liquid_stocks. Uses
        _fetch_yf_bar_for_date so every candidate's OHLC is pinned to
        target_date instead of "whatever yfinance's last row is right now."

        Scans in parallel batches of SCAN_BATCH_SIZE via a thread pool
        (SCAN_WORKERS workers) instead of one symbol at a time, stopping as
        soon as enough candidates have been gathered.
        """
        self.logger.info(f"Scanning liquid stocks for {target_date.date().isoformat()}...")

        candidates = []
        liquid_stocks = self._get_liquid_stocks_list()
        liquid_stocks = [s for s in liquid_stocks if s not in exclude_symbols]

        target_pool = limit * 3
        scanned = 0

        for batch_start in range(0, len(liquid_stocks), self.SCAN_BATCH_SIZE):
            batch = liquid_stocks[batch_start:batch_start + self.SCAN_BATCH_SIZE]
            scanned += len(batch)

            with ThreadPoolExecutor(max_workers=self.SCAN_WORKERS) as executor:
                future_to_symbol = {
                    executor.submit(self._fetch_yf_bar_for_date, symbol, target_date): symbol
                    for symbol in batch
                }
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        bar = future.result()
                    except Exception:
                        continue
                    if bar is None:
                        continue

                    close, volume, change_pct = bar['close'], bar['volume'], bar['change_pct']
                    if (close < self.min_price or close > self.max_price
                            or volume < self.min_volume or change_pct < self.min_change_pct):
                        continue
                    if self._is_excluded_symbol(symbol, 'NASDAQ'):
                        continue

                    candidates.append({
                        'symbol':     symbol.upper(),
                        'exchange':   'NASDAQ',
                        'price':      close,
                        'change_pct': float(change_pct),
                        'volume':     volume,
                        'high':       bar['high'],
                        'low':        bar['low'],
                        'open':       bar['open'],
                        'close':      close,
                        'source':     'yfinance_liquid_backfill',
                    })

            self.logger.info(
                f"  Scanned {scanned}/{len(liquid_stocks)} symbols, "
                f"{len(candidates)}/{target_pool} candidates so far..."
            )

            if len(candidates) >= target_pool:
                break

        candidates = sorted(candidates, key=lambda x: x['change_pct'], reverse=True)[:limit]
        self.logger.info(f"✅ Found {len(candidates)} from liquid stocks (backfill, date-pinned)")
        return candidates
    
    def _batch_verify_freshness(
        self, 
        candidates: List[Dict[str, Any]], 
        target_date: datetime,
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Enhanced debug version — shows exactly why each symbol passes or fails."""
        if not candidates:
            return []
        
        valid_candidates = []
        target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date

        if not self._is_live(target_date):
            # period='5d' below fetches the 5 days most recent to *right now*,
            # not the 5 days around target_date. For a historical backfill
            # that makes days_diff go negative (always <= 5), so it silently
            # passed and stamped today's OHLC onto old candidates — the same
            # class of bug fixed in _backfill_ohlc. Route backfills through
            # the date-pinned helper instead.
            return self._verify_freshness_for_date(candidates, target_date)

        symbols = [c['symbol'] for c in candidates]
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔍 FRESHNESS VALIDATION DEBUG - Target date: {target_date_obj}")
        self.logger.info("=" * 80)
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols    = symbols[i:i + batch_size]
            batch_candidates = candidates[i:i + batch_size]
            
            self.logger.info(f"📦 Batch {i//batch_size + 1}: Checking {len(batch_symbols)} symbols")
            
            try:
                data = self.rate_limiter.call_with_backoff(
                    yf.download,
                    batch_symbols,
                    period='5d',
                    interval='1d',
                    group_by='ticker',
                    progress=False,
                    threads=True,
                    auto_adjust=True,
                    label=f"freshness batch {i//batch_size + 1}",
                )
                
                if data.empty:
                    self.logger.warning(f"⚠️ No data returned for batch {i//batch_size + 1}")
                    continue
                
                for symbol, candidate in zip(batch_symbols, batch_candidates):
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"🔍 Checking: {symbol}")
                    
                    try:
                        if len(batch_symbols) == 1:
                            symbol_data = data
                        else:
                            if symbol not in data.columns.get_level_values(0):
                                self.logger.warning(f"  ❌ {symbol}: Not in response data")
                                continue
                            symbol_data = data[symbol]
                        
                        if symbol_data.empty:
                            self.logger.warning(f"  ❌ {symbol}: Empty data frame")
                            continue
                        
                        available_dates = symbol_data.index.date
                        self.logger.info(f"  📅 Available dates: {[str(d) for d in available_dates]}")
                        
                        last_date   = symbol_data.index[-1].date()
                        last_close  = symbol_data['Close'].iloc[-1]
                        last_volume = symbol_data['Volume'].iloc[-1]
                        
                        self.logger.info(f"  📊 Last trading day: {last_date}")
                        self.logger.info(f"  💰 Last close: {'NaN' if pd.isna(last_close) else f'${last_close:.2f}'}")
                        self.logger.info(f"  📈 Last volume: {'NaN' if pd.isna(last_volume) else f'{last_volume:,}'}")
                        
                        days_diff = (target_date_obj - last_date).days
                        self.logger.info(f"  ⏱️  Age: {days_diff} days old")
                        
                        if days_diff > 5:
                            self.logger.warning(f"  ❌ REJECTED: Data is {days_diff} days old (limit: 5)")
                            continue
                        if pd.isna(last_close) or pd.isna(last_volume):
                            self.logger.warning(f"  ❌ REJECTED: NaN close or volume")
                            continue
                        if last_volume < self.min_volume:
                            self.logger.warning(f"  ❌ REJECTED: volume {last_volume:,} < {self.min_volume:,}")
                            continue
                        
                        # ── Opportunistically fill OHLC from this validation fetch ──
                        row = symbol_data.iloc[-1]
                        for field, col in (('high', 'High'), ('low', 'Low'),
                                           ('open', 'Open'), ('close', 'Close')):
                            if candidate.get(field) is None and not pd.isna(row.get(col, float('nan'))):
                                candidate[field] = float(row[col])
                        
                        self.logger.info(f"  ✅✅ {symbol} PASSED ALL VALIDATION CHECKS")
                        valid_candidates.append(candidate)
                        
                    except Exception as e:
                        self.logger.error(f"  ❌ {symbol}: Error: {e}", exc_info=True)
                        continue
                
                if i + batch_size < len(symbols):
                    time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"❌ Batch {i//batch_size + 1} error: {e}", exc_info=True)
                continue
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"📊 VALIDATION SUMMARY: "
                         f"total={len(candidates)}, "
                         f"passed={len(valid_candidates)}, "
                         f"rejected={len(candidates)-len(valid_candidates)}")
        self.logger.info("=" * 80 + "\n")
        
        return valid_candidates

    def _verify_freshness_for_date(
        self,
        candidates: List[Dict[str, Any]],
        target_date: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Backfill-safe counterpart to _batch_verify_freshness. Validates and
        fills OHLC using _fetch_yf_bar_for_date so nothing gets compared
        against, or overwritten with, "whatever's most recent right now."
        """
        target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
        valid_candidates = []

        for candidate in candidates:
            symbol = candidate['symbol']
            bar = self._fetch_yf_bar_for_date(symbol, target_date)
            if bar is None:
                self.logger.warning(f"  ❌ {symbol}: no bar for {target_date_obj} — rejected")
                continue
            if bar['volume'] < self.min_volume:
                self.logger.warning(f"  ❌ {symbol}: volume {bar['volume']:,} < {self.min_volume:,}")
                continue

            for field, key in (('high', 'high'), ('low', 'low'),
                                ('open', 'open'), ('close', 'close')):
                if candidate.get(field) is None:
                    candidate[field] = bar[key]

            valid_candidates.append(candidate)

        self.logger.info(
            f"📊 VALIDATION SUMMARY (backfill, date-pinned): "
            f"total={len(candidates)}, passed={len(valid_candidates)}, "
            f"rejected={len(candidates)-len(valid_candidates)}"
        )
        return valid_candidates

    def _load_selection_counts(self) -> Dict[str, int]:
        if self.SELECTION_COUNTS_PATH.exists():
            try:
                with open(self.SELECTION_COUNTS_PATH) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load selection counts, starting fresh: {e}")
        return {}

    def _save_selection_counts(self) -> None:
        self.SELECTION_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SELECTION_COUNTS_PATH, "w") as f:
            json.dump(self._selection_counts, f, indent=2)

    def _record_selections(self, symbols: List[str]) -> None:
        if not symbols:
            return
        for s in symbols:
            self._selection_counts[s] = self._selection_counts.get(s, 0) + 1
        self._save_selection_counts()

    def _get_liquid_stocks_list(self) -> List[str]:
        """Broad, shuffled, non-overused symbol pool for backfill sampling.
        See daily_non_winners_detector.py's version of this method for the
        full rationale — same fix, same reasoning, applied here too."""
        if self._universe_cache is None:
            if not self.UNIVERSE_PATH.exists():
                raise FileNotFoundError(
                    f"Universe file not found at {self.UNIVERSE_PATH}. Refusing to "
                    f"fall back to a small hardcoded list. Populate a broad symbol "
                    f"list at that path first (see build_universe.py)."
                )
            df = pd.read_csv(self.UNIVERSE_PATH)
            self._universe_cache = sorted(set(df["symbol"].astype(str).str.upper().tolist()))
            self.logger.info(
                f"Loaded point-in-time universe of {len(self._universe_cache)} symbols "
                f"from {self.UNIVERSE_PATH}"
            )

        available = [
            s for s in self._universe_cache
            if self._selection_counts.get(s, 0) < self.MAX_USES_PER_SYMBOL
        ]
        random.shuffle(available)
        return available
