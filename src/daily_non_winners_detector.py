"""
Daily Non-Winners Detector
Finds stocks that did NOT explode - critical negative examples for ML training
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
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

        # Loosening config — edit these two values in config.yaml under non_winners:
        #   loosening_passes:    how many extra screener attempts before hard fallback
        #   loosening_step_pct:  how many percent to relax filters per pass
        non_winners_config = config.get("non_winners", {})
        self.loosening_passes   = int(non_winners_config.get("loosening_passes",   3))
        self.loosening_step_pct = float(non_winners_config.get("loosening_step_pct", 20.0))
        self.min_pool_factor    = float(non_winners_config.get("min_pool_factor",   1.5))

        # NOTE: learned_filters are loaded fresh on each detect_non_winners() call
        # so that changes to ml_models/learned_filters.json take effect immediately
        # without restarting the process.

        self.logger.info(
            f"Non-Winners detector initialized: "
            f"min_price={self.min_price}, min_volume={self.min_volume}, "
            f"loosening_passes={self.loosening_passes}, "
            f"loosening_step_pct={self.loosening_step_pct}%"
        )

    def _is_live(self, target_date: datetime) -> bool:
        """
        True if `target_date` is today (a live run), False if it's a
        historical backfill date.

        The TradingView screener/scanner endpoints only ever return *current*
        market data — they have no historical-date parameter. Using them for
        a backfill silently returns today's data mislabeled as the backfill
        date (the same class of bug fixed in _fetch_yf_bar_for_date below).
        Call sites use this to route backfill runs through the date-aware
        yfinance path exclusively.
        """
        if target_date is None:
            return True
        target = target_date.date() if isinstance(target_date, datetime) else target_date
        return target >= datetime.now().date()

    def _fetch_yf_bar_for_date(self, symbol: str, target_date: datetime) -> Optional[Dict[str, float]]:
        """
        Fetch the daily OHLCV bar for `symbol` on `target_date` specifically,
        plus derived metrics (change_pct, hv10, hv20, relative_volume_10d,
        atr14) computed from the trading days immediately preceding
        target_date — never from "whatever's most recent."

        FIX: The three call sites that used to inline
            ticker.history(period='2d', interval='1d') ... hist.iloc[-1]
        always returned whatever the MOST RECENT trading bar was at the
        moment the code ran — completely ignoring target_date. When this
        detector was run as part of a historical backfill (looping over many
        past detection_dates), every single one of those dates got patched
        with the same "latest" bar, silently duplicating one real snapshot
        across dozens of rows (confirmed via identical OHLCV values recurring
        under many different detection_date rows in the DB).

        This fetches an explicit window ending at target_date and returns
        None if that exact date isn't present (e.g. weekend/holiday/no data),
        rather than substituting a different day's bar. The window is wide
        enough (45 calendar days) to compute HV10/HV20/ATR14 — the same
        metrics the TradingView screener filters on — so backfill runs can
        apply the full learned-filter set instead of only price/volume.
        """
        target_date_only = target_date.date() if isinstance(target_date, datetime) else target_date
        # Need ~25 trading days of lookback for hv20/atr14; yfinance's
        # `end` is exclusive.
        start = target_date_only - timedelta(days=45)
        end = target_date_only + timedelta(days=1)

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start.isoformat(), end=end.isoformat(), interval='1d')
        except Exception:
            return None

        if hist.empty:
            return None

        idx_dates = hist.index.date
        target_mask = idx_dates == target_date_only
        if not target_mask.any():
            return None

        target_pos = int(np.where(target_mask)[0][0])
        if target_pos == 0:
            # No prior bar available in the lookback window to compute change_pct from.
            return None

        window = hist.iloc[:target_pos + 1]  # everything up to and including target_date
        latest = window.iloc[-1]
        previous = window.iloc[-2]

        close = float(latest['Close'])
        prev_close = float(previous['Close'])
        if prev_close == 0:
            return None

        result: Dict[str, float] = {
            'close':      close,
            'prev_close': prev_close,
            'change_pct': ((close - prev_close) / prev_close) * 100,
            'volume':     float(latest['Volume']),
            'high':       float(latest['High']),
            'low':        float(latest['Low']),
            'open':       float(latest['Open']),
        }

        # Historical volatility: annualized std of daily log returns.
        closes = window['Close']
        log_returns = np.log(closes / closes.shift(1)).dropna()
        for n, key in ((10, 'hv10'), (20, 'hv20')):
            if len(log_returns) >= n:
                result[key] = float(log_returns.tail(n).std() * np.sqrt(252) * 100)

        # Relative volume: target day's volume vs. average of the prior 10 days.
        volumes = window['Volume']
        if len(volumes) > 10:
            avg_vol = float(volumes.iloc[-11:-1].mean())
            if avg_vol > 0:
                result['relative_volume_10d'] = float(result['volume'] / avg_vol)

        # ATR(14): average true range over the last 14 days.
        if len(window) >= 15:
            highs, lows, closes_prev = window['High'], window['Low'], window['Close'].shift(1)
            tr = pd.concat([
                highs - lows,
                (highs - closes_prev).abs(),
                (lows - closes_prev).abs(),
            ], axis=1).max(axis=1)
            result['atr14'] = float(tr.tail(14).mean())

        return result

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
    
    def _loosen_filters(self, base_filters: dict, step_pct: float, pass_number: int) -> dict:
        """
        Return a new filter dict with thresholds loosened by (step_pct * pass_number) %.

        For every min_* key the threshold is reduced, giving more stocks a chance to
        pass.  For every max_* key the ceiling is raised for the same reason.

        Example: step_pct=20, pass_number=1  →  min values drop by 20 %, max values
        rise by 20 %.  pass_number=2 with the same step_pct drops/raises by 40 %, etc.

        Args:
            base_filters: The original learned filters (never mutated).
            step_pct:     How many percent to relax per pass (set in config under
                          non_winners.loosening_step_pct, default 20).
            pass_number:  Which pass this is (1-based so pass 0 = full filters).
        """
        lf = dict(base_filters)
        reduction = (step_pct / 100.0) * pass_number   # e.g. 0.20, 0.40, 0.60 …
        min_factor = max(0.0, 1.0 - reduction)          # can't go below 0
        max_factor = 1.0 + reduction                    # no upper bound needed

        for key in list(lf.keys()):
            if lf[key] is None or key.startswith("_"):
                continue
            if key.startswith("min_"):
                lf[key] = lf[key] * min_factor
            elif key.startswith("max_"):
                lf[key] = lf[key] * max_factor

        return lf

    def detect_non_winners(
        self, 
        top_n: int = 15, 
        target_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Detect non-winners (negative examples).

        Strategy:
        1. Load learned_filters fresh from disk (picks up any changes automatically).
        2. Apply filters to find a pool of stocks that look like winners but didn't
           gain ≥20 % intraday.
        3. If the pool is too small, loosen the filters by loosening_step_pct % per
           pass and retry up to loosening_passes times, accumulating candidates.
        4. Only after all passes are exhausted does it fall back to the original
           random-liquid-stock approach for any remaining slots.

        Tuning knobs (all in config.yaml under non_winners:):
            loosening_passes   – number of extra screener attempts (default 3)
            loosening_step_pct – % to relax filters per pass (default 20)
            min_pool_factor    – minimum pool = top_n * this (default 1.5)

        Args:
            top_n: Number of non-winners to collect
            target_date: Target date

        Returns:
            List of non-winner dictionaries
        """
        if target_date is None:
            target_date = datetime.now()

        target_date_str = target_date.date().isoformat()
        min_pool = max(top_n, int(top_n * self.min_pool_factor))

        # ── Load filters fresh every run ─────────────────────────────────────
        # This means edits to learned_filters.json are picked up automatically
        # without restarting anything.
        base_filters = self._load_learned_filters()

        self.logger.info(f"Detecting non-winners for {target_date_str}...")
        self.logger.info(
            f"Strategy: up to {self.loosening_passes} loosening pass(es) at "
            f"{self.loosening_step_pct}%/pass → need pool ≥ {min_pool}"
        )

        # Get actual winners to exclude
        winners_symbols = self._get_winners_symbols(target_date)
        self.logger.info(f"Found {len(winners_symbols)} winners to exclude")

        # ── Phase 1: progressive learned-filter candidates ───────────────────
        all_candidates: Dict[str, Dict] = {}   # symbol → data  (deduped across passes)
        passes_needed = 0

        # Pass 0 = full filters; passes 1…loosening_passes = progressively looser
        for pass_idx in range(self.loosening_passes + 1):
            if pass_idx == 0:
                active_filters = base_filters
                label = "full filters"
            else:
                active_filters = self._loosen_filters(
                    base_filters, self.loosening_step_pct, pass_idx
                )
                relaxed_pct = self.loosening_step_pct * pass_idx
                label = f"loosened {relaxed_pct:.0f}%"

            def _fmt(v, spec):
                return format(v, spec) if v is not None else "None"
            self.logger.info(
                f"Phase 1 pass {pass_idx} ({label}): fetching candidates "
                f"[min_price={_fmt(active_filters.get('min_price'), '.3f')}, "
                f"min_volume={_fmt(active_filters.get('min_volume'), '.0f')}, "
                f"min_rvol={_fmt(active_filters.get('min_relative_volume'), '.2f')}]..."
            )

            new_candidates = self._get_filtered_candidates(winners_symbols, active_filters, target_date)

            added = 0
            for c in new_candidates:
                if c["symbol"] not in all_candidates:
                    all_candidates[c["symbol"]] = c
                    added += 1

            passes_needed = pass_idx
            self.logger.info(
                f"  Pass {pass_idx}: +{added} new symbols "
                f"(cumulative pool: {len(all_candidates)})"
            )

            if len(all_candidates) >= min_pool:
                self.logger.info(
                    f"  Pool size {len(all_candidates)} ≥ {min_pool} — "
                    f"stopping after pass {pass_idx} ({label})"
                )
                break

        if passes_needed > 0:
            self.logger.info(
                f"  NOTE: needed {passes_needed} loosening pass(es) "
                f"(filters relaxed by {self.loosening_step_pct * passes_needed:.0f}% total) "
                f"to reach pool of {len(all_candidates)}"
            )

        filtered_candidates = list(all_candidates.values())
        self.logger.info(f"  Final learned-filter pool: {len(filtered_candidates)} stocks")

        non_winners: List[Dict] = []

        if filtered_candidates:
            filtered_map: Dict[str, Dict] = {c["symbol"]: c for c in filtered_candidates}
            filtered_syms = set(filtered_map.keys())

            def _pick_from_filtered(min_chg, max_chg, n):
                picked = []
                already = {nw["symbol"] for nw in non_winners}
                for sym, data in filtered_map.items():
                    if sym in already:
                        continue
                    if min_chg <= data["change_pct"] <= max_chg:
                        picked.append(data)
                    if len(picked) >= n:
                        break
                return picked

            flat_count        = int(top_n * 0.3)
            slight_gain_count = int(top_n * 0.3)
            slight_loss_count = int(top_n * 0.2)

            flat_stocks    = _pick_from_filtered(-2.0,  2.0,  flat_count)
            non_winners.extend(flat_stocks)
            self.logger.info(f"  Flat  (-2% to +2%):  {len(flat_stocks)}/{flat_count} from filtered pool")

            slight_gainers = _pick_from_filtered(2.0,  10.0, slight_gain_count)
            non_winners.extend(slight_gainers)
            self.logger.info(f"  Slight gain (+2% to +10%): {len(slight_gainers)}/{slight_gain_count} from filtered pool")

            slight_losers  = _pick_from_filtered(-10.0, -2.0, slight_loss_count)
            non_winners.extend(slight_losers)
            self.logger.info(f"  Slight loss (-2% to -10%): {len(slight_losers)}/{slight_loss_count} from filtered pool")

            big_loss_count = top_n - len(non_winners)
            big_losers     = _pick_from_filtered(-50.0, -10.0, big_loss_count)
            non_winners.extend(big_losers)
            self.logger.info(f"  Big loss (< -10%): {len(big_losers)}/{big_loss_count} from filtered pool")
        else:
            self.logger.info("  No filtered candidates returned from any pass — skipping Phase 1")
            filtered_syms = set()

        # ── Phase 2: fallback for any remaining shortage ─────────────────────
        if len(non_winners) < top_n:
            shortage = top_n - len(non_winners)
            self.logger.info(
                f"Phase 2 (hard fallback): Only {len(non_winners)}/{top_n} after all loosening passes. "
                f"Filling {shortage} slots via random-liquid-stock method..."
            )

            already_syms = {nw["symbol"] for nw in non_winners}
            per_cat_needed = max(shortage // 4, 1)

            fallback: List[Dict] = []
            for min_chg, max_chg in [(-2.0, 2.0), (2.0, 10.0), (-10.0, -2.0), (-50.0, -10.0)]:
                if len(fallback) >= shortage:
                    break
                batch = self._get_stocks_by_change_range(
                    target_date, min_chg, max_chg,
                    per_cat_needed * 2,
                    winners_symbols | already_syms | filtered_syms,
                )
                for stock in batch:
                    if stock["symbol"] not in already_syms and stock["symbol"] not in filtered_syms:
                        fallback.append(stock)
                        already_syms.add(stock["symbol"])
                    if len(fallback) >= shortage:
                        break

            if len(fallback) < shortage:
                remaining = shortage - len(fallback)
                self.logger.info(f"  Still {remaining} short — using fully random liquid stocks")
                random_stocks = self._get_random_liquid_stocks(
                    target_date, remaining * 2,
                    winners_symbols,
                    already_syms | filtered_syms,
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
    
    def _get_filtered_candidates(
        self,
        exclude_symbols: set,
        filters: dict = None,
        target_date: datetime = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch a pool of stocks that pass the given filters but did NOT gain ≥20 % intraday.

        Uses the TradingView screener when available (applying filter values as screener
        constraints), otherwise falls back to yfinance on the liquid-stocks list.
        In both cases only stocks with change_pct < 20 % are returned.

        Args:
            exclude_symbols: Symbols to exclude (winners + already-selected)
            filters: Filter dict to apply.  Defaults to self.learned_filters.
            target_date: Date the candidates are being collected for. Required
                for the yfinance fallback path to fetch the correct historical
                bar instead of "whatever is most recent."
        """
        if target_date is None:
            target_date = datetime.now()
        lf = filters if filters is not None else self.learned_filters

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

        # FIX: TradingView's screener has no historical-date parameter — it
        # only ever reflects the current market. Using it during a backfill
        # would silently label today's screen results as the backfill date.
        # Route backfill runs straight to the (now date-aware) yfinance path.
        if SCREENER_AVAILABLE and self.screener and self._is_live(target_date):
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
                bar = self._fetch_yf_bar_for_date(symbol, target_date)
                if bar is None:
                    continue

                change_pct = bar["change_pct"]
                close      = bar["close"]
                volume     = bar["volume"]

                if change_pct >= 20.0:
                    continue
                if close < min_price_filter:
                    continue
                if max_price_filter is not None and close > max_price_filter:
                    continue
                if volume < min_volume_filter:
                    continue

                # FIX: These filters used to be TradingView-screener-only —
                # a backfill run that skipped the screener silently dropped
                # them. Apply the same bounds here using the metrics computed
                # by _fetch_yf_bar_for_date, so filter behavior matches
                # between live and backfill runs.
                hv10 = bar.get("hv10")
                if lf.get("min_hv10") is not None and (hv10 is None or hv10 < lf["min_hv10"]):
                    continue
                if lf.get("max_hv10") is not None and (hv10 is None or hv10 > lf["max_hv10"]):
                    continue

                hv20 = bar.get("hv20")
                if lf.get("min_hv20") is not None and (hv20 is None or hv20 < lf["min_hv20"]):
                    continue
                if lf.get("max_hv20") is not None and (hv20 is None or hv20 > lf["max_hv20"]):
                    continue

                rel_vol = bar.get("relative_volume_10d")
                min_rel_vol = lf.get("min_relative_volume")
                if min_rel_vol is None:
                    min_rel_vol = lf.get("min_volume_ratio")
                if min_rel_vol is not None and (rel_vol is None or rel_vol < min_rel_vol):
                    continue

                atr14 = bar.get("atr14")
                if lf.get("min_atr14") is not None and (atr14 is None or atr14 < lf["min_atr14"]):
                    continue

                candidates.append({
                    "symbol":     symbol,
                    "exchange":   "NASDAQ",
                    "price":      float(close),
                    "change_pct": float(change_pct),
                    "volume":     int(volume),
                    "high":       float(bar["high"]),
                    "low":        float(bar["low"]),
                    "open":       float(bar["open"]),
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
            if SCREENER_AVAILABLE and self.screener and self._is_live(target_date):
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
                bar = self._fetch_yf_bar_for_date(symbol, target_date)
                if bar is None:
                    continue

                change_pct = bar['change_pct']
                close      = bar['close']
                volume     = bar['volume']

                if min_change <= change_pct <= max_change:
                    if close >= self.min_price and volume >= self.min_volume:
                        candidates.append({
                            'symbol': symbol,
                            'exchange': 'NASDAQ',
                            'price': float(close),
                            'change_pct': float(change_pct),
                            'volume': int(volume),
                            'high': float(bar['high']),
                            'low': float(bar['low']),
                            'open': float(bar['open']),
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
                bar = self._fetch_yf_bar_for_date(symbol, target_date)
                if bar is None:
                    continue

                change_pct = bar['change_pct']
                close      = bar['close']
                volume     = bar['volume']

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
                        'high': float(bar['high']),
                        'low': float(bar['low']),
                        'open': float(bar['open']),
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
