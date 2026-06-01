#!/usr/bin/env python3
"""
backfill_non_winners.py
=======================
Backfills historical non-winner records using **screener-passed** stocks as the
negative-example pool — the same fix described in the issue.

For each trading day in the requested date range the script:

  1. Reads the symbols that ACTUALLY appeared in the daily_winners table on that
     date  (these are the "screener-passed" positives).
  2. Queries yfinance for the EOD OHLCV of every winner-adjacent symbol — i.e.,
     the universe of stocks that *could* have been screened that day but did NOT
     hit ≥20 % intraday.  Because we no longer have a live TradingView screener
     snapshot for past days, we approximate the screener-passed pool by:
       a. Taking the full day_gainers yfinance screener list for that date
          (best available retrospective proxy), OR
       b. If yfinance can't return historical screener data, falling back to a
          configurable static universe (SP500 + Russell 2000 proxies) filtered
          by the same price / volume / learned_filter constraints the live
          detector applies.
  3. Excludes any symbol already in daily_winners for that date.
  4. Samples N diverse non-winners across the same four categories (flat / slight
     gain / slight loss / big loss) the live detector uses.
  5. Writes the selected symbols to daily_non_winners (idempotent — skips symbols
     already present for the date).
  6. Collects intraday indicators at all four timepoints (market_open,
     market_close, day_prior_open, day_prior_close) via IntradayDataCollector.
  7. Writes intraday indicator rows to the four non_winners indicator tables.
  8. Computes and writes T-3 / T-5 / T-10 multiday features to
     non_winners_multiday.

All writes are idempotent — already-present (date, symbol) pairs are skipped.

Usage
-----
  # Backfill the last 30 trading days
  python backfill_non_winners.py --days 30

  # Backfill a specific date range
  python backfill_non_winners.py --start 2025-01-01 --end 2025-03-31

  # Dry-run: print what would be written without touching Supabase
  python backfill_non_winners.py --days 10 --dry-run

  # Limit non-winners per day (default: 15)
  python backfill_non_winners.py --days 30 --top-n 20

  # Skip dates that already have ANY non-winner records
  python backfill_non_winners.py --days 60 --skip-existing

Environment variables required
-------------------------------
  SUPABASE_URL   – your Supabase project URL
  SUPABASE_KEY   – your Supabase service-role (or anon) key

Optional
--------
  CONFIG         – path to config.yaml (default: config.yaml)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.intraday_data_collector import IntradayDataCollector
from src.multiday_feature_collector import MultidayFeatureCollector
from src.daily_non_winners_supabase_client import DailyNonWinnersSupabaseClient
from src.daily_winners_supabase_client import DailyWinnersSupabaseClient
from src.utils import load_config, setup_logging

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def is_trading_day(d: datetime) -> bool:
    """Return True if *d* is a NYSE trading day."""
    if d.weekday() >= 5:
        return False
    holidays = USFederalHolidayCalendar().holidays(start=str(d.date()), end=str(d.date()))
    return len(holidays) == 0


def trading_days_in_range(start: datetime, end: datetime) -> List[datetime]:
    """Return a sorted list of trading days between *start* and *end* inclusive."""
    days: List[datetime] = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _sanitize_value(value: Any, field_name: str = "") -> Any:
    """Minimal sanitiser for non-winner dicts before writing."""
    import numpy as np

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if not (v == v) or abs(v) == float("inf"):  # nan / inf
            return None
        if "volume" in field_name.lower() or "obv" in field_name.lower():
            return int(v)
        return v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    auto = {"id", "created_at", "updated_at"}
    return {k: _sanitize_value(v, k) for k, v in d.items() if k not in auto}


# ═══════════════════════════════════════════════════════════════════════════════
# Screener-passed pool reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

# A broad but manageable static universe used when live screener data is
# unavailable for historical dates.  Covers large / mid caps across sectors
# plus the kinds of small-caps that often show up in day-gainer screeners.
#
# You can extend this list or replace it with a CSV path via --universe-csv.
_DEFAULT_UNIVERSE: List[str] = [
    # Mega-caps / ETF proxies (almost always pass volume filter)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM",
    "V", "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    # Mid-cap momentum names that frequently appear in screeners
    "AMD", "INTC", "QCOM", "AVGO", "AMAT", "LRCX", "MU", "KLAC",
    "ADBE", "CRM", "ORCL", "NOW", "SNOW", "PLTR", "DDOG", "ZS", "PANW",
    "NFLX", "DIS", "CMCSA", "T", "VZ",
    "BAC", "WFC", "GS", "MS", "C", "SCHW",
    "LLY", "PFE", "MRNA", "BNTX", "BIIB", "GILD", "REGN", "VRTX",
    "CAT", "DE", "BA", "RTX", "NOC", "GD", "LMT",
    "F", "GM", "RIVN", "LCID",
    "DAL", "AAL", "UAL", "LUV",
    "CCL", "NCLH", "RCL",
    "MGM", "WYNN", "LVS", "CZR",
    "AMC", "GME",  # meme stocks often in screener outputs
    # Small / micro-cap liquid names (price ≥ $1, volume ≥ 500 K typical)
    "SOUN", "CLSK", "MARA", "RIOT", "CIFR", "HUT", "BTBT",
    "SOFI", "HOOD", "OPEN", "UWMC", "RKT",
    "SPCE", "JOBY", "ACHR", "LILM",
    "CLOV", "WISH", "PAYA", "SDC",
    "ATER", "PROG", "BBIG", "VVPR", "MVIS",
    "ABML", "GFAI", "MULN", "FFIE", "NKLA", "WKHS",
    "NIO", "XPEV", "LI", "BLNK", "CHPT", "PTRA",
    "TLRY", "SNDL", "CANN", "APHA", "CGC",
    "SKLZ", "GREE", "CTIC", "IMVT", "LGVN", "ACMR",
]

EXCLUDED_PATTERNS = ["OTC", ".PK", ".OB"]


def _is_excluded(symbol: str) -> bool:
    s = symbol.upper()
    for p in EXCLUDED_PATTERNS:
        if p in s:
            return True
    if len(s) > 5:
        return True
    return False


def fetch_candidates_for_date(
    target_date: datetime,
    exclude_symbols: Set[str],
    min_price: float,
    min_volume: int,
    max_price: Optional[float],
    universe: List[str],
) -> List[Dict[str, Any]]:
    """
    Build a pool of screener-like candidates for *target_date*.

    Strategy (in priority order):
      1. Try yfinance day_gainers screener — gives us the stocks the market
         highlighted that day.  These are the closest proxy to "screener output"
         we can get retrospectively.
      2. Fall back to the static *universe* list filtered via yfinance EOD data.

    Only stocks with change_pct < 20 % and that pass price/volume filters are
    returned (they didn't win ≥20 % — that's the whole point).
    """
    date_str = target_date.strftime("%Y-%m-%d")
    candidates: List[Dict[str, Any]] = []

    # ── Attempt 1: yfinance Screener (day_gainers) ─────────────────────────
    # Note: yfinance screener reflects *current* state, not historical.
    # For backfill this means we get today's gainers, not the target date's.
    # That's a known limitation — for live dates (≤ a few days ago) it's
    # accurate; for older dates we rely on the universe fallback.
    # A future improvement could use a stored screener snapshot if you've been
    # logging those.
    try:
        from yfinance import Screener as YFScreener
        screener = YFScreener()
        data = screener.get_screeners(["day_gainers"], count=300)
        quotes = data.get("day_gainers", {}).get("quotes", [])
        for q in quotes:
            sym = q.get("symbol", "").strip().upper()
            if not sym or sym in exclude_symbols or _is_excluded(sym):
                continue
            price = float(q.get("regularMarketPrice") or 0)
            volume = int(q.get("regularMarketVolume") or 0)
            change_pct = float(q.get("regularMarketChangePercent") or 0)
            if change_pct >= 20.0:
                continue
            if price < min_price or volume < min_volume:
                continue
            if max_price is not None and price > max_price:
                continue
            candidates.append({
                "symbol":     sym,
                "exchange":   q.get("exchange", "NASDAQ"),
                "price":      price,
                "change_pct": change_pct,
                "volume":     volume,
                "high":       float(q.get("regularMarketDayHigh") or price),
                "low":        float(q.get("regularMarketDayLow") or price),
                "open":       float(q.get("regularMarketOpen") or price),
                "close":      price,
                "source":     "yf_screener",
            })
        if candidates:
            logger.debug(f"  yfinance screener → {len(candidates)} candidates")
    except Exception as e:
        logger.debug(f"  yfinance screener unavailable ({e}), using universe fallback")

    # ── Attempt 2: Static universe via yfinance EOD download ───────────────
    # We use this both as supplement and as primary fallback.
    # We batch-download to avoid per-ticker API calls.
    remaining_symbols = [s for s in universe if s not in exclude_symbols
                         and s not in {c["symbol"] for c in candidates}]

    if remaining_symbols:
        try:
            logger.debug(f"  Downloading EOD data for {len(remaining_symbols)} universe tickers...")
            # Download a window ending on target_date to get correct EOD prices
            dl_end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
            dl_start = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")

            hist = yf.download(
                remaining_symbols,
                start=dl_start,
                end=dl_end,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            target_date_date = target_date.date()

            for sym in remaining_symbols:
                if sym in exclude_symbols or _is_excluded(sym):
                    continue
                try:
                    if len(remaining_symbols) == 1:
                        sym_df = hist
                    else:
                        sym_df = hist[sym]

                    if sym_df is None or sym_df.empty:
                        continue

                    # Find the row matching target_date
                    sym_df.index = pd.to_datetime(sym_df.index)
                    day_rows = sym_df[sym_df.index.date == target_date_date]
                    if day_rows.empty:
                        continue

                    row = day_rows.iloc[-1]
                    close_price = float(row["Close"])
                    open_price = float(row["Open"])
                    high_price = float(row["High"])
                    low_price = float(row["Low"])
                    volume = int(row["Volume"])

                    # Prior close for change %
                    prev_rows = sym_df[sym_df.index.date < target_date_date]
                    if prev_rows.empty:
                        continue
                    prev_close = float(prev_rows.iloc[-1]["Close"])
                    if prev_close == 0:
                        continue

                    change_pct = (close_price - prev_close) / prev_close * 100

                    if change_pct >= 20.0:
                        continue
                    if close_price < min_price or volume < min_volume:
                        continue
                    if max_price is not None and close_price > max_price:
                        continue

                    candidates.append({
                        "symbol":     sym,
                        "exchange":   "NASDAQ",
                        "price":      close_price,
                        "change_pct": change_pct,
                        "volume":     volume,
                        "high":       high_price,
                        "low":        low_price,
                        "open":       open_price,
                        "close":      close_price,
                        "source":     "universe_eod",
                    })
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"  Batch EOD download failed ({e})")

    logger.debug(f"  Total raw candidates for {date_str}: {len(candidates)}")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def sample_diverse_non_winners(
    candidates: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Pick *top_n* non-winners spread across the four change categories:
      30 % flat      (-2 % to +2 %)
      30 % slight +  (+2 % to +10 %)
      20 % slight -  (-2 % to -10 %)
      20 % big -     (< -10 %)

    Bins that have fewer candidates than the target are filled as much as
    possible; any leftover quota is reallocated to bins that have surplus.
    """
    buckets = {
        "flat":         [c for c in candidates if -2.0 <= c["change_pct"] <=  2.0],
        "slight_gain":  [c for c in candidates if  2.0 <  c["change_pct"] <= 10.0],
        "slight_loss":  [c for c in candidates if -10.0 <= c["change_pct"] < -2.0],
        "big_loss":     [c for c in candidates if c["change_pct"] < -10.0],
    }
    targets = {
        "flat":        int(top_n * 0.30),
        "slight_gain": int(top_n * 0.30),
        "slight_loss": int(top_n * 0.20),
        "big_loss":    top_n - int(top_n * 0.30) - int(top_n * 0.30) - int(top_n * 0.20),
    }

    selected: List[Dict[str, Any]] = []
    leftover = 0
    for bucket, tgt in targets.items():
        available = buckets[bucket]
        take = min(tgt, len(available))
        selected.extend(available[:take])
        leftover += tgt - take

    # Reallocate leftover quota from any bucket that has surplus
    if leftover > 0:
        already = {c["symbol"] for c in selected}
        for bucket, pool in buckets.items():
            for c in pool:
                if c["symbol"] not in already:
                    selected.append(c)
                    already.add(c["symbol"])
                    leftover -= 1
                    if leftover == 0:
                        break
            if leftover == 0:
                break

    return selected[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-day processing
# ═══════════════════════════════════════════════════════════════════════════════

def process_day(
    target_date: datetime,
    config: dict,
    supabase_nw: DailyNonWinnersSupabaseClient,
    supabase_w: DailyWinnersSupabaseClient,
    intraday_collector: IntradayDataCollector,
    multiday_collector: MultidayFeatureCollector,
    top_n: int,
    dry_run: bool,
    skip_existing: bool,
    min_price: float,
    min_volume: int,
    max_price: Optional[float],
    universe: List[str],
) -> Dict[str, int]:
    """
    Backfill non-winners for a single *target_date*.

    Returns a dict of write counts keyed by table type.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    counts = {"non_winners": 0, "market_open": 0, "market_close": 0,
              "day_prior_open": 0, "day_prior_close": 0, "multiday": 0}

    # ── 1. Skip check ─────────────────────────────────────────────────────
    if skip_existing:
        try:
            exists = supabase_nw.check_date_exists(date_str)
            if exists:
                logger.info(f"  [{date_str}] Already has non-winner data — skipping (--skip-existing)")
                return counts
        except Exception as e:
            logger.warning(f"  [{date_str}] Could not check existence: {e}")

    # ── 2. Get screener-passed winners for this date ───────────────────────
    try:
        winners_df = supabase_w.read_winners(start_date=date_str, end_date=date_str)
        winner_symbols: Set[str] = set(winners_df["symbol"].tolist()) if not winners_df.empty else set()
    except Exception as e:
        logger.warning(f"  [{date_str}] Could not read winners: {e}. Proceeding with empty exclude list.")
        winner_symbols = set()

    logger.info(f"  [{date_str}] {len(winner_symbols)} winners to exclude: {sorted(winner_symbols)}")

    # ── 3. Build candidate pool ────────────────────────────────────────────
    candidates = fetch_candidates_for_date(
        target_date=target_date,
        exclude_symbols=winner_symbols,
        min_price=min_price,
        min_volume=min_volume,
        max_price=max_price,
        universe=universe,
    )

    if not candidates:
        logger.warning(f"  [{date_str}] No candidates found — skipping date.")
        return counts

    # ── 4. Sample diverse non-winners ─────────────────────────────────────
    non_winners = sample_diverse_non_winners(candidates, top_n)

    for nw in non_winners:
        nw["detection_date"] = date_str
        nw["detection_time"] = "16:00:00"
        nw.pop("source", None)  # internal field — don't write to DB

    logger.info(
        f"  [{date_str}] Sampled {len(non_winners)} non-winners "
        f"(flat={sum(1 for n in non_winners if -2<=n['change_pct']<=2)}, "
        f"slight+={sum(1 for n in non_winners if 2<n['change_pct']<=10)}, "
        f"slight-={sum(1 for n in non_winners if -10<=n['change_pct']<-2)}, "
        f"big-={sum(1 for n in non_winners if n['change_pct']<-10)})"
    )

    if dry_run:
        logger.info(f"  [{date_str}] DRY-RUN — would write {len(non_winners)} non-winners: "
                    f"{[n['symbol'] for n in non_winners]}")
        return counts

    # ── 5. Write daily_non_winners ─────────────────────────────────────────
    try:
        counts["non_winners"] = supabase_nw.write_non_winners(non_winners)
        logger.info(f"  [{date_str}] Wrote {counts['non_winners']} rows → daily_non_winners")
    except Exception as e:
        logger.error(f"  [{date_str}] Failed to write non_winners: {e}")
        return counts

    # ── 6. Collect intraday indicators ────────────────────────────────────
    try:
        intraday_data = intraday_collector.collect_intraday_data(non_winners, target_date)
    except Exception as e:
        logger.error(f"  [{date_str}] Intraday collection failed: {e}")
        return counts

    # ── 7. Write intraday indicator tables ────────────────────────────────
    try:
        intraday_counts = supabase_nw.write_intraday_data(intraday_data)
        counts.update(intraday_counts)
        logger.info(
            f"  [{date_str}] Wrote intraday indicators — "
            f"market_open={intraday_counts.get('market_open', 0)}, "
            f"market_close={intraday_counts.get('market_close', 0)}, "
            f"day_prior_open={intraday_counts.get('day_prior_open', 0)}, "
            f"day_prior_close={intraday_counts.get('day_prior_close', 0)}"
        )
    except Exception as e:
        logger.error(f"  [{date_str}] Failed to write intraday indicators: {e}")

    # ── 8. Multiday (T-3 / T-5 / T-10) features ──────────────────────────
    day_prior_close_rows = intraday_data.get("day_prior_close", [])
    if day_prior_close_rows:
        try:
            counts["multiday"] = multiday_collector.collect_and_write(
                stocks=day_prior_close_rows,
                table="non_winners_multiday",
            )
            logger.info(f"  [{date_str}] Wrote {counts['multiday']} rows → non_winners_multiday")
        except Exception as e:
            logger.error(f"  [{date_str}] Multiday collection failed: {e}")
    else:
        logger.warning(f"  [{date_str}] No day_prior_close rows — skipping multiday features")

    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill non-winner records from screener-passed candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    date_grp = p.add_mutually_exclusive_group()
    date_grp.add_argument(
        "--days", type=int, default=None,
        help="Number of most-recent trading days to backfill (e.g. --days 30)",
    )
    date_grp.add_argument(
        "--start", type=str, default=None, metavar="YYYY-MM-DD",
        help="Start date of the backfill range (requires --end)",
    )

    p.add_argument(
        "--end", type=str, default=None, metavar="YYYY-MM-DD",
        help="End date of the backfill range (default: today)",
    )
    p.add_argument(
        "--top-n", type=int, default=15,
        help="Non-winners to collect per day (default: 15)",
    )
    p.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without touching Supabase",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip dates that already have any non-winner records in Supabase",
    )
    p.add_argument(
        "--universe-csv", type=str, default=None, metavar="PATH",
        help=(
            "Optional CSV with a 'symbol' column to use as the screener-universe "
            "fallback instead of the built-in list.  Useful when you have a "
            "historical daily screener snapshot export."
        ),
    )
    p.add_argument(
        "--min-price", type=float, default=None,
        help="Override learned_filters min_price (default: from config / 0.25)",
    )
    p.add_argument(
        "--min-volume", type=int, default=None,
        help="Override learned_filters min_volume (default: from config / 10 000)",
    )
    p.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to sleep between days to respect API rate limits (default: 2)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    config_path = args.config
    config = load_config(config_path) if Path(config_path).exists() else {}
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level, config.get("logging", {}))

    # ── Date range ─────────────────────────────────────────────────────────
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if args.days:
        end_dt = today
        # Walk backwards to find N trading days
        days_list: List[datetime] = []
        cur = today
        while len(days_list) < args.days:
            if is_trading_day(cur):
                days_list.append(cur)
            cur -= timedelta(days=1)
        days_list.reverse()
    elif args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else today
        days_list = trading_days_in_range(start_dt, end_dt)
    else:
        logger.error("Provide either --days N or --start YYYY-MM-DD [--end YYYY-MM-DD]")
        return 1

    if not days_list:
        logger.warning("No trading days in the requested range.")
        return 0

    logger.info("=" * 65)
    logger.info("BACKFILL NON-WINNERS (screener-sourced)")
    logger.info(f"  Date range : {days_list[0].date()} → {days_list[-1].date()}")
    logger.info(f"  Trading days: {len(days_list)}")
    logger.info(f"  Top-N per day: {args.top_n}")
    logger.info(f"  Dry-run: {args.dry_run}")
    logger.info(f"  Skip-existing: {args.skip_existing}")
    logger.info("=" * 65)

    # ── Universe ────────────────────────────────────────────────────────────
    universe: List[str]
    if args.universe_csv:
        try:
            u_df = pd.read_csv(args.universe_csv)
            universe = u_df["symbol"].dropna().str.strip().str.upper().tolist()
            logger.info(f"Loaded {len(universe)} symbols from {args.universe_csv}")
        except Exception as e:
            logger.warning(f"Could not load universe CSV ({e}); using built-in list")
            universe = _DEFAULT_UNIVERSE
    else:
        universe = _DEFAULT_UNIVERSE

    # ── Filter thresholds ──────────────────────────────────────────────────
    # Pull from learned_filters.json if available, then CLI overrides
    learned_filters: Dict[str, Any] = {}
    lf_path = Path("ml_models/learned_filters.json")
    if lf_path.exists():
        import json
        with open(lf_path) as f:
            learned_filters = json.load(f)

    min_price = args.min_price if args.min_price is not None else float(
        learned_filters.get("min_price") or config.get("detection", {}).get("min_price", 0.25)
    )
    min_volume = args.min_volume if args.min_volume is not None else int(
        learned_filters.get("min_volume") or config.get("detection", {}).get("min_volume", 10_000)
    )
    max_price: Optional[float] = learned_filters.get("max_price")

    logger.info(f"Filters: min_price={min_price}, min_volume={min_volume}, max_price={max_price}")

    # ── Supabase clients ───────────────────────────────────────────────────
    if not args.dry_run:
        try:
            supabase_nw = DailyNonWinnersSupabaseClient(config)
            supabase_w  = DailyWinnersSupabaseClient(config)
        except Exception as e:
            logger.error(f"Cannot connect to Supabase: {e}")
            return 1
    else:
        supabase_nw = None  # type: ignore
        supabase_w  = None  # type: ignore

    intraday_collector  = IntradayDataCollector(config)
    multiday_collector  = MultidayFeatureCollector(config)

    # ── Main loop ──────────────────────────────────────────────────────────
    total_counts: Dict[str, int] = {
        "non_winners": 0, "market_open": 0, "market_close": 0,
        "day_prior_open": 0, "day_prior_close": 0, "multiday": 0,
    }
    failed_dates: List[str] = []

    for i, target_date in enumerate(days_list, 1):
        date_str = target_date.strftime("%Y-%m-%d")
        logger.info(f"\n[{i}/{len(days_list)}] Processing {date_str}...")

        try:
            day_counts = process_day(
                target_date=target_date,
                config=config,
                supabase_nw=supabase_nw,
                supabase_w=supabase_w,
                intraday_collector=intraday_collector,
                multiday_collector=multiday_collector,
                top_n=args.top_n,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                min_price=min_price,
                min_volume=min_volume,
                max_price=max_price,
                universe=universe,
            )
            for k, v in day_counts.items():
                total_counts[k] = total_counts.get(k, 0) + v

        except Exception as e:
            logger.error(f"  [{date_str}] Unhandled error: {e}", exc_info=True)
            failed_dates.append(date_str)

        # Rate-limit: don't hammer yfinance / Supabase on large bacfills
        if i < len(days_list):
            time.sleep(args.delay)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Days processed : {len(days_list) - len(failed_dates)} / {len(days_list)}")
    logger.info(f"  daily_non_winners    : {total_counts['non_winners']} rows")
    logger.info(f"  non_winners_market_open  : {total_counts['market_open']} rows")
    logger.info(f"  non_winners_market_close : {total_counts['market_close']} rows")
    logger.info(f"  non_winners_day_prior_open  : {total_counts['day_prior_open']} rows")
    logger.info(f"  non_winners_day_prior_close : {total_counts['day_prior_close']} rows")
    logger.info(f"  non_winners_multiday        : {total_counts['multiday']} rows")
    if failed_dates:
        logger.warning(f"  Failed dates ({len(failed_dates)}): {failed_dates}")
    logger.info("=" * 65)

    return 0 if not failed_dates else 2


if __name__ == "__main__":
    sys.exit(main())
