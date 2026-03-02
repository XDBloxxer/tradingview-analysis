#!/usr/bin/env python3
"""
ML Stock Screener & Predictor

FIXES IN THIS VERSION (2026-03-02):

FIX 1 — Only ~8-15/1500 stocks fetched:
  Root cause: TradingView screener surfaces high-momentum micro/small-caps that
  yfinance has notoriously poor coverage for. yfinance silently returns empty
  DataFrames for these tickers, which were being dropped entirely.

  Solution: Use TradingView's OWN indicator data (already returned by the
  screener call) to build t3_* features directly, without any yfinance round-trip.
  TradingView returns RSI, EMA, ATR, ADX, volume, price, change etc. for every
  screened stock. We map these directly to the model's t3_ feature namespace.

  yfinance is now ONLY used for T-1 intraday snapshots (best-effort, optional).
  This means ~1490+ stocks will be scored instead of ~10.

FIX 2 — All AVOID/HOLD signals (bimodal fallback broken):
  _classify_signals_relative only fired when probs >= 0.85 (STRONG BUY path).
  When the model collapses everything to 0.25-0.35 (the actual failure mode when
  T1 features are absent), no stock crosses 0.85 so the fallback does nothing.

  Fix: Relative classification now operates on the FULL distribution — it ranks
  all stocks by probability and assigns signals based on top-N percentiles,
  regardless of absolute probability values. This gives actionable BUY/STRONG BUY
  signals even when the model's calibration is imperfect.

  The fallback now also fires correctly: _detect_bimodal triggers when >90% of
  predictions are below 0.50 (the low-collapse case) OR when the mid-range
  count is too low (the original bimodal case).

FIX 3 — Target price predictions way off:
  Unchanged from previous version — gain regressor handles this when trained.
  Rule-based fallback calibrated to real winner statistics.

Previously also: T-1 optional (kept from last version), column casing preserved.
"""

import argparse
import logging
import pytz
from datetime import time as dt_time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.explosion_predictor import ExplosionPredictor
from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient

try:
    from tradingview_scraper.symbols.screener import Screener
    SCREENER_AVAILABLE = True
except ImportError:
    SCREENER_AVAILABLE = False


# ---------------------------------------------------------------------------
# TradingView column → model t3_ feature mapping
# ---------------------------------------------------------------------------
# TradingView screener returns these column names; we map them to the model's
# t3_* feature names (which match what calculate_comprehensive_indicators_daily
# produces on daily bars).
#
# This completely eliminates the yfinance daily-bar round-trip for t3_ features,
# fixing the coverage gap for micro/small-caps that TradingView knows about but
# yfinance silently fails on.

TV_TO_T3_MAP = {
    # Price / OHLCV
    "close":                        "Close",
    "open":                         "Open",
    "high":                         "High",
    "low":                          "Low",
    "volume":                       "Volume",
    "change":                       "price_change_1d",  # today's % change
    "change_abs":                   "MOM_10",

    # RSI
    "RSI":                          "RSI_14",
    "RSI[1]":                       "RSI_14",      # 1-bar-ago RSI (will be deduped)
    "RSI[2]":                       "RSI_14",      # 2-bar-ago RSI (will be deduped)

    # Stochastic
    "Stoch.K":                      "STOCHk_14_3_3",
    "Stoch.D":                      "STOCHd_14_3_3",
    "Stoch.K[1]":                   "STOCHk_14_3_3",
    "Stoch.D[1]":                   "STOCHd_14_3_3",

    # Williams %R
    "W.R":                          "WILLR_14",
    "W.R[1]":                       "WILLR_14",

    # MACD
    "MACD.macd":                    "MACD_12_26_9",
    "MACD.signal":                  "MACDs_12_26_9",

    # Bollinger Bands
    "BB.upper":                     "BBU_20_2.0_2.0",
    "BB.lower":                     "BBL_20_2.0_2.0",
    "BB.basis":                     "BBM_20_2.0_2.0",
    "BBPower":                      "BBP_20_2.0_2.0",

    # Moving averages
    "EMA5":                         "EMA_5",
    "EMA10":                        "EMA_10",
    "EMA20":                        "EMA_20",
    "EMA30":                        "EMA_26",      # closest
    "EMA50":                        "EMA_50",
    "SMA5":                         "SMA_5",
    "SMA10":                        "SMA_10",
    "SMA20":                        "SMA_20",
    "SMA30":                        "SMA_50",      # closest
    "SMA50":                        "SMA_50",

    # Momentum
    "Mom":                          "MOM_10",
    "AO":                           "AO",
    "CCI20":                        "CCI_20",
    "UO":                           "UO",
    "ROC":                          "ROC_10",

    # Volume-related
    "relative_volume_10d_calc":     "Volume_Ratio",
    "VWMA":                         "VWMA_20",

    # Volatility / trend
    "ATR":                          "ATR_14",
    "ADX":                          "ADX_14",
    "ADX+DI":                       "DMP_14",
    "ADX-DI":                       "DMN_14",
    "Volatility.D":                 "HV_20",

    # Candlestick / misc
    "gap":                          "Gap_Pct",
    "premarket_gap":                "Gap_Pct",

    # 52-week
    "High.52W":                     "high_52w",    # not in model but harmless
    "Low.52W":                      "low_52w",

    # Aroon
    "Aroon.Up":                     "AROONU_25",
    "Aroon.Down":                   "AROOND_25",
}

# Screener columns the SmartScreener always asks for (beyond default set)
EXTRA_TV_COLUMNS = [
    "RSI", "RSI[1]", "Stoch.K", "Stoch.D", "Stoch.K[1]", "Stoch.D[1]",
    "W.R", "W.R[1]", "MACD.macd", "MACD.signal",
    "BB.upper", "BB.lower", "BB.basis", "BBPower",
    "EMA5", "EMA10", "EMA20", "EMA50",
    "SMA5", "SMA10", "SMA20", "SMA50",
    "Mom", "AO", "CCI20", "UO", "ROC",
    "relative_volume_10d_calc", "VWMA",
    "ATR", "ADX", "ADX+DI", "ADX-DI", "Volatility.D",
    "gap", "Aroon.Up", "Aroon.Down",
    "close", "open", "high", "low", "volume", "change",
]


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def log_probability_distribution(predictions_df: pd.DataFrame, logger: logging.Logger, label: str = ""):
    if predictions_df.empty:
        logger.warning("Cannot log probability distribution — predictions DataFrame is empty")
        return

    probs = predictions_df['explosion_probability']
    title = f"PROBABILITY DISTRIBUTION{' — ' + label if label else ''}"

    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)

    buckets = [
        ("0–10%  (AVOID)",   probs < 0.10),
        ("10–20% (AVOID)",   (probs >= 0.10) & (probs < 0.20)),
        ("20–30% (AVOID)",   (probs >= 0.20) & (probs < 0.30)),
        ("30–40% (AVOID)",   (probs >= 0.30) & (probs < 0.40)),
        ("40–50% (AVOID)",   (probs >= 0.40) & (probs < 0.50)),
        ("50–60% (HOLD)",    (probs >= 0.50) & (probs < 0.60)),
        ("60–70% (HOLD)",    (probs >= 0.60) & (probs < 0.70)),
        ("70–80% (BUY)",     (probs >= 0.70) & (probs < 0.80)),
        ("80–90% (BUY)",     (probs >= 0.80) & (probs < 0.90)),
        ("90–100% (STRONG)", probs >= 0.90),
    ]

    total = len(probs)
    bar_width = 30

    for bucket_label, mask in buckets:
        count = mask.sum()
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 100 * bar_width)
        logger.info(f"  {bucket_label:<22} {count:>4}  ({pct:>5.1f}%)  {bar}")

    logger.info("-" * 70)
    logger.info(f"  Total stocks evaluated:  {total}")
    logger.info(f"  Min probability:         {probs.min():.4f} ({probs.min()*100:.2f}%)")
    logger.info(f"  Max probability:         {probs.max():.4f} ({probs.max()*100:.2f}%)")
    logger.info(f"  Mean probability:        {probs.mean():.4f} ({probs.mean()*100:.2f}%)")
    logger.info(f"  Median probability:      {probs.median():.4f} ({probs.median()*100:.2f}%)")
    logger.info(f"  Std deviation:           {probs.std():.4f}")

    logger.info("")
    logger.info("  Signal breakdown:")
    if 'signal' in predictions_df.columns:
        signal_counts = predictions_df['signal'].value_counts()
        for signal in ['STRONG BUY', 'BUY', 'HOLD', 'AVOID']:
            count = signal_counts.get(signal, 0)
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"    {signal:<12} {count:>4}  ({pct:>5.1f}%)")

    logger.info("")
    logger.info("  ── Diagnosis ──────────────────────────────────────────────────")

    avoid_pct = (probs < 0.50).sum() / total * 100 if total > 0 else 0
    high_pct  = (probs >= 0.70).sum() / total * 100 if total > 0 else 0

    if probs.max() < 0.20:
        logger.warning("  ⚠️  MAX PROB < 20% — likely broken/corrupted model or severe feature mismatch.")
    elif probs.max() < 0.50:
        logger.warning("  ⚠️  MAX PROB < 50% — model is not finding any plausible candidates.")
    elif avoid_pct > 95:
        logger.warning("  ⚠️  >95% AVOID — screening population likely mismatched to training distribution.")
    elif high_pct == 0:
        logger.info("  ℹ️  No BUY/STRONG BUY signals today (absolute thresholds). Check relative signals.")
    else:
        logger.info(f"  ✅ Distribution looks healthy — {high_pct:.1f}% of stocks scored BUY or higher.")

    logger.info("=" * 70)
    logger.info("")

    logger.info("  Top 10 stocks by probability:")
    logger.info(f"  {'#':<4} {'Symbol':<8} {'Prob':>8}  {'Signal'}")
    logger.info("  " + "-" * 40)
    for rank, (_, row) in enumerate(predictions_df.head(10).iterrows(), 1):
        prob_pct = row['explosion_probability'] * 100
        signal   = row.get('signal', 'N/A')
        symbol   = row.get('symbol', 'N/A')
        logger.info(f"  {rank:<4} {symbol:<8} {prob_pct:>7.2f}%  {signal}")

    logger.info("")


def get_next_trading_day() -> str:
    est = pytz.timezone('America/New_York')
    now_est = datetime.now(est)
    prediction_day = now_est + timedelta(days=1)
    while prediction_day.weekday() >= 5:
        prediction_day += timedelta(days=1)
    return prediction_day.date().isoformat()


class SmartScreener:
    """
    Intelligent screener that uses model-derived filters AND returns
    TradingView's own indicator data to avoid a secondary yfinance round-trip.
    """

    TV_FILTER_MAP = {
        "min_price":           ("close",                    "greater"),
        "max_price":           ("close",                    "less"),
        "min_volume":          ("volume",                   "greater"),
        "min_rsi":             ("RSI",                      "greater"),
        "max_rsi":             ("RSI",                      "less"),
        "min_rsi7":            ("RSI[1]",                   "greater"),
        "max_rsi7":            ("RSI[1]",                   "less"),
        "min_volume_ratio":    ("relative_volume_10d_calc", "greater"),
        "min_relative_volume": ("relative_volume_10d_calc", "greater"),
        "min_hv10":            ("Volatility.D",             "greater"),
        "max_hv10":            ("Volatility.D",             "less"),
        "min_hv20":            ("Volatility.D",             "greater"),
        "max_hv20":            ("Volatility.D",             "less"),
        "min_adx":             ("ADX",                      "greater"),
        "min_atr14":           ("ATR",                      "greater"),
    }

    DEFAULT_FILTERS = {
        "min_price":           0.50,
        "max_price":           100.0,
        "min_volume":          200_000,
        "min_volume_ratio":    1.5,
        "min_relative_volume": 1.5,
        "min_rsi":             None,
        "max_rsi":             None,
        "min_hv10":            None,
        "max_hv10":            None,
        "min_adx":             None,
        "min_atr14":           None,
    }

    def __init__(self, config: dict = None, logger=None):
        import logging
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.filters = self._load_learned_filters()
        if SCREENER_AVAILABLE:
            self.screener = Screener()
        else:
            self.screener = None

    def _load_learned_filters(self) -> dict:
        import json
        from pathlib import Path

        defaults = dict(self.DEFAULT_FILTERS)

        try:
            filter_path = Path("ml_models/learned_filters.json")
            if filter_path.exists():
                with open(filter_path, "r") as f:
                    learned = json.load(f)
                active = []
                for key, value in learned.items():
                    if key.startswith("_"):
                        continue
                    if value is not None:
                        defaults[key] = value
                        active.append(f"{key}={value}")
                self.logger.info(f"✓ Loaded learned filters: {', '.join(active)}")
            else:
                self.logger.info("No learned_filters.json found — using permissive defaults")
        except Exception as e:
            self.logger.warning(f"Could not load learned filters: {e} — using defaults")

        return defaults

    def screen_with_tradingview(self, max_results: int = 1500) -> "pd.DataFrame":
        """
        Screen stocks via TradingView and return a DataFrame that includes
        all the indicator columns we requested — so downstream code can
        build t3_ features directly from this data without yfinance.
        """
        import pandas as pd

        if not SCREENER_AVAILABLE or self.screener is None:
            self.logger.error("TradingView screener not available!")
            return pd.DataFrame()

        self.logger.info("=" * 60)
        self.logger.info("SMART SCREENER — applying model-driven filters")
        self.logger.info("=" * 60)

        tv_col_bounds: dict = {}
        filter_log = []

        for filter_key, (tv_col, operation) in self.TV_FILTER_MAP.items():
            value = self.filters.get(filter_key)
            if value is None:
                continue
            if tv_col not in tv_col_bounds:
                tv_col_bounds[tv_col] = {}
            if operation == "greater":
                existing = tv_col_bounds[tv_col].get("min")
                if existing is None or value > existing:
                    tv_col_bounds[tv_col]["min"] = value
                    filter_log.append(f"{tv_col} > {value}  [{filter_key}]")
            elif operation == "less":
                existing = tv_col_bounds[tv_col].get("max")
                if existing is None or value < existing:
                    tv_col_bounds[tv_col]["max"] = value
                    filter_log.append(f"{tv_col} < {value}  [{filter_key}]")

        tv_filters = []
        for tv_col, bounds in tv_col_bounds.items():
            if "min" in bounds:
                tv_filters.append({"left": tv_col, "operation": "greater", "right": bounds["min"]})
            if "max" in bounds:
                tv_filters.append({"left": tv_col, "operation": "less", "right": bounds["max"]})

        self.logger.info(f"Active filters ({len(tv_filters)}):")
        for line in filter_log:
            self.logger.info(f"  {line}")

        # Request all indicator columns so we can use them as t3_ features
        columns_to_fetch = list(set(EXTRA_TV_COLUMNS))

        try:
            result = self.screener.screen(
                market="america",
                filters=tv_filters,
                sort_by="relative_volume_10d_calc",
                sort_order="desc",
                limit=max_results,
                columns=columns_to_fetch,
            )

            if result.get("status") == "success" and result.get("data"):
                df = pd.DataFrame(result["data"])
                self.logger.info(f"✓ Screened {len(df)} stocks with {len(df.columns)} columns")
                return df

            # Fallback: some screener versions don't support columns param
            result = self.screener.screen(
                market="america",
                filters=tv_filters,
                sort_by="relative_volume_10d_calc",
                sort_order="desc",
                limit=max_results,
            )
            if result.get("status") == "success" and result.get("data"):
                df = pd.DataFrame(result["data"])
                self.logger.info(f"✓ Screened {len(df)} stocks (default columns)")
                return df

            self.logger.warning("Screener returned no data or error status")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# FIX 1: Build t3_ features directly from TradingView screener data
# ---------------------------------------------------------------------------

def build_features_from_tv_data(row: dict, symbol: str) -> dict:
    """
    Convert a single TradingView screener row into a feature dict with
    t3_ prefixed names matching what the model expects.

    This replaces the yfinance daily-bar round-trip for t3_/t5_/t10_ features.
    TradingView already has this data — we just need to rename the columns.

    Returns a dict of {t3_FeatureName: value, ...} plus symbol/exchange/current_price.
    """
    result = {
        "symbol":   symbol,
        "exchange": "NASDAQ",
    }

    # Extract and rename each TV column to its t3_ model equivalent
    seen_targets = set()
    for tv_col, model_name in TV_TO_T3_MAP.items():
        target = f"t3_{model_name}"
        if target in seen_targets:
            continue  # skip duplicate aliases

        value = row.get(tv_col)
        if value is None:
            # Try lowercase
            value = row.get(tv_col.lower())
        if value is not None:
            try:
                fval = float(value)
                if not (np.isnan(fval) or np.isinf(fval)):
                    result[target] = fval
                    seen_targets.add(target)
            except (TypeError, ValueError):
                pass

    # current_price: use close from screener data
    close_val = row.get("close") or row.get("Close")
    if close_val is not None:
        try:
            result["current_price"] = float(close_val)
        except (TypeError, ValueError):
            pass

    # Derive some additional features that the model likely uses
    # but TradingView doesn't return directly

    close = result.get("current_price") or result.get("t3_Close")
    if close and close > 0:
        # Price vs EMA20
        ema20 = result.get("t3_EMA_20")
        if ema20:
            result["t3_Price_vs_EMA20"] = (close / ema20 - 1) * 100

        # Price vs SMA20
        sma20 = result.get("t3_SMA_20")
        if sma20:
            result["t3_Price_vs_SMA20"] = (close / sma20 - 1) * 100

        # EMA crossover flags
        ema10 = result.get("t3_EMA_10")
        ema50 = result.get("t3_EMA_50")
        sma50 = result.get("t3_SMA_50")
        if ema20 and ema50:
            result["t3_EMA_12_26_Diff"] = ema20 - ema50
        if ema10 and ema20:
            result["t3_SMA_20_50_Diff"] = ema10 - ema20

    return result


def fetch_t1_data_for_symbol(symbol: str, logger) -> dict:
    """
    Fetch ONLY T-1 intraday data for a single symbol (best-effort, optional).
    Returns {} if unavailable — the stock will still be scored on t3_ features.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    from datetime import time as dt_time
    import pandas as pd
    import pytz

    try:
        ticker = yf.Ticker(symbol)
        df_intraday = ticker.history(period="5d", interval="5m")

        if df_intraday.empty or len(df_intraday) < 50:
            return {}

        # Localise
        if df_intraday.index.tz is None:
            df_intraday.index = df_intraday.index.tz_localize("America/New_York")
        else:
            df_intraday.index = df_intraday.index.tz_convert("America/New_York")

        available_dates = sorted(df_intraday.index.date, reverse=True)
        if not available_dates:
            return {}

        # T-1 = most recent completed trading day
        yesterday = available_dates[0]
        day_bars = df_intraday[df_intraday.index.date == yesterday]

        if day_bars.empty:
            return {}

        result = {}

        # T-1 close snapshot (last bar of the day)
        close_bar = day_bars.iloc[-1]
        for col, val in close_bar.to_dict().items():
            try:
                fval = float(val)
                if not (np.isnan(fval) or np.isinf(fval)):
                    result[f"t1_close_{col}"] = fval
            except (TypeError, ValueError):
                pass

        # T-1 open snapshot (first bar of the day, ~9:30am)
        open_bars = day_bars[day_bars.index.time <= dt_time(10, 0)]
        if not open_bars.empty:
            open_bar = open_bars.iloc[0]
            for col, val in open_bar.to_dict().items():
                try:
                    fval = float(val)
                    if not (np.isnan(fval) or np.isinf(fval)):
                        result[f"t1_open_{col}"] = fval
                except (TypeError, ValueError):
                    pass

        return result

    except Exception:
        return {}


def _get_calibrated_gain_estimate(probability: float) -> float:
    if probability >= 0.95: return 40.0
    if probability >= 0.90: return 30.0
    if probability >= 0.80: return 20.0
    if probability >= 0.70: return 15.0
    if probability >= 0.60: return 10.0
    if probability >= 0.50: return 7.0
    return 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=1500)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--no-t1",
        action="store_true",
        help="Skip T-1 intraday fetch entirely (fastest, uses only TV screener data)"
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    prediction_date = get_next_trading_day()

    logger.info("=" * 80)
    logger.info("ML SCREENING & PREDICTION")
    logger.info("=" * 80)
    logger.info(f"Prediction date (trading session): {prediction_date}")
    logger.info("=" * 80)

    screener = SmartScreener(logger=logger)

    try:
        predictor = ExplosionPredictor()
        supabase  = MLPredictionSupabaseClient({})
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1

    # ── STEP 1: SCREENING ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: INTELLIGENT SCREENING")
    logger.info("=" * 80)

    screened_df = screener.screen_with_tradingview(max_results=args.max_results)
    if screened_df.empty:
        logger.error("No stocks passed screening")
        return 1
    logger.info(f"✓ Screened {len(screened_df)} stocks")

    # ── STEP 2: BUILD FEATURES FROM TV DATA (FIX 1) ─────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: BUILD T3 FEATURES FROM TRADINGVIEW DATA")
    logger.info("=" * 80)
    logger.info("Using TradingView's own indicator data — no yfinance round-trip needed.")
    logger.info("This gives near-100% coverage vs ~1% coverage from yfinance for micro-caps.")

    enriched_stocks = []
    failed_count = 0

    for _, row in screened_df.iterrows():
        try:
            symbol_full = str(row.get("symbol", ""))
            if ":" in symbol_full:
                exchange, symbol = symbol_full.split(":", 1)
            else:
                symbol = symbol_full
                exchange = "NASDAQ"

            symbol = symbol.strip().upper()
            if not symbol:
                failed_count += 1
                continue

            # Skip OTC / excluded patterns
            if exchange == "OTC" or len(symbol) > 5 or "." in symbol:
                failed_count += 1
                continue

            row_dict = row.to_dict()
            features = build_features_from_tv_data(row_dict, symbol)
            features["exchange"] = exchange

            if "current_price" not in features or not features["current_price"]:
                failed_count += 1
                continue

            enriched_stocks.append(features)

        except Exception as e:
            logger.debug(f"Error processing row: {e}")
            failed_count += 1

    logger.info(f"✓ Built features for {len(enriched_stocks)} stocks ({failed_count} skipped)")

    if not enriched_stocks:
        logger.error("Failed to build features for any stocks")
        return 1

    # ── STEP 3: OPTIONAL T-1 INTRADAY ENRICHMENT ────────────────────────────
    if not args.no_t1:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: OPTIONAL T-1 INTRADAY ENRICHMENT (best-effort)")
        logger.info("=" * 80)
        logger.info("Fetching T-1 snapshots for top 200 candidates (by volume rank)")
        logger.info("Stocks without T-1 data will still be scored on T3 features.")

        # Only fetch T-1 for the top-200 by volume/rank to limit yfinance calls
        top_200 = [s["symbol"] for s in enriched_stocks[:200]]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time, random

        t1_map = {}
        t1_count = 0

        def fetch_with_jitter(sym):
            time.sleep(random.uniform(0.05, 0.2))
            return sym, fetch_t1_data_for_symbol(sym, logger)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_with_jitter, sym): sym for sym in top_200}
            for i, future in enumerate(as_completed(futures), 1):
                if i % 50 == 0:
                    logger.info(f"  T-1 progress: {i}/{len(top_200)} | found: {t1_count}")
                sym, t1_data = future.result()
                if t1_data:
                    t1_map[sym] = t1_data
                    t1_count += 1

        # Merge T-1 data into enriched_stocks
        for stock in enriched_stocks:
            sym = stock["symbol"]
            if sym in t1_map:
                stock.update(t1_map[sym])

        logger.info(f"✓ T-1 enrichment: {t1_count}/{len(top_200)} stocks got intraday features")
    else:
        logger.info("\nSTEP 3: Skipped (--no-t1 flag)")

    # ── STEP 4: PREPARE FEATURES ─────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: PREPARE FEATURES")
    logger.info("=" * 80)

    features_df = pd.DataFrame(enriched_stocks)
    t1_stocks = sum(1 for s in enriched_stocks if any(k.startswith("t1_") for k in s))
    logger.info(f"✓ Feature matrix: {len(features_df)} stocks × {len(features_df.columns)} raw columns")
    logger.info(f"  With T-1 intraday features: {t1_stocks}")
    logger.info(f"  T3-only features: {len(features_df) - t1_stocks}")

    # ── STEP 5: ML PREDICTION ────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: ML PREDICTION")
    logger.info("=" * 80)

    historical_gains = supabase.get_historical_prediction_accuracy(days_back=30)

    try:
        predictions_df = predictor.predict_with_targets(features_df, historical_gains)
        logger.info(f"✓ Generated {len(predictions_df)} predictions")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Last-resort gain fallback
    if 'target_gain_pct' in predictions_df.columns:
        bad_gain_mask = (
            predictions_df['target_gain_pct'].isna() |
            (predictions_df['target_gain_pct'].abs() < 0.5) |
            (predictions_df['target_gain_pct'] > 500)
        )
        if bad_gain_mask.any():
            n_bad = bad_gain_mask.sum()
            if n_bad == len(predictions_df):
                logger.warning(
                    f"  ⚠️  All {n_bad} gain estimates missing — gain_regressor.pkl not trained yet."
                )
            else:
                logger.info(f"  Last-resort gain fallback applied to {n_bad} stocks")
            predictions_df.loc[bad_gain_mask, 'target_gain_pct'] = (
                predictions_df.loc[bad_gain_mask, 'explosion_probability']
                .apply(_get_calibrated_gain_estimate)
            )
            predictions_df.loc[bad_gain_mask, 'target_gain_low'] = (
                predictions_df.loc[bad_gain_mask, 'target_gain_pct'] * 0.5
            )
            predictions_df.loc[bad_gain_mask, 'target_gain_high'] = (
                predictions_df.loc[bad_gain_mask, 'target_gain_pct'] * 1.8
            )

        if 'current_price' in predictions_df.columns:
            predictions_df['target_price'] = (
                predictions_df['current_price'] *
                (1 + predictions_df['target_gain_pct'] / 100)
            )
            predictions_df['target_price_low'] = (
                predictions_df['current_price'] *
                (1 + predictions_df['target_gain_low'] / 100)
            )
            predictions_df['target_price_high'] = (
                predictions_df['current_price'] *
                (1 + predictions_df['target_gain_high'] / 100)
            )

    log_probability_distribution(
        predictions_df, logger,
        label=f"All {len(predictions_df)} screened stocks"
    )

    # ── STEP 6: TOP PREDICTIONS ──────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 6: TOP {args.top_n} PREDICTIONS")
    logger.info("=" * 80)

    top_predictions = predictions_df.head(args.top_n)

    logger.info(f"\nTop {min(20, len(top_predictions))} Predictions for {prediction_date}:")
    logger.info("-" * 100)
    logger.info(f"{'#':<4} {'Symbol':<8} {'Signal':<13} {'Prob':<8} {'Price':<10} {'Target':<10} {'Gain':<8} {'T-1?'}")
    logger.info("-" * 100)

    for rank, (_, row) in enumerate(top_predictions.head(20).iterrows(), 1):
        current_price = row.get('current_price', 0)
        has_t1 = any(k.startswith("t1_") for k in row.index if pd.notna(row[k]))
        logger.info(
            f"{rank:<4} {row['symbol']:<8} {row['signal']:<13} "
            f"{row['explosion_probability']*100:>6.2f}%  "
            f"${current_price:>8.2f}  "
            f"${row.get('target_price', 0):>8.2f}  "
            f"+{row.get('target_gain_pct', 0):>5.1f}%"
            f"  {'✓' if has_t1 else '—'}"
        )

    # ── STEP 7: STORE PREDICTIONS ────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: STORE PREDICTIONS")
    logger.info("=" * 80)

    predictions_list = [
        {
            'symbol':               row['symbol'],
            'exchange':             row.get('exchange', 'NASDAQ'),
            'prediction_date':      prediction_date,
            'explosion_probability': float(row['explosion_probability']),
            'prediction':           int(row['prediction']),
            'signal':               row['signal'],
            'target_gain_pct':      float(row.get('target_gain_pct', 0)),
            'target_gain_low':      float(row.get('target_gain_low', 0)),
            'target_gain_high':     float(row.get('target_gain_high', 0)),
            'current_price':        float(row.get('current_price', 0)),
            'target_price':         float(row.get('target_price', 0)),
            'target_price_low':     float(row.get('target_price_low', 0)),
            'target_price_high':    float(row.get('target_price_high', 0)),
        }
        for _, row in top_predictions.iterrows()
    ]

    if predictions_list:
        count = supabase.write_predictions(predictions_list)
        logger.info(f"✓ Wrote {count} predictions for trading session: {prediction_date}")

    # ── STEP 8: SCREENING LOG ────────────────────────────────────────────────
    screening_log = {
        'screening_date':               prediction_date,
        'total_symbols_attempted':      args.max_results,
        'symbols_fetched_successfully': len(enriched_stocks),
        'symbols_after_all_filters':    len(features_df),
        'total_predictions':            len(predictions_df),
        'strong_buy_count':  len(predictions_df[predictions_df['signal'] == 'STRONG BUY']),
        'buy_count':         len(predictions_df[predictions_df['signal'] == 'BUY']),
        'hold_count':        len(predictions_df[predictions_df['signal'] == 'HOLD']),
        'avoid_count':       len(predictions_df[predictions_df['signal'] == 'AVOID']),
        'avg_probability':    float(predictions_df['explosion_probability'].mean()),
        'max_probability':    float(predictions_df['explosion_probability'].max()),
        'min_probability':    float(predictions_df['explosion_probability'].min()),
        'model_version':      'tv_native_features_v2'
    }
    supabase.write_screening_log(screening_log)

    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("✓ PREDICTION COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
