#!/usr/bin/env python3
"""
ML Stock Screener & Predictor

FIXES IN THIS VERSION (2026-03-03 v6):

FIX 1 — fetch_t1_data_for_symbol now computes real indicators from 5-min bars.

FIX 2 — build_features_from_tv_data lowercases keys for t3/t5/t10 prefix models.

FIX 3 — write_predictions_upsert instead of insert-skip-on-duplicate.

FIX 4 — SmartScreener._load_learned_filters now clamps aggressive HV minimums
  (min_hv10 / min_hv20) and volume-ratio minimums before applying them as TV
  screener filters.  Previously, winner-derived p10 HV values of 56%+ were
  passed straight to TradingView, excluding ~98% of the market and leaving
  only 8-15 stocks — too few for the model to produce varied probabilities.

FIX 5 — Pre-flight variance check in STEP 4 diagnoses zero-variance indicator
  columns before the model runs, giving an actionable error message instead of
  silently producing identical probabilities.
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
# TV screener column → base indicator name mappings
# ---------------------------------------------------------------------------

TV_TO_MODEL_BASE = {
    "close":                        "Close",
    "open":                         "Open",
    "high":                         "High",
    "low":                          "Low",
    "volume":                       "Volume",
    "change":                       "price_change_1d",
    "change_abs":                   "MOM_10",
    "RSI":                          "RSI_14",
    "RSI[1]":                       "RSI_14",
    "Stoch.K":                      "STOCHk_14_3_3",
    "Stoch.D":                      "STOCHd_14_3_3",
    "Stoch.K[1]":                   "STOCHk_14_3_3",
    "Stoch.D[1]":                   "STOCHd_14_3_3",
    "W.R":                          "WILLR_14",
    "W.R[1]":                       "WILLR_14",
    "MACD.macd":                    "MACD_12_26_9",
    "MACD.signal":                  "MACDs_12_26_9",
    "BB.upper":                     "BBU_20_2.0_2.0",
    "BB.lower":                     "BBL_20_2.0_2.0",
    "BB.basis":                     "BBM_20_2.0_2.0",
    "BBPower":                      "BBP_20_2.0_2.0",
    "EMA5":                         "EMA_5",
    "EMA10":                        "EMA_10",
    "EMA20":                        "EMA_20",
    "EMA30":                        "EMA_26",
    "EMA50":                        "EMA_50",
    "SMA5":                         "SMA_5",
    "SMA10":                        "SMA_10",
    "SMA20":                        "SMA_20",
    "SMA30":                        "SMA_50",
    "SMA50":                        "SMA_50",
    "Mom":                          "MOM_10",
    "AO":                           "AO",
    "CCI20":                        "CCI_20",
    "UO":                           "UO",
    "ROC":                          "ROC_10",
    "relative_volume_10d_calc":     "Volume_Ratio",
    "VWMA":                         "VWMA_20",
    "ATR":                          "ATR_14",
    "ADX":                          "ADX_14",
    "ADX+DI":                       "DMP_14",
    "ADX-DI":                       "DMN_14",
    "Volatility.D":                 "HV_20",
    "gap":                          "Gap_Pct",
    "premarket_gap":                "Gap_Pct",
    "High.52W":                     "high_52w",
    "Low.52W":                      "low_52w",
    "Aroon.Up":                     "AROONU_25",
    "Aroon.Down":                   "AROOND_25",
}

TV_TO_MODEL_BASE_T3_OVERRIDES = {
    "close":        "Close",
    "open":         "Open",
    "high":         "High",
    "low":          "Low",
    "volume":       "Volume",
    "RSI":          "RSI_14",
    "ATR":          "ATR_14",
    "ADX":          "ADX_14",
    "MACD.macd":    "MACD_12_26_9",
    "MACD.signal":  "MACDs_12_26_9",
}

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

GAIN_FLOOR   = 5.0
GAIN_CEILING = 55.0

_LOWERCASE_PREFIXES = ("t3", "t5", "t10")

# FIX 4: Screener-level caps — keep the funnel wide so ML has a real distribution
SCREENER_HV_MIN_CAP    = 30.0   # never require HV > 30% at screen time
SCREENER_VOL_RATIO_CAP = 2.5   # never require vol_ratio > 2.5 at screen time


def _uses_lowercase(prefix: str) -> bool:
    return any(prefix == p or prefix.startswith(p + "_") for p in _LOWERCASE_PREFIXES)


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

    total     = len(probs)
    bar_width = 30

    for bucket_label, mask in buckets:
        count = mask.sum()
        pct   = (count / total * 100) if total > 0 else 0
        bar   = "█" * int(pct / 100 * bar_width)
        logger.info(f"  {bucket_label:<22} {count:>4}  ({pct:>5.1f}%)  {bar}")

    logger.info("-" * 70)
    logger.info(f"  Total stocks evaluated:  {total}")
    logger.info(f"  Min probability:         {probs.min():.4f} ({probs.min()*100:.2f}%)")
    logger.info(f"  Max probability:         {probs.max():.4f} ({probs.max()*100:.2f}%)")
    logger.info(f"  Mean probability:        {probs.mean():.4f} ({probs.mean()*100:.2f}%)")
    logger.info(f"  Median probability:      {probs.median():.4f} ({probs.median()*100:.2f}%)")
    logger.info(f"  Std deviation:           {probs.std():.6f}")

    if probs.std() < 0.02:
        logger.warning(
            f"  ⚠️  VERY LOW PROB STD ({probs.std():.6f}) — likely feature prefix mismatch."
            f"\n      Check that build_features_from_tv_data is using the same prefix"
            f"\n      as the model (see predictor.model_feature_prefix in logs)."
        )
    elif probs.std() < 0.05:
        logger.warning(
            f"  ⚠️  LOW PROB STD ({probs.std():.4f}) — limited feature discrimination."
        )
    else:
        logger.info(f"  ✅ Probability std {probs.std():.4f} — distribution looks healthy.")

    logger.info("")
    logger.info("  Signal breakdown:")
    if 'signal' in predictions_df.columns:
        signal_counts = predictions_df['signal'].value_counts()
        for signal in ['STRONG BUY', 'BUY', 'HOLD', 'AVOID']:
            count = signal_counts.get(signal, 0)
            pct   = (count / total * 100) if total > 0 else 0
            logger.info(f"    {signal:<12} {count:>4}  ({pct:>5.1f}%)")

    if 'target_gain_pct' in predictions_df.columns:
        gains     = predictions_df['target_gain_pct']
        gain_std  = gains.std()
        gain_mean = gains.mean()
        logger.info("")
        logger.info(f"  Gain estimates:  mean={gain_mean:.1f}%  std={gain_std:.2f}%"
                    + ("  ⚠️  FLAT — rank correction applied" if gain_std < 1.0 else "  ✅"))

    logger.info("")
    logger.info("  Top 10 stocks by probability:")
    logger.info(f"  {'#':<4} {'Symbol':<8} {'Prob':>8}  {'Signal':<13} {'Gain'}")
    logger.info("  " + "-" * 50)
    for rank, (_, row) in enumerate(predictions_df.head(10).iterrows(), 1):
        prob_pct = row['explosion_probability'] * 100
        signal   = row.get('signal', 'N/A')
        symbol   = row.get('symbol', 'N/A')
        gain     = row.get('target_gain_pct', 0)
        logger.info(f"  {rank:<4} {symbol:<8} {prob_pct:>7.2f}%  {signal:<13} +{gain:.1f}%")

    logger.info("")


def get_next_trading_day() -> str:
    est = pytz.timezone('America/New_York')
    now_est = datetime.now(est)
    prediction_day = now_est + timedelta(days=1)
    while prediction_day.weekday() >= 5:
        prediction_day += timedelta(days=1)
    return prediction_day.date().isoformat()


# ---------------------------------------------------------------------------
# SmartScreener
# ---------------------------------------------------------------------------

class SmartScreener:
    """Intelligent screener using model-derived filters."""

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
                    if value is None:
                        continue

                    # FIX 4: Clamp aggressive HV minimums.
                    # Winner-derived p10 HV can be 56%+ — passing that to
                    # TradingView leaves only 8-15 stocks, too few for the
                    # model to produce a meaningful probability distribution.
                    if key in ("min_hv10", "min_hv20") and float(value) > SCREENER_HV_MIN_CAP:
                        self.logger.info(
                            f"  Clamping {key}={value:.2f} → {SCREENER_HV_MIN_CAP} "
                            f"(raw winner p10 too aggressive for broad screening)"
                        )
                        value = SCREENER_HV_MIN_CAP

                    # FIX 4: Clamp aggressive volume-ratio minimums.
                    if key in ("min_volume_ratio", "min_relative_volume") and float(value) > SCREENER_VOL_RATIO_CAP:
                        self.logger.info(
                            f"  Clamping {key}={value:.2f} → {SCREENER_VOL_RATIO_CAP}"
                        )
                        value = SCREENER_VOL_RATIO_CAP

                    defaults[key] = value
                    active.append(f"{key}={value}")

                self.logger.info(f"✓ Loaded learned filters: {', '.join(active)}")
            else:
                self.logger.info("No learned_filters.json found — using permissive defaults")
        except Exception as e:
            self.logger.warning(f"Could not load learned filters: {e} — using defaults")
        return defaults

    def screen_with_tradingview(self, max_results: int = 1500) -> "pd.DataFrame":
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
                n_indicator_cols = sum(1 for c in df.columns if c in EXTRA_TV_COLUMNS)
                self.logger.info(
                    f"✓ Screened {len(df)} stocks with {len(df.columns)} columns "
                    f"({n_indicator_cols} indicator columns)"
                )
                if n_indicator_cols < 5:
                    self.logger.warning(
                        "  ⚠️  Very few indicator columns returned. "
                        "tradingview-scraper may not support columns= for this version."
                    )
                return df

            self.logger.warning("Retrying without columns= parameter (version fallback)...")
            result = self.screener.screen(
                market="america",
                filters=tv_filters,
                sort_by="relative_volume_10d_calc",
                sort_order="desc",
                limit=max_results,
            )
            if result.get("status") == "success" and result.get("data"):
                df = pd.DataFrame(result["data"])
                self.logger.info(
                    f"✓ Screened {len(df)} stocks (default columns only: {list(df.columns)})"
                )
                self.logger.warning(
                    "  ⚠️  Using default TV columns only (no RSI/ATR/etc)."
                    "\n      All stocks will get default feature values → near-identical probabilities."
                    "\n      Upgrade tradingview-scraper to >=0.4.19 for indicator columns."
                )
                return df

            self.logger.warning("Screener returned no data or error status")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features_from_tv_data(row: dict, symbol: str, feature_prefix: str = "t1_close") -> dict:
    """
    Convert a single TradingView screener row into a feature dict.
    FIX 2: Lowercases the full key for t3/t5/t10 prefix models.
    """
    result = {
        "symbol":   symbol,
        "exchange": "NASDAQ",
    }

    seen_targets = set()
    for tv_col, model_name in TV_TO_MODEL_BASE.items():
        target = f"{feature_prefix}_{model_name}"

        if _uses_lowercase(feature_prefix):
            target = target.lower()

        if target in seen_targets:
            continue

        value = row.get(tv_col)
        if value is None:
            value = row.get(tv_col.lower())
        if value is None:
            for k in row:
                if k.lower() == tv_col.lower():
                    value = row[k]
                    break

        if value is not None:
            try:
                fval = float(value)
                if not (np.isnan(fval) or np.isinf(fval)):
                    result[target] = fval
                    seen_targets.add(target)
            except (TypeError, ValueError):
                pass

    close_val = row.get("close") or row.get("Close")
    if close_val is not None:
        try:
            result["current_price"] = float(close_val)
        except (TypeError, ValueError):
            pass

    close = result.get("current_price") or result.get(f"{feature_prefix}_Close")
    if close and close > 0:
        if _uses_lowercase(feature_prefix):
            ema20 = result.get(f"{feature_prefix}_ema_20")
            ema50 = result.get(f"{feature_prefix}_ema_50")
            ema10 = result.get(f"{feature_prefix}_ema_10")
            sma20 = result.get(f"{feature_prefix}_sma_20")
            if ema20:
                result[f"{feature_prefix}_price_vs_ema20"] = (close / ema20 - 1) * 100
            if sma20:
                result[f"{feature_prefix}_price_vs_sma20"] = (close / sma20 - 1) * 100
            if ema20 and ema50:
                result[f"{feature_prefix}_ema_12_26_diff"] = ema20 - ema50
            if ema10 and ema20:
                result[f"{feature_prefix}_sma_20_50_diff"] = ema10 - ema20
        else:
            ema20 = result.get(f"{feature_prefix}_EMA_20")
            ema50 = result.get(f"{feature_prefix}_EMA_50")
            ema10 = result.get(f"{feature_prefix}_EMA_10")
            sma20 = result.get(f"{feature_prefix}_SMA_20")
            if ema20:
                result[f"{feature_prefix}_Price_vs_EMA20"] = (close / ema20 - 1) * 100
            if sma20:
                result[f"{feature_prefix}_Price_vs_SMA20"] = (close / sma20 - 1) * 100
            if ema20 and ema50:
                result[f"{feature_prefix}_EMA_12_26_Diff"] = ema20 - ema50
            if ema10 and ema20:
                result[f"{feature_prefix}_SMA_20_50_Diff"] = ema10 - ema20

    return result


# ---------------------------------------------------------------------------
# T-1 intraday indicator fetch
# ---------------------------------------------------------------------------

def fetch_t1_data_for_symbol(symbol: str, logger) -> dict:
    """
    Fetch T-1 intraday 5-min data and compute technical indicators.
    FIX 1: Computes the full indicator suite and renames via t1_column_map.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    try:
        from t1_column_map import rename_t1_columns as _rename
        _t1_map_available = True
    except ImportError:
        _rename = None
        _t1_map_available = False

    try:
        ticker      = yf.Ticker(symbol)
        df_intraday = ticker.history(period="5d", interval="5m")

        if df_intraday.empty or len(df_intraday) < 50:
            return {}

        if df_intraday.index.tz is None:
            df_intraday.index = df_intraday.index.tz_localize("America/New_York")
        else:
            df_intraday.index = df_intraday.index.tz_convert("America/New_York")

        available_dates = sorted(df_intraday.index.date, reverse=True)
        if not available_dates:
            return {}

        yesterday = available_dates[0]
        day_bars  = df_intraday[df_intraday.index.date == yesterday].copy()

        if len(day_bars) < 20:
            return {}

        day_bars.columns = [c.lower() for c in day_bars.columns]

        def _compute_indicators(c, h, l, v, o) -> dict:
            ind = {}

            def safe(series, default=0.0):
                try:
                    val = float(series.iloc[-1])
                    return default if (np.isnan(val) or np.isinf(val)) else val
                except Exception:
                    return default

            ind["close"]  = float(c.iloc[-1])
            ind["open"]   = float(o.iloc[0])
            ind["high"]   = float(h.max())
            ind["low"]    = float(l.min())
            ind["volume"] = float(v.sum())
            close_v = ind["close"]

            delta = c.diff()
            gain  = delta.clip(lower=0)
            loss  = (-delta.clip(upper=0))
            for period, col_name in [(7, "rsi7"), (14, "rsi"), (21, "rsi21"), (28, "rsi28")]:
                ag = gain.ewm(com=period - 1, min_periods=period).mean()
                al = loss.ewm(com=period - 1, min_periods=period).mean()
                rs = ag / al.replace(0, np.nan)
                ind[col_name] = safe(100 - (100 / (1 + rs)), 50.0)
            ind["rsi[1]"] = ind["rsi"]

            ema12     = c.ewm(span=12, adjust=False).mean()
            ema26     = c.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
            ind["macd.macd"]   = safe(macd_line)
            ind["macd.signal"] = safe(macd_sig)
            ind["macd_diff"]   = safe(macd_line - macd_sig)

            for n in [5, 10, 12, 20, 26, 50]:
                ind[f"ema{n}"] = safe(c.ewm(span=n, adjust=False).mean(), float(c.mean()))
            for n in [5, 10, 20, 50]:
                ind[f"sma{n}"] = safe(c.rolling(n).mean(), float(c.mean()))

            sma20_v = ind.get("sma20") or float(c.mean())
            ema20_v = ind.get("ema20") or float(c.mean())
            ema10_v = ind.get("ema10") or float(c.mean())
            ema12_v = ind.get("ema12") or float(c.mean())
            ema26_v = ind.get("ema26") or float(c.mean())

            if sma20_v:
                ind["price_vs_sma20"] = (close_v / sma20_v - 1) * 100
            if ema20_v:
                ind["price_vs_ema20"] = (close_v / ema20_v - 1) * 100
            ind["ema_12_26_diff"] = ema12_v - ema26_v
            ind["sma_20_50_diff"] = ind.get("sma20", 0) - ind.get("sma50", 0)

            lo14  = l.rolling(14).min()
            hi14  = h.rolling(14).max()
            rng14 = (hi14 - lo14).replace(0, np.nan)
            stk   = (100 * (c - lo14) / rng14).rolling(3).mean()
            std   = stk.rolling(3).mean()
            ind["stoch.k"]    = safe(stk, 50.0)
            ind["stoch.d"]    = safe(std, 50.0)
            ind["stoch.k[1]"] = ind["stoch.k"]
            ind["stoch.d[1]"] = ind["stoch.d"]

            ind["w.r"] = safe(-100 * (hi14 - c) / rng14, -50.0)

            tr = pd.concat([
                h - l,
                (h - c.shift()).abs(),
                (l - c.shift()).abs(),
            ], axis=1).max(axis=1)
            for period, col_name in [(7, "atr7"), (14, "atr"), (20, "atr20")]:
                ind[col_name] = safe(tr.rolling(period).mean(), 0.5)

            up_move = h.diff()
            dn_move = -l.diff()
            pdm = pd.Series(
                np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0),
                index=c.index
            )
            ndm = pd.Series(
                np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0),
                index=c.index
            )
            atr14 = tr.rolling(14).mean().replace(0, np.nan)
            pdi   = 100 * pdm.rolling(14).mean() / atr14
            ndi   = 100 * ndm.rolling(14).mean() / atr14
            dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
            ind["adx"]    = safe(dx.rolling(14).mean(), 20.0)
            ind["adx+di"] = safe(pdi, 20.0)
            ind["adx-di"] = safe(ndi, 20.0)

            bb_mid = c.rolling(20).mean()
            bb_std = c.rolling(20).std()
            bb_up  = bb_mid + 2 * bb_std
            bb_lo  = bb_mid - 2 * bb_std
            ind["bb.upper"]  = safe(bb_up,  close_v)
            ind["bb.lower"]  = safe(bb_lo,  close_v)
            ind["bb.middle"] = safe(bb_mid, close_v)
            ind["bb_width"]  = safe(
                (bb_up - bb_lo) / bb_mid.replace(0, np.nan) * 100, 0.0
            )
            ind["bbpower"] = safe(
                (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan), 0.5
            )

            vm5  = v.rolling(5).mean()
            vm10 = v.rolling(10).mean()
            vm20 = v.rolling(20).mean()
            ind["volume_sma5"]  = safe(vm5,  float(v.mean()))
            ind["volume_sma10"] = safe(vm10, float(v.mean()))
            ind["volume_sma20"] = safe(vm20, float(v.mean()))
            ind["volume_ratio"] = safe(v / vm20.replace(0, np.nan), 1.0)

            obv_vals = [0.0]
            c_arr, v_arr = c.values, v.values
            for i in range(1, len(c_arr)):
                if   c_arr[i] > c_arr[i - 1]: obv_vals.append(obv_vals[-1] + v_arr[i])
                elif c_arr[i] < c_arr[i - 1]: obv_vals.append(obv_vals[-1] - v_arr[i])
                else:                          obv_vals.append(obv_vals[-1])
            ind["obv"] = float(obv_vals[-1])

            mf_mult   = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
            mf_volume = mf_mult * v
            ind["cmf"] = safe(
                mf_volume.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan),
                0.0
            )

            tp    = (h + l + c) / 3
            tp_ma = tp.rolling(20).mean()
            tp_md = tp.rolling(20).apply(
                lambda x: np.abs(x - x.mean()).mean(), raw=True
            )
            ind["cci20"] = safe(
                (tp - tp_ma) / (0.015 * tp_md.replace(0, np.nan)), 0.0
            )

            ind["ao"]  = safe(
                (h + l).rolling(5).mean() / 2 - (h + l).rolling(34).mean() / 2,
                0.0
            )
            ind["mom"] = safe(c.diff(10), 0.0)
            ind["roc"] = safe(c.pct_change(10) * 100, 0.0)

            log_ret = np.log(c / c.shift(1))
            for hv_w, col_name in [
                (10, "volatility_10d"), (20, "volatility_20d"), (30, "volatility_30d")
            ]:
                ind[col_name] = safe(
                    log_ret.rolling(hv_w).std() * np.sqrt(252 * 78) * 100, 0.0
                )

            for n, col_name in [
                (1, "price_change_1d"), (2, "price_change_2d"),
                (3, "price_change_3d"), (5, "price_change_5d"),
            ]:
                if len(c) > n:
                    prev = float(c.iloc[-(n + 1)])
                    ind[col_name] = ((close_v / prev) - 1) * 100 if prev else 0.0

            if len(c) > 1:
                prev_bar = float(c.iloc[-2])
                if prev_bar:
                    ind["gap_%"] = (float(o.iloc[0]) / prev_bar - 1) * 100

            if len(h) >= 26:
                hi_idx = h.rolling(26).apply(
                    lambda x: float(np.argmax(x)), raw=True
                )
                lo_idx = l.rolling(26).apply(
                    lambda x: float(np.argmin(x)), raw=True
                )
                ind["aroon_up"]        = safe(hi_idx / 25 * 100, 50.0)
                ind["aroon_down"]      = safe(lo_idx / 25 * 100, 50.0)
                ind["aroon_indicator"] = ind["aroon_up"] - ind["aroon_down"]

            bp   = c - pd.concat([l, c.shift()], axis=1).min(axis=1)
            tr_u = (
                pd.concat([h, c.shift()], axis=1).max(axis=1)
                - pd.concat([l, c.shift()], axis=1).min(axis=1)
            )
            a7  = bp.rolling(7).sum()  / tr_u.rolling(7).sum().replace(0, np.nan)
            a14 = bp.rolling(14).sum() / tr_u.rolling(14).sum().replace(0, np.nan)
            a28 = bp.rolling(28).sum() / tr_u.rolling(28).sum().replace(0, np.nan)
            ind["uo"] = safe(100 * (4 * a7 + 2 * a14 + a28) / 7, 50.0)

            return ind

        close_indicators = _compute_indicators(
            day_bars["close"], day_bars["high"], day_bars["low"],
            day_bars["volume"], day_bars["open"]
        )

        open_bars = day_bars[day_bars.index.time <= dt_time(10, 0)]
        if len(open_bars) >= 10:
            open_indicators = _compute_indicators(
                open_bars["close"], open_bars["high"], open_bars["low"],
                open_bars["volume"], open_bars["open"]
            )
        else:
            open_indicators = {}

        result = {}

        if _t1_map_available:
            close_df      = pd.DataFrame([close_indicators])
            close_renamed = _rename(close_df, prefix="t1_close")
            for col in close_renamed.columns:
                val = close_renamed.iloc[0][col]
                if pd.notna(val):
                    try:
                        result[col] = float(val)
                    except (TypeError, ValueError):
                        pass

            if open_indicators:
                open_df      = pd.DataFrame([open_indicators])
                open_renamed = _rename(open_df, prefix="t1_open")
                for col in open_renamed.columns:
                    val = open_renamed.iloc[0][col]
                    if pd.notna(val) and col not in result:
                        try:
                            result[col] = float(val)
                        except (TypeError, ValueError):
                            pass
        else:
            for k, val in close_indicators.items():
                result[f"t1_close_{k}"] = val
            for k, val in open_indicators.items():
                result.setdefault(f"t1_open_{k}", val)

        return result

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Gain correction helpers
# ---------------------------------------------------------------------------

def _apply_gain_rank_correction(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_prefix: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if 'target_gain_pct' not in predictions_df.columns:
        return predictions_df

    gains    = predictions_df['target_gain_pct']
    gain_std = gains.std()

    if gain_std >= 1.0:
        return predictions_df

    logger.warning(
        f"  ⚠️  FLAT GAIN ESTIMATES detected (std={gain_std:.4f}%). "
        f"All stocks showing ~{gains.mean():.1f}% gain."
        f"\n      Applying rank-based correction "
        f"(floor={GAIN_FLOOR}%, ceiling={GAIN_CEILING}%)."
    )

    df    = predictions_df.copy()
    probs = df['explosion_probability']

    base_ranks = probs.rank(pct=True)
    corrected_gains = GAIN_FLOOR + base_ranks * (GAIN_CEILING - GAIN_FLOOR)

    if _uses_lowercase(feature_prefix):
        rsi_col = f"{feature_prefix}_rsi_14"
        vol_col = f"{feature_prefix}_volume_ratio"
    else:
        rsi_col = f"{feature_prefix}_RSI_14"
        vol_col = f"{feature_prefix}_Volume_Ratio"

    if features_df is not None and not features_df.empty:
        feat_indexed = features_df.set_index("symbol") if "symbol" in features_df.columns else features_df

        if rsi_col in feat_indexed.columns:
            rsi_map  = feat_indexed[rsi_col]
            rsi_vals = df['symbol'].map(rsi_map) if 'symbol' in df.columns else None
            if rsi_vals is not None and rsi_vals.notna().sum() > 5:
                rsi_score = 1.0 - (abs(rsi_vals.fillna(55) - 60) / 40).clip(0, 1)
                corrected_gains += rsi_score * 5.0
                logger.info(f"  Applied RSI-based gain adjustment (mean RSI: {rsi_vals.mean():.1f})")

        if vol_col in feat_indexed.columns:
            vol_map  = feat_indexed[vol_col]
            vol_vals = df['symbol'].map(vol_map) if 'symbol' in df.columns else None
            if vol_vals is not None and vol_vals.notna().sum() > 5:
                vol_score = (vol_vals.fillna(1.0) - 1.0).clip(0, 4) / 4.0
                corrected_gains += vol_score * 8.0
                logger.info(f"  Applied volume-based gain adjustment (mean vol_ratio: {vol_vals.mean():.2f})")

    corrected_gains = corrected_gains.clip(GAIN_FLOOR, GAIN_CEILING * 1.5)

    df['target_gain_pct']  = corrected_gains
    df['target_gain_low']  = corrected_gains * 0.65
    df['target_gain_high'] = corrected_gains * 1.40

    if 'current_price' in df.columns:
        df['target_price']      = df['current_price'] * (1 + df['target_gain_pct']  / 100)
        df['target_price_low']  = df['current_price'] * (1 + df['target_gain_low']  / 100)
        df['target_price_high'] = df['current_price'] * (1 + df['target_gain_high'] / 100)

    logger.info(
        f"  Corrected gain range: {corrected_gains.min():.1f}%–{corrected_gains.max():.1f}%  "
        f"std={corrected_gains.std():.1f}%"
    )

    return df


def _get_calibrated_gain_estimate(probability: float) -> float:
    if probability >= 0.95: return 40.0
    if probability >= 0.90: return 30.0
    if probability >= 0.80: return 20.0
    if probability >= 0.70: return 15.0
    if probability >= 0.60: return 10.0
    if probability >= 0.50: return 7.0
    return 3.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=1500)
    parser.add_argument("--top-n",       type=int, default=50)
    parser.add_argument("--verbose",     "-v", action="store_true")
    parser.add_argument(
        "--no-t1",
        action="store_true",
        help="Skip T-1 intraday fetch (fastest — uses only TV screener data)"
    )

    args   = parser.parse_args()
    logger = setup_logging(args.verbose)

    prediction_date = get_next_trading_day()

    logger.info("=" * 80)
    logger.info("ML SCREENING & PREDICTION")
    logger.info("=" * 80)
    logger.info(f"Prediction date (trading session): {prediction_date}")
    logger.info(f"Top N to store: {args.top_n}")
    logger.info("=" * 80)

    screener = SmartScreener(logger=logger)

    try:
        predictor = ExplosionPredictor()
        supabase  = MLPredictionSupabaseClient({})
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1

    model_prefix = predictor.model_feature_prefix
    logger.info(f"✓ Model feature prefix detected: '{model_prefix}'")
    logger.info(f"  All TV screener features will be mapped to {model_prefix}_* columns")
    logger.info(f"  Lowercase keys: {_uses_lowercase(model_prefix)}")

    # ── STEP 1: SCREENING ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: INTELLIGENT SCREENING")
    logger.info("=" * 80)

    screened_df = screener.screen_with_tradingview(max_results=args.max_results)
    if screened_df.empty:
        logger.error("No stocks passed screening")
        return 1
    logger.info(f"✓ Screened {len(screened_df)} stocks")

    # ── STEP 2: BUILD FEATURES FROM TV DATA ──────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: BUILD FEATURES FROM TRADINGVIEW DATA")
    logger.info("=" * 80)
    logger.info(f"Mapping TV screener columns → {model_prefix}_* model features.")

    enriched_stocks = []
    failed_count    = 0
    t1_feature_hits = 0

    for _, row in screened_df.iterrows():
        try:
            symbol_full = str(row.get("symbol", ""))
            if ":" in symbol_full:
                exchange, symbol = symbol_full.split(":", 1)
            else:
                symbol   = symbol_full
                exchange = "NASDAQ"

            symbol = symbol.strip().upper()
            if not symbol:
                failed_count += 1
                continue

            if exchange == "OTC" or len(symbol) > 5 or "." in symbol:
                failed_count += 1
                continue

            row_dict = row.to_dict()
            features = build_features_from_tv_data(row_dict, symbol, feature_prefix=model_prefix)
            features["exchange"] = exchange

            if "current_price" not in features or not features["current_price"]:
                failed_count += 1
                continue

            n_model_feats = sum(1 for k in features if k.startswith(f"{model_prefix}_"))
            if n_model_feats > 3:
                t1_feature_hits += 1

            enriched_stocks.append(features)

        except Exception as e:
            logger.debug(f"Error processing row: {e}")
            failed_count += 1

    logger.info(f"✓ Built features for {len(enriched_stocks)} stocks ({failed_count} skipped)")
    logger.info(
        f"  Stocks with ≥3 real {model_prefix}_ indicator values: "
        f"{t1_feature_hits}/{len(enriched_stocks)}"
    )

    if t1_feature_hits == 0:
        logger.warning(
            f"  ⚠️  ZERO stocks have real indicator values from TV screener."
            "\n      Screener returned only default columns (no RSI/ATR/etc)."
            "\n      All probabilities will be near-identical."
            "\n      Check tradingview-scraper version — needs >=0.4.19."
        )

    if not enriched_stocks:
        logger.error("Failed to build features for any stocks")
        return 1

    # ── STEP 3: T-1 INTRADAY INDICATOR ENRICHMENT ────────────────────────────
    if not args.no_t1:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: T-1 INTRADAY INDICATOR ENRICHMENT")
        logger.info("=" * 80)
        logger.info(
            "Computing RSI, MACD, ATR, Bollinger, ADX, Stoch, OBV, CMF etc. "
            "from 5-min bars for top 200 candidates."
        )

        top_200 = [s["symbol"] for s in enriched_stocks[:200]]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time, random

        t1_map   = {}
        t1_count = 0

        def fetch_with_jitter(sym):
            time.sleep(random.uniform(0.05, 0.2))
            return sym, fetch_t1_data_for_symbol(sym, logger)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_with_jitter, sym): sym for sym in top_200}
            for i, future in enumerate(as_completed(futures), 1):
                if i % 50 == 0:
                    logger.info(f"  T-1 progress: {i}/{len(top_200)} | enriched: {t1_count}")
                sym, t1_data = future.result()
                if t1_data:
                    t1_map[sym] = t1_data
                    t1_count   += 1

        for stock in enriched_stocks:
            sym = stock["symbol"]
            if sym in t1_map:
                stock.update(t1_map[sym])

        sample = next((s for s in enriched_stocks if s["symbol"] in t1_map), None)
        if sample:
            logger.info(
                f"  Sample ({sample['symbol']}): "
                f"t1_close_RSI_14={sample.get('t1_close_RSI_14', 'MISSING')}, "
                f"t1_close_ATR_14={sample.get('t1_close_ATR_14', 'MISSING')}"
            )

        logger.info(f"✓ T-1 indicator enrichment: {t1_count}/{len(top_200)} stocks")
    else:
        logger.info("\nSTEP 3: Skipped (--no-t1 flag)")

    # ── STEP 4: PREPARE FEATURES + PRE-FLIGHT VARIANCE CHECK ─────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: PREPARE FEATURES + PRE-FLIGHT VARIANCE CHECK")
    logger.info("=" * 80)

    features_df  = pd.DataFrame(enriched_stocks)
    model_cols   = [c for c in features_df.columns if c.startswith(f"{model_prefix}_")]
    t1_open_cols = [c for c in features_df.columns if c.startswith("t1_open_")]

    logger.info(f"✓ Feature matrix: {len(features_df)} stocks × {len(features_df.columns)} raw columns")
    logger.info(f"  {model_prefix}_ features present: {len(model_cols)}")
    logger.info(f"  t1_open_ features present:  {len(t1_open_cols)}")

    # FIX 5: Pre-flight variance check — catches identical-probability root cause
    # before the model runs so we get an actionable error message.
    key_indicators = [
        "t1_close_RSI_14", "t1_close_ATR_14", "t1_close_ADX_14",
        "t1_close_Volume_Ratio", "t1_close_MACD_12_26_9",
        f"{model_prefix}_RSI_14", f"{model_prefix}_ATR_14",
        f"{model_prefix}_rsi_14", f"{model_prefix}_atr_14",
    ]
    zero_var_indicators = []
    for col in key_indicators:
        if col in features_df.columns:
            col_data = pd.to_numeric(features_df[col], errors='coerce').dropna()
            std_val  = col_data.std() if len(col_data) > 1 else 0.0
            logger.info(
                f"  {col}: n={len(col_data)}, "
                f"mean={col_data.mean():.2f}, "
                f"std={std_val:.4f}"
            )
            if std_val < 1e-4 and len(col_data) > 5:
                zero_var_indicators.append(col)

    if zero_var_indicators:
        logger.warning(
            f"\n  ⚠️  ZERO-VARIANCE INDICATOR COLUMNS DETECTED: {zero_var_indicators}"
            "\n      These columns are constant — model will output near-identical probabilities."
            "\n      Root causes to check:"
            "\n        1. Too few stocks in features_df (currently "
            f"{len(features_df)}) — HV filters may be too tight"
            "\n        2. TV screener returned no indicator columns (tradingview-scraper version)"
            "\n        3. T-1 yfinance fetch produced no variance (all stocks similar intraday)"
            "\n      → Run with --no-t1 to isolate whether T-1 or TV data is the culprit"
        )
    elif len(features_df) < 20:
        logger.warning(
            f"\n  ⚠️  Only {len(features_df)} stocks in feature matrix."
            "\n      With < 20 stocks there is not enough distribution for meaningful probabilities."
            "\n      The HV / volume-ratio screener filters are almost certainly too tight."
            "\n      Check ml_models/learned_filters.json and re-run:"
            "\n        python ml_track_comprehensive_accuracy.py --filters-only"
        )
    else:
        logger.info(f"  ✅ Feature variance looks healthy ({len(features_df)} stocks)")

    if len(model_cols) == 0:
        logger.warning(
            f"  ⚠️  NO {model_prefix}_ features in DataFrame. "
            "Check that build_features_from_tv_data and fetch_t1_data_for_symbol "
            "are producing keys with the correct prefix."
        )

    # ── STEP 5: ML PREDICTION ─────────────────────────────────────────────────
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

    # FIX 5: Post-prediction sanity check
    prob_std = predictions_df['explosion_probability'].std()
    if prob_std < 0.001 and len(predictions_df) > 5:
        logger.error(
            f"\n  ❌ POST-PREDICTION: prob_std={prob_std:.6f} — all probabilities identical."
            "\n     The model received uniform feature inputs."
            "\n     Most likely fix: recompute learned_filters.json by running:"
            "\n       python ml_track_comprehensive_accuracy.py --filters-only"
            "\n     Then re-run. If the problem persists with > 100 stocks,"
            "\n     the issue is in feature building (TV screener indicator columns absent)."
        )

    # ── STEP 6: GAIN CORRECTION ───────────────────────────────────────────────
    if 'target_gain_pct' in predictions_df.columns:
        bad_gain_mask = (
            predictions_df['target_gain_pct'].isna() |
            (predictions_df['target_gain_pct'].abs() < 0.5) |
            (predictions_df['target_gain_pct'] > 500)
        )
        if bad_gain_mask.any():
            n_bad = bad_gain_mask.sum()
            logger.info(f"  Individual gain fallback applied to {n_bad} stocks")
            predictions_df.loc[bad_gain_mask, 'target_gain_pct'] = (
                predictions_df.loc[bad_gain_mask, 'explosion_probability']
                .apply(_get_calibrated_gain_estimate)
            )
            predictions_df.loc[bad_gain_mask, 'target_gain_low']  = (
                predictions_df.loc[bad_gain_mask, 'target_gain_pct'] * 0.5
            )
            predictions_df.loc[bad_gain_mask, 'target_gain_high'] = (
                predictions_df.loc[bad_gain_mask, 'target_gain_pct'] * 1.8
            )

        predictions_df = _apply_gain_rank_correction(
            predictions_df, features_df, model_prefix, logger
        )

        if 'current_price' in predictions_df.columns:
            predictions_df['target_price'] = (
                predictions_df['current_price'] * (1 + predictions_df['target_gain_pct'] / 100)
            )
            predictions_df['target_price_low'] = (
                predictions_df['current_price'] * (1 + predictions_df['target_gain_low'] / 100)
            )
            predictions_df['target_price_high'] = (
                predictions_df['current_price'] * (1 + predictions_df['target_gain_high'] / 100)
            )

    log_probability_distribution(
        predictions_df, logger,
        label=f"All {len(predictions_df)} screened stocks"
    )

    # ── STEP 7: TOP PREDICTIONS ───────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 7: TOP {args.top_n} PREDICTIONS")
    logger.info("=" * 80)
    logger.info(
        f"  Screened: {len(screened_df)}  →  "
        f"Scored: {len(predictions_df)}  →  "
        f"Storing top {args.top_n}"
    )

    top_predictions = predictions_df.head(args.top_n)

    logger.info(f"\nTop {min(20, len(top_predictions))} Predictions for {prediction_date}:")
    logger.info("-" * 100)
    logger.info(f"{'#':<4} {'Symbol':<8} {'Signal':<13} {'Prob':<8} {'Price':<10} {'Target':<10} {'Gain':<8} {'Has Inds?'}")
    logger.info("-" * 100)

    for rank, (_, row) in enumerate(top_predictions.head(20).iterrows(), 1):
        current_price = row.get('current_price', 0)
        has_real_data = any(
            k.startswith(f"{model_prefix}_RSI") or k.startswith(f"{model_prefix}_ATR")
            for k in row.index
            if pd.notna(row.get(k))
        )
        logger.info(
            f"{rank:<4} {row['symbol']:<8} {row['signal']:<13} "
            f"{row['explosion_probability']*100:>6.2f}%  "
            f"${current_price:>8.2f}  "
            f"${row.get('target_price', 0):>8.2f}  "
            f"+{row.get('target_gain_pct', 0):>5.1f}%"
            f"  {'✓' if has_real_data else '—'}"
        )

    # ── STEP 8: STORE PREDICTIONS (FIX 3: upsert) ────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: STORE PREDICTIONS")
    logger.info("=" * 80)

    predictions_list = [
        {
            'symbol':                row['symbol'],
            'exchange':              row.get('exchange', 'NASDAQ'),
            'prediction_date':       prediction_date,
            'explosion_probability': float(row['explosion_probability']),
            'prediction':            int(row['prediction']),
            'signal':                row['signal'],
            'target_gain_pct':       float(row.get('target_gain_pct', 0)),
            'target_gain_low':       float(row.get('target_gain_low', 0)),
            'target_gain_high':      float(row.get('target_gain_high', 0)),
            'current_price':         float(row.get('current_price', 0)),
            'target_price':          float(row.get('target_price', 0)),
            'target_price_low':      float(row.get('target_price_low', 0)),
            'target_price_high':     float(row.get('target_price_high', 0)),
            'model_version':         f"{model_prefix}_v6",
        }
        for _, row in top_predictions.iterrows()
    ]

    if predictions_list:
        count = supabase.write_predictions_upsert(predictions_list)
        logger.info(f"✓ Wrote {count} predictions for trading session: {prediction_date}")

    # ── STEP 9: SCREENING LOG ─────────────────────────────────────────────────
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
        'model_version':      f"{model_prefix}_features_v6",
    }
    supabase.write_screening_log(screening_log)

    csv_path = Path(f"ml_screening_results_{prediction_date}.csv")
    top_predictions.to_csv(csv_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("✓ PREDICTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Stocks screened:    {len(screened_df)}")
    logger.info(f"  Stocks scored:      {len(predictions_df)}")
    logger.info(f"  Predictions stored: {len(predictions_list)}")
    logger.info(f"  Model prefix used:  {model_prefix}")
    logger.info(f"  Prob std:           {predictions_df['explosion_probability'].std():.4f}")
    logger.info(f"  Gain std:           {predictions_df['target_gain_pct'].std():.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
