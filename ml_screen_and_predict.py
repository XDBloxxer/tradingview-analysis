#!/usr/bin/env python3
"""
ML Stock Screener & Predictor

FIXES IN THIS VERSION (2026-03-12 v9):

FIX 1 — fetch_t1_data_for_symbol now computes real indicators from 5-min bars.

FIX 2 — build_features_from_tv_data lowercases keys for t3/t5/t10 prefix models.

FIX 3 — write_predictions_upsert instead of insert-skip-on-duplicate.

FIX 4 — SmartScreener._load_learned_filters clamps aggressive HV/vol-ratio minimums.

FIX 5 — Pre-flight and post-prediction variance diagnostics.

FIX 6 — Hybrid model prefix handling: TV screener data mapped to t1_close_*
         only.  t3_/t5_/t10_* are NOT populated from the TV screener row.
         (Reverted the old fill_flat_prefixes=True path in
         build_features_from_tv_data which was writing identical TV-screener
         values into all three flat prefixes, corrupting 2/3 of the hybrid
         model's flat-prefix features.  See FIX 9 for the correct source.)

FIX 7 — t1_open_* features now reliably populated (lowered min bar threshold,
         extended open window, fallback to close indicators).

FIX 8 — Added missing indicators to _compute_indicators:
  TSI, Keltner Channel, Donchian Channel, VWAP, ATR slope.

FIX 9 (revised) — t3_/t5_/t10_* columns now populated from REAL daily-bar
  snapshots at the correct calendar offsets, not from T-1 intraday data.

  ROOT CAUSE: The model was trained with t3_*/t5_*/t10_* features representing
  indicator snapshots taken 3, 5, and 10 calendar days before detection_date,
  computed from daily OHLCV bars via multiday_feature_collector.py.  The
  original FIX 9 implementation paperd over a feature-pipeline gap by copying
  yesterday's intraday bar indicators into all three prefix columns, causing
  the model to receive identical data under three different names and degrading
  any signal that depends on how conditions differed at T-3 vs T-5 vs T-10.

  The revised fix calls _fetch_real_multiday_features(), which:
    1. Fetches daily OHLCV bars from yfinance (same as training time).
    2. Calls _snapshot_for_offset() from multiday_feature_collector for each
       offset (3, 5, 10 calendar days before detection_date).
    3. Uses the same _compute_indicators + PANDAS_TA_TO_BASE pipeline as the
       training data pipeline, so feature names and scaling match exactly.
  The result is that prediction-time t3_*/t5_*/t10_* values are structurally
  identical to what the model saw during training.

FIX 10 — get_next_trading_day now queries the daily_winners table for the most
  recent detection_date and returns the next weekday after it, rather than
  deriving the date from wall-clock time. This prevents mis-dating when GitHub
  Actions runs late or on the wrong side of midnight. Falls back to the
  original time-based logic if no winners data exists yet.
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
    # RSI — use today's value only; RSI[1] is yesterday's bar and must NOT
    # overwrite RSI_14 with a stale reading.
    "RSI":                          "RSI_14",
    # Stochastic — today's K/D only; [1] variants are T-1 bars.
    "Stoch.K":                      "STOCHk_14_3_3",
    "Stoch.D":                      "STOCHd_14_3_3",
    # Williams %R — today's value only; W.R[1] is T-1.
    "W.R":                          "WILLR_14",
    "MACD.macd":                    "MACD_12_26_9",
    "MACD.signal":                  "MACDs_12_26_9",
    "BB.upper":                     "BBU_20_2.0_2.0",
    "BB.lower":                     "BBL_20_2.0_2.0",
    "BB.basis":                     "BBM_20_2.0_2.0",
    "BBPower":                      "BBP_20_2.0_2.0",
    "EMA5":                         "EMA_5",
    "EMA10":                        "EMA_10",
    "EMA20":                        "EMA_20",
    # EMA30 is NOT mapped: the TV screener exposes a 30-period EMA while the
    # model feature EMA_26 is a 26-period EMA.  Feeding a wrong-period value
    # corrupts the feature; EMA_26 is left NaN and imputed by the model.
    "EMA50":                        "EMA_50",
    "SMA5":                         "SMA_5",
    "SMA10":                        "SMA_10",
    "SMA20":                        "SMA_20",
    # SMA30 is NOT mapped: the TV screener's 30-period SMA is a distinct
    # indicator from the model's SMA_50 (50-period).  The old mapping silently
    # duplicated SMA_50 with the wrong period value.
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

# ---------------------------------------------------------------------------
# Gain estimation curve — probability → expected intraday high % gain.
#
# These values are anchored to the actual distribution of explosive mover
# intraday highs (from daily_winners / ml_prediction_accuracy history):
#   - p ≥ 0.95: truly explosive moves (100 %+ seen regularly on STRONG BUY)
#   - p ≥ 0.90: strong movers, typically 50–80 % intraday high
#   - p ≥ 0.80: solid movers, 30–50 %
#   - Below 0.60: conservative — model is not confident
#
# The curve is used ONLY when the gain regressor and isotonic calibrator are
# both unavailable (rank_fallback path).  When the regressor is healthy it
# predicts freely from the training data, so there is no ceiling there.
# ---------------------------------------------------------------------------
_GAIN_CURVE: list[tuple[float, float]] = [
    # (min_probability, base_gain_pct)
    (0.95, 100.0),
    (0.90,  60.0),
    (0.85,  45.0),
    (0.80,  35.0),
    (0.75,  28.0),
    (0.70,  22.0),
    (0.65,  17.0),
    (0.60,  13.0),
    (0.55,  10.0),
    (0.50,   7.0),
    (0.00,   4.0),
]

_LOWERCASE_PREFIXES = ("t3", "t5", "t10")

SCREENER_HV_MIN_CAP    = 30.0
SCREENER_VOL_RATIO_CAP = 2.5

# FIX 6: All flat-bar prefixes used by the base CSV training data.
_FLAT_PREFIXES = ("t3", "t5", "t10")

# FIX 7: Minimum bars needed for open-window indicator calculation.
T1_OPEN_MIN_BARS = 5


def _uses_lowercase(prefix: str) -> bool:
    return any(prefix == p or prefix.startswith(p + "_") for p in _LOWERCASE_PREFIXES)


def _is_hybrid_model(predictor: "ExplosionPredictor") -> bool:
    """Return True if the model has both t1_ and t3/t5/t10 features."""
    has_t1   = any(f.startswith("t1_") for f in predictor.feature_names)
    has_flat = any(
        f.startswith("t3_") or f.startswith("t5_") or f.startswith("t10_")
        for f in predictor.feature_names
    )
    return has_t1 and has_flat


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
            f"\n      Check that build_features_from_tv_data is populating the correct prefix."
        )
    elif probs.std() < 0.05:
        logger.warning(f"  ⚠️  LOW PROB STD ({probs.std():.4f}) — limited feature discrimination.")
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


def get_next_trading_day(supabase: "MLPredictionSupabaseClient") -> str:
    """
    FIX 10: Determine the prediction date as the next trading weekday after
    the most recent detection_date in daily_winners.

    This is more reliable than deriving the date from wall-clock time because
    GitHub Actions can run late, early, or on the wrong side of midnight.

    Logic:
      1. Query daily_winners for the latest detection_date.
      2. Add one calendar day.
      3. Skip forward over weekends until we land on a weekday.
         e.g. last_winners = Friday → +1 = Saturday → skip → Monday ✓

    Fallback: if the table is empty or the query fails, fall back to the
    original time-based approach so the workflow never hard-crashes.
    """
    logger = logging.getLogger(__name__)

    try:
        response = (
            supabase.client.table("daily_winners")
            .select("detection_date")
            .order("detection_date", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            last_date_str = response.data[0]["detection_date"]
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            prediction_day = last_date + timedelta(days=1)
            while prediction_day.weekday() >= 5:  # 5=Sat, 6=Sun
                prediction_day += timedelta(days=1)
            logger.info(
                f"Prediction date derived from daily_winners: "
                f"last={last_date}  →  next trading day={prediction_day}"
            )
            return prediction_day.isoformat()
    except Exception as e:
        logger.warning(f"Could not fetch most recent winners date: {e}")

    # ── Fallback: wall-clock based ────────────────────────────────────────
    logger.warning("Falling back to time-based prediction date.")
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

                    if key in ("min_hv10", "min_hv20") and float(value) > SCREENER_HV_MIN_CAP:
                        self.logger.info(
                            f"  Clamping {key}={value:.2f} → {SCREENER_HV_MIN_CAP} "
                            f"(raw winner p10 too aggressive for broad screening)"
                        )
                        value = SCREENER_HV_MIN_CAP

                    if key in ("min_volume_ratio", "min_relative_volume") and float(value) > SCREENER_VOL_RATIO_CAP:
                        self.logger.info(f"  Clamping {key}={value:.2f} → {SCREENER_VOL_RATIO_CAP}")
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
                    "  ⚠️  Using default TV columns only (no RSI/ATR/etc). "
                    "Upgrade tradingview-scraper to >=0.4.19 for indicator columns."
                )
                return df

            self.logger.warning("Screener returned no data or error status")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# FIX 6: Feature building — fills ALL relevant prefix groups
# ---------------------------------------------------------------------------

def build_features_from_tv_data(
    row: dict,
    symbol: str,
    feature_prefix: str = "t1_close",
) -> dict:
    """
    Convert a single TradingView screener row into a feature dict.

    Writes values ONLY to the requested feature_prefix (e.g. t1_close_*).
    t3_/t5_/t10_* columns must NOT be populated from TV screener data —
    those prefixes represent daily-bar snapshots from the training CSV and
    must be filled exclusively from T-1 yfinance intraday indicators via
    fetch_t1_data_for_symbol → _write_flat_prefix_features.  Filling them
    with today's TV screener row (as the old fill_flat_prefixes=True path
    did) made t3/t5/t10 identical to each other and to t1, corrupting 2/3
    of the hybrid model's flat-prefix features on every production run.
    """
    result = {
        "symbol":   symbol,
        "exchange": "NASDAQ",
    }

    seen_targets: set = set()

    def _write(prefix: str, model_name: str, fval: float):
        target = f"{prefix}_{model_name}"
        if _uses_lowercase(prefix):
            target = target.lower()
        if target not in seen_targets:
            result[target] = fval
            seen_targets.add(target)

    for tv_col, model_name in TV_TO_MODEL_BASE.items():
        value = row.get(tv_col)
        if value is None:
            value = row.get(tv_col.lower())
        if value is None:
            for k in row:
                if k.lower() == tv_col.lower():
                    value = row[k]
                    break

        if value is None:
            continue

        try:
            fval = float(value)
            if np.isnan(fval) or np.isinf(fval):
                continue
        except (TypeError, ValueError):
            continue

        # Write to the primary model prefix (e.g. t1_close_RSI_14).
        # t3_/t5_/t10_* columns are intentionally NOT written here — they are
        # populated from T-1 yfinance intraday data in fetch_t1_data_for_symbol.
        _write(feature_prefix, model_name, fval)

    close_val = row.get("close") or row.get("Close")
    if close_val is not None:
        try:
            result["current_price"] = float(close_val)
        except (TypeError, ValueError):
            pass

    # Derived features
    close = result.get("current_price") or result.get(f"{feature_prefix}_Close")

    def _derived(prefix: str, close_v: float):
        if _uses_lowercase(prefix):
            ema20 = result.get(f"{prefix}_ema_20")
            ema50 = result.get(f"{prefix}_ema_50")
            ema10 = result.get(f"{prefix}_ema_10")
            sma20 = result.get(f"{prefix}_sma_20")
            if ema20: result[f"{prefix}_price_vs_ema20"] = (close_v / ema20 - 1) * 100
            if sma20: result[f"{prefix}_price_vs_sma20"] = (close_v / sma20 - 1) * 100
            if ema20 and ema50: result[f"{prefix}_ema_12_26_diff"] = ema20 - ema50
            if ema10 and ema20: result[f"{prefix}_sma_20_50_diff"] = ema10 - ema20
        else:
            ema20 = result.get(f"{prefix}_EMA_20")
            ema50 = result.get(f"{prefix}_EMA_50")
            ema10 = result.get(f"{prefix}_EMA_10")
            sma20 = result.get(f"{prefix}_SMA_20")
            if ema20: result[f"{prefix}_Price_vs_EMA20"] = (close_v / ema20 - 1) * 100
            if sma20: result[f"{prefix}_Price_vs_SMA20"] = (close_v / sma20 - 1) * 100
            if ema20 and ema50: result[f"{prefix}_EMA_12_26_Diff"] = ema20 - ema50
            if ema10 and ema20: result[f"{prefix}_SMA_20_50_Diff"] = ema10 - ema20

    if close and close > 0:
        _derived(feature_prefix, close)

    return result


# ---------------------------------------------------------------------------
# T-1 intraday indicator computation
# ---------------------------------------------------------------------------

def _compute_indicators(c: pd.Series, h: pd.Series, l: pd.Series,
                         v: pd.Series, o: pd.Series) -> dict:
    """
    Compute comprehensive technical indicators from OHLCV series.

    FIX 8: Added TSI, Keltner Channel, Donchian Channel, VWAP, and ATR slope.
    """
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

    # ── RSI (multiple periods) ──────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    for period, col_name in [(7, "rsi7"), (14, "rsi"), (21, "rsi21"), (28, "rsi28")]:
        ag = gain.ewm(com=period - 1, min_periods=period).mean()
        al = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = ag / al.replace(0, np.nan)
        ind[col_name] = safe(100 - (100 / (1 + rs)), 50.0)
    ind["rsi[1]"] = ind["rsi"]
    ind["rsi14"]  = ind["rsi"]   # alias for column-map dedup

    # ── MACD ───────────────────────────────────────────────────────────────
    ema12     = c.ewm(span=12, adjust=False).mean()
    ema26     = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    ind["macd.macd"]   = safe(macd_line)
    ind["macd.signal"] = safe(macd_sig)
    ind["macd_diff"]   = safe(macd_line - macd_sig)

    # ── Moving averages ─────────────────────────────────────────────────────
    for n in [5, 10, 12, 20, 26, 50]:
        ind[f"ema{n}"] = safe(c.ewm(span=n, adjust=False).mean(), float(c.mean()))
    for n in [5, 10, 20, 50]:
        ind[f"sma{n}"] = safe(c.rolling(n).mean(), float(c.mean()))

    # WMA approximations
    for n in [10, 20]:
        weights = np.arange(1, n + 1, dtype=float)
        wma_vals = c.rolling(n).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum()
            if len(x) >= 2 else float(x.iloc[-1]),
            raw=True
        )
        ind[f"wma{n}"] = safe(wma_vals, float(c.mean()))

    sma20_v = ind.get("sma20") or float(c.mean())
    ema20_v = ind.get("ema20") or float(c.mean())
    ema10_v = ind.get("ema10") or float(c.mean())
    ema12_v = ind.get("ema12") or float(c.mean())
    ema26_v = ind.get("ema26") or float(c.mean())

    if sma20_v: ind["price_vs_sma20"] = (close_v / sma20_v - 1) * 100
    if ema20_v: ind["price_vs_ema20"] = (close_v / ema20_v - 1) * 100
    ind["ema_12_26_diff"] = ema12_v - ema26_v
    ind["sma_20_50_diff"] = ind.get("sma20", 0) - ind.get("sma50", 0)

    # ── Stochastic ─────────────────────────────────────────────────────────
    lo14  = l.rolling(14).min()
    hi14  = h.rolling(14).max()
    rng14 = (hi14 - lo14).replace(0, np.nan)
    stk   = (100 * (c - lo14) / rng14).rolling(3).mean()
    std   = stk.rolling(3).mean()
    ind["stoch.k"]    = safe(stk, 50.0)
    ind["stoch.d"]    = safe(std, 50.0)
    ind["stoch.k[1]"] = ind["stoch.k"]
    ind["stoch.d[1]"] = ind["stoch.d"]
    ind["w.r"]        = safe(-100 * (hi14 - c) / rng14, -50.0)

    # ── ATR ────────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    for period, col_name in [(7, "atr7"), (14, "atr"), (20, "atr20")]:
        ind[col_name] = safe(tr.rolling(period).mean(), 0.5)
    ind["atr14"] = ind["atr"]   # alias

    # ATR slope
    atr14_series = tr.rolling(14).mean()
    if len(atr14_series.dropna()) >= 6:
        atr_slope = atr14_series.diff(5)
        ind["atr_pct"] = safe(atr_slope, 0.0)
    else:
        ind["atr_pct"] = 0.0

    # ── ADX / DMI ──────────────────────────────────────────────────────────
    up_move = h.diff()
    dn_move = -l.diff()
    pdm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=c.index)
    ndm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=c.index)
    atr14 = tr.rolling(14).mean().replace(0, np.nan)
    pdi   = 100 * pdm.rolling(14).mean() / atr14
    ndi   = 100 * ndm.rolling(14).mean() / atr14
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    ind["adx"]    = safe(dx.rolling(14).mean(), 20.0)
    ind["adx+di"] = safe(pdi, 20.0)
    ind["adx-di"] = safe(ndi, 20.0)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    ind["bb.upper"]  = safe(bb_up,  close_v)
    ind["bb.lower"]  = safe(bb_lo,  close_v)
    ind["bb.middle"] = safe(bb_mid, close_v)
    ind["bb_width"]  = safe((bb_up - bb_lo) / bb_mid.replace(0, np.nan) * 100, 0.0)
    ind["bbpower"]   = safe((c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan), 0.5)

    # Keltner Channel
    kc_mid  = c.ewm(span=20, adjust=False).mean()
    kc_atr  = tr.rolling(10).mean()
    kc_mult = 2.0
    ind["keltner_upper"]  = safe(kc_mid + kc_mult * kc_atr, close_v)
    ind["keltner_lower"]  = safe(kc_mid - kc_mult * kc_atr, close_v)
    ind["keltner_middle"] = safe(kc_mid, close_v)

    # Donchian Channel (20-period)
    dc_up  = h.rolling(20).max()
    dc_lo  = l.rolling(20).min()
    dc_mid = (dc_up + dc_lo) / 2
    ind["donchian_upper"]  = safe(dc_up,  close_v)
    ind["donchian_lower"]  = safe(dc_lo,  close_v)
    ind["donchian_middle"] = safe(dc_mid, close_v)

    # ── Volume ─────────────────────────────────────────────────────────────
    vm5  = v.rolling(5).mean()
    vm10 = v.rolling(10).mean()
    vm20 = v.rolling(20).mean()
    ind["volume_sma5"]  = safe(vm5,  float(v.mean()))
    ind["volume_sma10"] = safe(vm10, float(v.mean()))
    ind["volume_sma20"] = safe(vm20, float(v.mean()))
    ind["volume_ratio"] = safe(v / vm20.replace(0, np.nan), 1.0)

    # ── OBV ────────────────────────────────────────────────────────────────
    obv_vals = [0.0]
    c_arr, v_arr = c.values, v.values
    for i in range(1, len(c_arr)):
        if   c_arr[i] > c_arr[i - 1]: obv_vals.append(obv_vals[-1] + v_arr[i])
        elif c_arr[i] < c_arr[i - 1]: obv_vals.append(obv_vals[-1] - v_arr[i])
        else:                          obv_vals.append(obv_vals[-1])
    ind["obv"] = float(obv_vals[-1])

    # ── CMF ────────────────────────────────────────────────────────────────
    mf_mult   = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    ind["cmf"] = safe(mf_mult * v / v.rolling(20).sum().replace(0, np.nan), 0.0)

    # ── CCI ────────────────────────────────────────────────────────────────
    tp    = (h + l + c) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    ind["cci20"] = safe((tp - tp_ma) / (0.015 * tp_md.replace(0, np.nan)), 0.0)

    # ── AO / MOM / ROC ─────────────────────────────────────────────────────
    ind["ao"]  = safe((h + l).rolling(5).mean() / 2 - (h + l).rolling(34).mean() / 2, 0.0)
    ind["mom"] = safe(c.diff(10), 0.0)
    ind["roc"] = safe(c.pct_change(10) * 100, 0.0)

    # TSI (True Strength Index)
    pc        = c.diff(1)
    double_smooth_pc  = pc.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    double_smooth_apc = pc.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi_denom = double_smooth_apc.replace(0, np.nan)
    tsi_series = 100 * double_smooth_pc / tsi_denom
    ind["tsi"]       = safe(tsi_series, 0.0)
    ind["kst"]       = ind["tsi"]

    # ── Ultimate Oscillator ────────────────────────────────────────────────
    bp   = c - pd.concat([l, c.shift()], axis=1).min(axis=1)
    tr_u = (pd.concat([h, c.shift()], axis=1).max(axis=1)
            - pd.concat([l, c.shift()], axis=1).min(axis=1))
    a7  = bp.rolling(7).sum()  / tr_u.rolling(7).sum().replace(0, np.nan)
    a14 = bp.rolling(14).sum() / tr_u.rolling(14).sum().replace(0, np.nan)
    a28 = bp.rolling(28).sum() / tr_u.rolling(28).sum().replace(0, np.nan)
    ind["uo"] = safe(100 * (4 * a7 + 2 * a14 + a28) / 7, 50.0)

    # ── Volatility (annualised HV) ─────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    for hv_w, col_name in [(10, "volatility_10d"), (20, "volatility_20d"), (30, "volatility_30d")]:
        ind[col_name] = safe(log_ret.rolling(hv_w).std() * np.sqrt(252 * 78) * 100, 0.0)

    # ── Price changes ──────────────────────────────────────────────────────
    for n, col_name in [(1, "price_change_1d"), (2, "price_change_2d"),
                         (3, "price_change_3d"), (5, "price_change_5d")]:
        if len(c) > n:
            prev = float(c.iloc[-(n + 1)])
            ind[col_name] = ((close_v / prev) - 1) * 100 if prev else 0.0

    # ── Gap ────────────────────────────────────────────────────────────────
    if len(c) > 1:
        prev_bar = float(c.iloc[-2])
        if prev_bar:
            ind["gap_%"] = (float(o.iloc[0]) / prev_bar - 1) * 100

    # ── Aroon ──────────────────────────────────────────────────────────────
    if len(h) >= 26:
        hi_idx = h.rolling(26).apply(lambda x: float(np.argmax(x)), raw=True)
        lo_idx = l.rolling(26).apply(lambda x: float(np.argmin(x)), raw=True)
        ind["aroon_up"]        = safe(hi_idx / 25 * 100, 50.0)
        ind["aroon_down"]      = safe(lo_idx / 25 * 100, 50.0)
        ind["aroon_indicator"] = ind["aroon_up"] - ind["aroon_down"]

    # VWAP
    try:
        tp_vwap = (h + l + c) / 3
        cum_vol = v.cumsum().replace(0, np.nan)
        vwap_series = (tp_vwap * v).cumsum() / cum_vol
        ind["vwap"] = safe(vwap_series, close_v)
    except Exception:
        ind["vwap"] = close_v

    # ── HMA (Hull Moving Average) ─────────────────────────────────────────
    for n, col_name in [(9, "hma9"), (20, "hma20")]:
        try:
            half = n // 2
            sqrt_n = int(round(n ** 0.5))
            weights_half = np.arange(1, half + 1, dtype=float)
            weights_n    = np.arange(1, n + 1, dtype=float)
            weights_sqrt = np.arange(1, sqrt_n + 1, dtype=float)
            wma_half = c.rolling(half).apply(
                lambda x: np.dot(x, weights_half[-len(x):]) / weights_half[-len(x):].sum(), raw=True)
            wma_n = c.rolling(n).apply(
                lambda x: np.dot(x, weights_n[-len(x):]) / weights_n[-len(x):].sum(), raw=True)
            hma_raw = 2 * wma_half - wma_n
            hma = hma_raw.rolling(sqrt_n).apply(
                lambda x: np.dot(x, weights_sqrt[-len(x):]) / weights_sqrt[-len(x):].sum(), raw=True)
            ind[col_name] = safe(hma, float(c.mean()))
        except Exception:
            ind[col_name] = float(c.mean())

    # ── Price vs SMA50 ───────────────────────────────────────────────────
    sma50_v = ind.get("sma50") or float(c.mean())
    if sma50_v:
        ind["price_vs_sma50"] = (close_v / sma50_v - 1) * 100

    # ── Slope indicators (5-bar linear slope) ────────────────────────────
    def _slope(series, n=5):
        s = series.dropna()
        if len(s) < n:
            return 0.0
        y = s.iloc[-n:].values.astype(float)
        x = np.arange(n, dtype=float)
        try:
            return float(np.polyfit(x, y, 1)[0])
        except Exception:
            return 0.0

    sma20_series = c.rolling(20).mean()
    ema20_series = c.ewm(span=20, adjust=False).mean()
    rsi14_delta  = c.diff()
    rsi14_gain   = rsi14_delta.clip(lower=0)
    rsi14_loss   = (-rsi14_delta.clip(upper=0))
    rsi14_ag     = rsi14_gain.ewm(com=13, min_periods=14).mean()
    rsi14_al     = rsi14_loss.ewm(com=13, min_periods=14).mean()
    rsi14_series = 100 - (100 / (1 + rsi14_ag / rsi14_al.replace(0, np.nan)))

    ind["sma_20_slope"] = _slope(sma20_series)
    ind["ema_20_slope"] = _slope(ema20_series)
    ind["rsi_14_slope"] = _slope(rsi14_series)

    # ── Fast MACD (5/13/1) and MACD ROC ─────────────────────────────────
    ema5_f  = c.ewm(span=5,  adjust=False).mean()
    ema13_f = c.ewm(span=13, adjust=False).mean()
    macd_fast_line = ema5_f - ema13_f
    macd_fast_sig  = macd_fast_line.ewm(span=1, adjust=False).mean()
    ind["macd_fast"]  = safe(macd_fast_line)
    ind["macdh_fast"] = safe(macd_fast_line - macd_fast_sig)
    ind["macds_fast"] = safe(macd_fast_sig)
    # MACD ROC = rate of change of standard MACD line
    macd_std = ema12 - ema26  # already computed above
    ind["macd_roc"] = safe(macd_std.pct_change(3) * 100, 0.0)

    # ── Stochastic variants ───────────────────────────────────────────────
    stk_raw = (100 * (c - l.rolling(14).min()) /
               (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan))
    stk_smooth = stk_raw.rolling(3).mean()
    ind["stochh_14_3_3"] = safe(stk_smooth.rolling(3).max(), 50.0)

    # STOCH 5,3,1
    lo5  = l.rolling(5).min()
    hi5  = h.rolling(5).max()
    rng5 = (hi5 - lo5).replace(0, np.nan)
    stk5 = (100 * (c - lo5) / rng5).rolling(3).mean()
    std5 = stk5.rolling(1).mean()
    ind["stochk_5_3_1"] = safe(stk5, 50.0)
    ind["stochd_5_3_1"] = safe(std5, 50.0)
    ind["stochh_5_3_1"] = safe(stk5.rolling(1).max(), 50.0)

    # ── StochRSI ─────────────────────────────────────────────────────────
    rsi_s   = rsi14_series
    rsi_min = rsi_s.rolling(14).min()
    rsi_max = rsi_s.rolling(14).max()
    stoch_rsi = (rsi_s - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    stochrsi_k = stoch_rsi.rolling(3).mean() * 100
    stochrsi_d = stochrsi_k.rolling(3).mean()
    ind["stochrsik_14_14_3_3"] = safe(stochrsi_k, 50.0)
    ind["stochrsid_14_14_3_3"] = safe(stochrsi_d, 50.0)

    # ── CCI 14 ───────────────────────────────────────────────────────────
    tp14   = (h + l + c) / 3
    tp_ma14 = tp14.rolling(14).mean()
    tp_md14 = tp14.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    ind["cci"] = safe((tp14 - tp_ma14) / (0.015 * tp_md14.replace(0, np.nan)), 0.0)

    # ── OBV SMA20 ─────────────────────────────────────────────────────────
    obv_series = pd.Series(obv_vals, index=c.index)
    ind["obv_sma20"] = safe(obv_series.rolling(20).mean(), 0.0)

    # ── ADXR (ADX smoothed) ───────────────────────────────────────────────
    adx_series = dx.rolling(14).mean()
    ind["adxr"] = safe(adx_series.rolling(2).mean(), 20.0)

    # ── MFI 14 (Money Flow Index) ─────────────────────────────────────────
    tp_mfi = (h + l + c) / 3
    mf     = tp_mfi * v
    pos_mf = mf.where(tp_mfi > tp_mfi.shift(1), 0.0)
    neg_mf = mf.where(tp_mfi < tp_mfi.shift(1), 0.0)
    mfr    = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    ind["mfi"] = safe(100 - (100 / (1 + mfr)), 50.0)

    # ── ROC 20 and MOM 20 ─────────────────────────────────────────────────
    ind["roc20"] = safe(c.pct_change(20) * 100, 0.0)
    ind["mom20"] = safe(c.diff(20), 0.0)

    # ── Supertrend (10, 3) ────────────────────────────────────────────────
    try:
        atr10 = tr.rolling(10).mean()
        basic_upper = (h + l) / 2 + 3 * atr10
        basic_lower = (h + l) / 2 - 3 * atr10
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        supertrend  = pd.Series(np.nan, index=c.index)
        direction   = pd.Series(1, index=c.index)
        for i in range(1, len(c)):
            fu_prev = final_upper.iloc[i-1]
            fl_prev = final_lower.iloc[i-1]
            final_upper.iloc[i] = (
                basic_upper.iloc[i]
                if basic_upper.iloc[i] < fu_prev or c.iloc[i-1] > fu_prev
                else fu_prev
            )
            final_lower.iloc[i] = (
                basic_lower.iloc[i]
                if basic_lower.iloc[i] > fl_prev or c.iloc[i-1] < fl_prev
                else fl_prev
            )
            if pd.isna(supertrend.iloc[i-1]):
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i]  = -1
            elif supertrend.iloc[i-1] == fu_prev:
                if c.iloc[i] <= final_upper.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i]  = -1
                else:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i]  = 1
            else:
                if c.iloc[i] >= final_lower.iloc[i]:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i]  = 1
                else:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i]  = -1
        ind["supert"]   = safe(supertrend, close_v)
        ind["supert_d"] = safe(direction.astype(float), 1.0)
        ind["supert_l"] = safe(final_lower, close_v)
        ind["supert_s"] = safe(final_upper, close_v)
    except Exception:
        ind["supert"]   = close_v
        ind["supert_d"] = 1.0
        ind["supert_l"] = close_v
        ind["supert_s"] = close_v

    return ind


# ---------------------------------------------------------------------------
# FIX 9 (revised): Fetch real T-3/T-5/T-10 daily-bar snapshots
# ---------------------------------------------------------------------------

# Calendar days of lookback needed so that all MAs (up to SMA_50) have
# enough history even when the furthest offset is T-10.
_MULTIDAY_LOOKBACK_DAYS = 120

# Offset in calendar days for each flat prefix — must match the training
# pipeline in multiday_feature_collector.py exactly.
_MULTIDAY_TIMEFRAMES = {"t3": 3, "t5": 5, "t10": 10}


def _fetch_real_multiday_features(
    symbol: str,
    detection_date: "datetime",
    result: dict,
    logger: "logging.Logger",
) -> None:
    """
    Fetch daily OHLCV bars from yfinance and write genuine T-3, T-5, and T-10
    indicator snapshots into `result` under the t3_*/t5_*/t10_* keys.

    Uses the same _compute_indicators logic and PANDAS_TA_TO_BASE column map
    as multiday_feature_collector.py so that prediction-time features are
    identical in structure to the training-time features.
    """
    try:
        from src.multiday_feature_collector import (
            _compute_indicators as _daily_compute,
            _snapshot_for_offset,
            PANDAS_TA_TO_BASE,
        )
    except ImportError as exc:
        logger.warning(
            f"{symbol}: could not import multiday_feature_collector — "
            f"t3/t5/t10 features will be absent ({exc})"
        )
        return

    try:
        import yfinance as yf

        fetch_start = detection_date - timedelta(days=_MULTIDAY_LOOKBACK_DAYS)
        ticker = yf.Ticker(symbol)
        daily_df = ticker.history(
            start=fetch_start.strftime("%Y-%m-%d"),
            end=(detection_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )

        if daily_df is None or daily_df.empty:
            logger.debug(f"{symbol}: no daily bar data for multiday features")
            return

        # Strip timezone so index comparisons work uniformly
        daily_df.index = pd.to_datetime(daily_df.index).tz_localize(None)

        detection_ts = pd.Timestamp(detection_date).tz_localize(None)

        filled = 0
        for prefix, offset in _MULTIDAY_TIMEFRAMES.items():
            snap = _snapshot_for_offset(daily_df, detection_ts, offset)
            if not snap:
                logger.debug(f"{symbol}: empty snapshot for {prefix} (offset={offset})")
                continue
            for base_name, val in snap.items():
                if val is None:
                    continue
                col = f"{prefix}_{base_name}"
                result[col] = val
                filled += 1

        logger.debug(
            f"{symbol}: wrote {filled} real multiday features "
            f"(t3/t5/t10) from daily bars"
        )

    except Exception as exc:
        logger.warning(
            f"{symbol}: failed to fetch real multiday features — {exc}. "
            "t3/t5/t10 columns will be absent for this symbol."
        )


def fetch_t1_data_for_symbol(symbol: str, logger, fill_flat_prefixes: bool = False) -> dict:
    """
    Fetch T-1 intraday 5-min data and compute technical indicators.
    Returns dict with t1_close_* and t1_open_* keys ready for model input.

    FIX 7: t1_open_* features are now always populated.
    FIX 9 (revised): When fill_flat_prefixes=True (hybrid model), REAL T-3/T-5/T-10
           daily-bar snapshots are fetched via _fetch_real_multiday_features() and
           written under t3_/t5_/t10_* columns.  The old approach of copying T-1
           intraday indicators into all three prefix columns has been removed.
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

        # ── Close-of-day indicators (full session) ────────────────────────
        close_indicators = _compute_indicators(
            day_bars["close"], day_bars["high"], day_bars["low"],
            day_bars["volume"], day_bars["open"]
        )

        # ── Open indicators (first ~1 hour of session) ─────────────────────
        open_bars = day_bars[day_bars.index.time <= dt_time(10, 30)]

        open_indicators: dict = {}
        if len(open_bars) >= T1_OPEN_MIN_BARS:
            open_indicators = _compute_indicators(
                open_bars["close"], open_bars["high"], open_bars["low"],
                open_bars["volume"], open_bars["open"]
            )
            logger.debug(f"{symbol}: computed open indicators from {len(open_bars)} bars")
        else:
            logger.debug(
                f"{symbol}: only {len(open_bars)} open bars (< {T1_OPEN_MIN_BARS}), "
                "copying close indicators as open fallback"
            )
            open_indicators = dict(close_indicators)

        result = {}

        if _t1_map_available:
            # ── Rename and write t1_close_* ────────────────────────────────
            close_df      = pd.DataFrame([close_indicators])
            close_renamed = _rename(close_df, prefix="t1_close")
            for col in close_renamed.columns:
                val = close_renamed.iloc[0][col]
                if pd.notna(val):
                    try:
                        result[col] = float(val)
                    except (TypeError, ValueError):
                        pass

            # ── Rename and write t1_open_* ─────────────────────────────────
            open_df      = pd.DataFrame([open_indicators])
            open_renamed = _rename(open_df, prefix="t1_open")
            for col in open_renamed.columns:
                val = open_renamed.iloc[0][col]
                if pd.notna(val):
                    try:
                        result[col] = float(val)
                    except (TypeError, ValueError):
                        pass

        else:
            for k, val in close_indicators.items():
                result[f"t1_close_{k}"] = val
            for k, val in open_indicators.items():
                result[f"t1_open_{k}"] = val

        # ── FIX 9 (revised): Fetch REAL T-3/T-5/T-10 daily-bar snapshots ──
        # Uses genuine daily bars fetched from yfinance, processed through the
        # same _compute_indicators + PANDAS_TA_TO_BASE pipeline as training.
        # This replaces the old approach of copying T-1 intraday indicators
        # into all three prefix columns, which gave the model identical data
        # under three different names.
        if fill_flat_prefixes:
            # detection_date = the trading day whose T-1 bar we just computed,
            # i.e. yesterday (the most recent date in df_intraday).
            detection_date_for_multiday = datetime.combine(yesterday, dt_time(0, 0))
            _fetch_real_multiday_features(symbol, detection_date_for_multiday, result, logger)

        t1_close_count = sum(1 for k in result if k.startswith("t1_close_"))
        t1_open_count  = sum(1 for k in result if k.startswith("t1_open_"))
        flat_count     = sum(1 for k in result if k.startswith("t3_"))
        logger.debug(
            f"{symbol}: t1_close_* = {t1_close_count}, "
            f"t1_open_* = {t1_open_count}, "
            f"t3_* = {flat_count} (from real daily bars)"
        )

        return result

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Gain estimation helpers
# ---------------------------------------------------------------------------

def _prob_to_gain(probability: float) -> float:
    """
    Convert a probability score to a base gain estimate using _GAIN_CURVE.
    Used as the final fallback when both the regressor and isotonic calibrator
    are unavailable.  The curve is anchored to real explosive-mover intraday
    highs, so high-confidence predictions can reach 100 %+.
    """
    for min_prob, gain in _GAIN_CURVE:
        if probability >= min_prob:
            return gain
    return _GAIN_CURVE[-1][1]


def _apply_gain_rank_correction(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_prefix: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Emergency fallback applied only when the predictor returns near-identical
    gain estimates (std < 1 %).  Instead of a fixed floor-to-ceiling formula,
    the correction is anchored to each stock's probability score via
    _GAIN_CURVE, with a small rank-based spread added on top.  This means:

    - A 0.95-probability STRONG BUY stock can receive a 100 %+ estimate.
    - A 0.55-probability HOLD stock gets ~10 %, not an artificially inflated
      number just because it happens to rank highly in a mediocre pool.
    - The `gain_source` column is set to "rank_fallback" on every overwritten
      row so callers and the Supabase table can distinguish model estimates
      from fallback values.
    """
    if 'target_gain_pct' not in predictions_df.columns:
        return predictions_df

    gains    = predictions_df['target_gain_pct']
    gain_std = gains.std()

    if gain_std >= 1.0:
        # Model gains look healthy — just ensure gain_source is stamped
        if 'gain_source' not in predictions_df.columns:
            predictions_df = predictions_df.copy()
            predictions_df['gain_source'] = predictions_df.get(
                'gain_source', pd.Series('model', index=predictions_df.index)
            )
        return predictions_df

    logger.warning(
        f"  ⚠️  FLAT GAIN ESTIMATES detected (std={gain_std:.4f}%). "
        f"Applying probability-anchored rank fallback correction."
    )

    df    = predictions_df.copy()
    probs = df['explosion_probability']

    # Base gain from the probability curve (tied to model confidence, not rank)
    base_gains = probs.apply(_prob_to_gain)

    # Small rank-based spread on top: ±20 % of the base gain, so stocks with
    # the same bucket probability are still differentiated.  This keeps the
    # spread proportional — a 100 % base gets ±20 %, a 10 % base gets ±2 %.
    prob_ranks   = probs.rank(pct=True)           # 0..1, higher = better
    rank_spread  = (prob_ranks - 0.5) * 0.40      # -0.20 .. +0.20 multiplier
    corrected_gains = base_gains * (1.0 + rank_spread)

    # RSI adjustment: stocks near RSI 60 (momentum without being overbought)
    # get a small boost; extremely overbought (RSI > 80) get a haircut
    if _uses_lowercase(feature_prefix):
        rsi_col = f"{feature_prefix}_rsi_14"
        vol_col = f"{feature_prefix}_volume_ratio"
    else:
        rsi_col = f"{feature_prefix}_RSI_14"
        vol_col = f"{feature_prefix}_Volume_Ratio"

    if features_df is not None and not features_df.empty:
        feat_indexed = features_df.set_index("symbol") if "symbol" in features_df.columns else features_df

        if rsi_col in feat_indexed.columns:
            rsi_vals = df['symbol'].map(feat_indexed[rsi_col]) if 'symbol' in df.columns else None
            if rsi_vals is not None and rsi_vals.notna().sum() > 5:
                # +5 % boost near RSI 60, haircut when RSI > 80
                rsi_filled   = rsi_vals.fillna(55)
                rsi_boost    = (1.0 - (abs(rsi_filled - 60) / 40).clip(0, 1)) * 0.05
                rsi_overbought_cut = ((rsi_filled - 80).clip(0, 20) / 20) * -0.10
                corrected_gains = corrected_gains * (1 + rsi_boost + rsi_overbought_cut)

        if vol_col in feat_indexed.columns:
            vol_vals = df['symbol'].map(feat_indexed[vol_col]) if 'symbol' in df.columns else None
            if vol_vals is not None and vol_vals.notna().sum() > 5:
                # Up to +10 % boost for relative volume 5×; proportional below that
                vol_score = (vol_vals.fillna(1.0) - 1.0).clip(0, 4) / 4.0
                corrected_gains = corrected_gains * (1 + vol_score * 0.10)

    # No hard ceiling — explosive movers genuinely go 100 %+.
    # Floor at 3 % to avoid negative or zero estimates on low-probability stocks.
    corrected_gains = corrected_gains.clip(lower=3.0)

    df['target_gain_pct']  = corrected_gains
    df['target_gain_low']  = corrected_gains * 0.50   # conservative scenario
    df['target_gain_high'] = corrected_gains * 1.60   # explosive scenario
    df['gain_source']      = 'rank_fallback'

    if 'current_price' in df.columns:
        df['target_price']      = df['current_price'] * (1 + df['target_gain_pct']  / 100)
        df['target_price_low']  = df['current_price'] * (1 + df['target_gain_low']  / 100)
        df['target_price_high'] = df['current_price'] * (1 + df['target_gain_high'] / 100)

    n_fallback = len(df)
    logger.warning(
        f"  ⚠️  gain_source='rank_fallback' applied to all {n_fallback} rows. "
        f"Gain estimates are NOT from the model — they are probability-anchored "
        f"approximations. Retrain the gain regressor to restore model-based estimates."
    )
    logger.info(
        f"  Corrected gain range: "
        f"{corrected_gains.min():.1f}%–{corrected_gains.max():.1f}%  "
        f"std={corrected_gains.std():.1f}%"
    )

    # Per-signal breakdown so the log makes the distribution visible
    if 'signal' in df.columns:
        for sig in ['STRONG BUY', 'BUY', 'HOLD', 'AVOID']:
            mask = df['signal'] == sig
            if mask.any():
                g = corrected_gains[mask]
                logger.info(
                    f"    {sig:<12}: n={mask.sum():>4}  "
                    f"gain {g.min():.1f}%–{g.max():.1f}%  "
                    f"(mean {g.mean():.1f}%)"
                )
    return df


def _get_calibrated_gain_estimate(probability: float) -> float:
    """
    Rule-based gain backstop for individual NaN rows after the main gain
    pipeline has run.  Uses the same _GAIN_CURVE as _prob_to_gain so both
    paths are consistent.  Replaces the old hard-coded table that topped out
    at 30 % regardless of model confidence.
    """
    return _prob_to_gain(probability)


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
        help="Skip T-1 intraday fetch (fastest — uses only TV screener data)",
    )
    parser.add_argument(
        "--t1-limit",
        type=int,
        default=500,
        help=(
            "Maximum number of screened stocks to enrich with T-1 yfinance intraday "
            "data (default: 500). Stocks are taken in screener sort order (descending "
            "relative volume), so stocks ranked beyond this limit are scored on TV "
            "screener features only — a known lower-fidelity signal. Set to 0 to "
            "enrich ALL screened stocks (slow: ~0.15 s per stock)."
        ),
    )
    parser.add_argument(
        "--t1-workers",
        type=int,
        default=5,
        help="Number of parallel threads for T-1 yfinance fetches (default: 5).",
    )

    args   = parser.parse_args()
    logger = setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("ML SCREENING & PREDICTION")
    logger.info("=" * 80)

    screener = SmartScreener(logger=logger)

    try:
        predictor = ExplosionPredictor()
        supabase  = MLPredictionSupabaseClient({})
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1

    # FIX 10: derive prediction date from daily_winners rather than wall clock
    prediction_date = get_next_trading_day(supabase)

    logger.info(f"Prediction date (trading session): {prediction_date}")
    logger.info(f"Top N to store: {args.top_n}")
    logger.info("=" * 80)

    # FIX 9: Detect hybrid model — determines whether real T-3/T-5/T-10
    # daily-bar snapshots should be fetched for t3_/t5_/t10_* columns.
    model_prefix       = predictor.model_feature_prefix
    hybrid             = _is_hybrid_model(predictor)
    fill_flat_prefixes = hybrid

    logger.info(f"✓ Model feature prefix detected: '{model_prefix}'")
    logger.info(f"  Model is hybrid (t1+t3/t5/t10): {hybrid}")
    if hybrid:
        logger.info(
            "  → TV screener features will be mapped to BOTH t1_close_* AND t3_/t5_/t10_* columns\n"
            "  → T-1 yfinance indicators will ALSO be written to t3_/t5_/t10_* (FIX 9)"
        )
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
            features = build_features_from_tv_data(
                row_dict, symbol,
                feature_prefix=model_prefix,
            )
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

    if hybrid and enriched_stocks:
        sample = enriched_stocks[0]
        t3_hits = sum(1 for k in sample if k.startswith("t3_"))
        t1_hits = sum(1 for k in sample if k.startswith("t1_close_"))
        logger.info(f"  Sample stock '{sample['symbol']}': t3_ cols={t3_hits}, t1_close_ cols={t1_hits}")

    if not enriched_stocks:
        logger.error("Failed to build features for any stocks")
        return 1

    # ── STEP 3: T-1 INTRADAY INDICATOR ENRICHMENT ────────────────────────────
    if not args.no_t1:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: T-1 INTRADAY INDICATOR ENRICHMENT")
        logger.info("=" * 80)

        # Resolve how many stocks to enrich.  --t1-limit 0 means "all of them".
        t1_limit = args.t1_limit if args.t1_limit > 0 else len(enriched_stocks)
        t1_candidates = [s["symbol"] for s in enriched_stocks[:t1_limit]]

        logger.info(
            f"Computing T-1 indicators from 5-min bars for "
            f"{len(t1_candidates)} / {len(enriched_stocks)} screened stocks "
            f"(--t1-limit={t1_limit}, --t1-workers={args.t1_workers})."
        )

        # ── Known-limitation notice ──────────────────────────────────────────
        # Stocks are taken in screener sort order (descending relative volume).
        # Stocks ranked beyond --t1-limit are scored on TV screener features
        # only, which are lower fidelity than T-1 yfinance intraday indicators.
        # To eliminate this bias entirely, pass --t1-limit 0, at the cost of
        # a longer runtime (~0.15 s per additional stock).
        if t1_limit < len(enriched_stocks):
            skipped = len(enriched_stocks) - t1_limit
            logger.warning(
                f"  ⚠️  {skipped} stocks beyond rank {t1_limit} will be scored on "
                f"TV screener features only (lower fidelity). "
                f"Pass --t1-limit 0 to enrich all stocks."
            )

        if hybrid:
            logger.info(
                "  Hybrid model: indicators will ALSO be written as t3_/t5_/t10_* "
                "to cover features like t3_ema_12, t3_rsi_7, t3_wma_10 (FIX 9)."
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time, random

        t1_map   = {}
        t1_count = 0

        def fetch_with_jitter(sym):
            time.sleep(random.uniform(0.05, 0.2))
            return sym, fetch_t1_data_for_symbol(sym, logger, fill_flat_prefixes=fill_flat_prefixes)

        with ThreadPoolExecutor(max_workers=args.t1_workers) as executor:
            futures = {executor.submit(fetch_with_jitter, sym): sym for sym in t1_candidates}
            for i, future in enumerate(as_completed(futures), 1):
                if i % 50 == 0:
                    logger.info(
                        f"  T-1 progress: {i}/{len(t1_candidates)} | enriched: {t1_count}"
                    )
                sym, t1_data = future.result()
                if t1_data:
                    t1_map[sym] = t1_data
                    t1_count   += 1

        for stock in enriched_stocks:
            sym = stock["symbol"]
            if sym in t1_map:
                stock.update(t1_map[sym])

        # Coverage report
        sample = next((s for s in enriched_stocks if s["symbol"] in t1_map), None)
        if sample:
            t1c = sum(1 for k in sample if k.startswith("t1_close_"))
            t1o = sum(1 for k in sample if k.startswith("t1_open_"))
            t3c = sum(1 for k in sample if k.startswith("t3_"))
            logger.info(
                f"  Sample ({sample['symbol']}): "
                f"t1_close_* = {t1c} features, "
                f"t1_open_* = {t1o} features, "
                f"t3_* = {t3c} features"
            )
            if t1o == 0:
                logger.warning(
                    "  ⚠️  t1_open_* still 0 — check that t1_column_map.py is present "
                    "alongside this script."
                )
            if hybrid and t3c == 0:
                logger.warning(
                    "  ⚠️  t3_* still 0 after T-1 enrichment — check that "
                    "t1_column_map.py exposes INTRADAY_TO_MODEL."
                )

        logger.info(
            f"✓ T-1 indicator enrichment: {t1_count}/{len(t1_candidates)} stocks enriched "
            f"({len(enriched_stocks) - len(t1_candidates)} stocks scored on TV screener features only)"
        )
    else:
        logger.info("\nSTEP 3: Skipped (--no-t1 flag)")

    # ── STEP 4: PREPARE FEATURES + PRE-FLIGHT VARIANCE CHECK ─────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: PREPARE FEATURES + PRE-FLIGHT VARIANCE CHECK")
    logger.info("=" * 80)

    features_df  = pd.DataFrame(enriched_stocks)
    model_cols   = [c for c in features_df.columns if c.startswith(f"{model_prefix}_")]
    t3_cols      = [c for c in features_df.columns if c.startswith("t3_")]
    t1_open_cols = [c for c in features_df.columns if c.startswith("t1_open_")]

    logger.info(f"✓ Feature matrix: {len(features_df)} stocks × {len(features_df.columns)} raw columns")
    logger.info(f"  {model_prefix}_ features present: {len(model_cols)}")
    logger.info(f"  t3_ features present:       {len(t3_cols)}")
    logger.info(f"  t1_open_ features present:  {len(t1_open_cols)}")

    key_indicators = [
        "t1_close_RSI_14", "t1_close_ATR_14", "t1_close_ADX_14",
        "t1_close_TSI_13_25_13", "t1_close_KCUe_20_2", "t1_close_VWAP",
        "t1_open_RSI_14",  "t1_open_ATR_14",  "t1_open_ADX_14",
        "t3_rsi_14",       "t3_atr_14",       "t3_adx_14",
        "t3_ema_12",       "t3_rsi_7",        "t3_wma_10",
        "t3_volume_ratio", "t1_close_Volume_Ratio",
    ]
    zero_var_indicators = []
    for col in key_indicators:
        if col in features_df.columns:
            col_data = pd.to_numeric(features_df[col], errors='coerce').dropna()
            std_val  = col_data.std() if len(col_data) > 1 else 0.0
            logger.info(f"  {col}: n={len(col_data)}, mean={col_data.mean():.2f}, std={std_val:.4f}")
            if std_val < 1e-4 and len(col_data) > 5:
                zero_var_indicators.append(col)

    if zero_var_indicators:
        logger.warning(
            f"\n  ⚠️  ZERO-VARIANCE COLUMNS: {zero_var_indicators}"
            "\n      Model will output near-identical probabilities for these features."
        )
    elif len(features_df) < 20:
        logger.warning(
            f"\n  ⚠️  Only {len(features_df)} stocks — too few for a meaningful distribution."
        )
    else:
        logger.info(f"  ✅ Feature variance looks healthy ({len(features_df)} stocks)")

    t3_nonzero = sum(
        1 for c in features_df.columns
        if c.startswith("t3_") and features_df[c].std() > 0.01
    )
    logger.info(f"t3_ columns with real variance: {t3_nonzero}")

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

    prob_std = predictions_df['explosion_probability'].std()
    if prob_std < 0.001 and len(predictions_df) > 5:
        logger.error(
            f"\n  ❌ POST-PREDICTION: prob_std={prob_std:.6f} — all probabilities identical."
            "\n     The model received uniform feature inputs."
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
            # Tag individual-fallback rows so they're distinguishable
            if 'gain_source' not in predictions_df.columns:
                predictions_df['gain_source'] = 'model'
            predictions_df.loc[bad_gain_mask, 'gain_source'] = 'individual_fallback'

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
    logger.info(f"{'#':<4} {'Symbol':<8} {'Signal':<13} {'Prob':<8} {'Price':<10} {'Target':<10} {'Gain':<8} {'t3_cols'}")
    logger.info("-" * 100)

    for rank, (_, row) in enumerate(top_predictions.head(20).iterrows(), 1):
        current_price = row.get('current_price', 0)
        n_t3 = sum(1 for k in row.index if k.startswith("t3_") and pd.notna(row.get(k)))
        logger.info(
            f"{rank:<4} {row['symbol']:<8} {row['signal']:<13} "
            f"{row['explosion_probability']*100:>6.2f}%  "
            f"${current_price:>8.2f}  "
            f"${row.get('target_price', 0):>8.2f}  "
            f"+{row.get('target_gain_pct', 0):>5.1f}%"
            f"  t3={n_t3}"
        )

    # ── STEP 8: STORE PREDICTIONS ─────────────────────────────────────────────
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
            'gain_source':           row.get('gain_source', 'model'),
            'model_version':         f"{model_prefix}_v9",
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
        'model_version':      f"{model_prefix}_features_v9",
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
    logger.info(f"  Hybrid model:       {hybrid}")
    logger.info(f"  Prob std:           {predictions_df['explosion_probability'].std():.4f}")
    logger.info(f"  Gain std:           {predictions_df['target_gain_pct'].std():.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
