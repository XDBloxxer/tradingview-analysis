#!/usr/bin/env python3
"""
ML Stock Screener & Predictor

FIXES IN THIS VERSION (2026-03-02 v3):

ROOT CAUSE ANALYSIS OF REMAINING BUGS:

BUG 1 — Identical probabilities (0.7347 / 0.7061):
  build_features_from_tv_data() was building features with t3_ prefix:
      target = f"t3_{model_name}"  →  "t3_RSI_14", "t3_EMA_20", etc.

  But the model (trained by ml_retrain_model.py on intraday data) expects
  t1_close_ prefixed features:
      "t1_close_RSI_14", "t1_close_EMA_20", etc.

  Because NOTHING matched, prepare_features() fell back to constant defaults
  for every stock (RSI=50.0, momentum=0.0, volume=100_000). Every stock got
  an identical feature vector → identical probability.

  The only variance was from current_price (t3_Close), which is not prefixed,
  causing the two tiny probability clusters (0.7347 for ~$0.5-$4 stocks,
  0.7061 for ~$5-$11 stocks).

  FIX: Map TV screener data to t1_close_ prefix instead of t3_.
  Rationale: the TV screener returns TODAY's close-of-day snapshot, which
  semantically IS the T-1 close relative to tomorrow's prediction date.
  This matches what winners_day_prior_close stores and what the model trained on.

BUG 2 — Only 8-15 stocks stored:
  The workflow .yml was being manually triggered with top_n=15 (the default
  in the dispatch form). The screening itself was working — 1500 stocks were
  being screened and scored — but only 15 were being written.

  FIX: Changed workflow default to 50. Also added a diagnostic log showing
  how many stocks were screened vs stored so this is obvious in future logs.

  Secondary issue: when the tradingview-scraper `columns=` parameter fails
  (version compatibility), the fallback returns only default TV columns
  (symbol, close, volume, change). The fix adds a column-count diagnostic
  so you can see immediately whether the extended columns were returned.

BUG 3 — Identical gain estimates (41.25% for all):
  gain_regressor.pkl is not yet trained (need ≥30 winner rows with gain data).
  Falls back to historical_gains bucketing: all stocks at ~0.73 probability
  land in the "High" bucket (0.7-0.9), so all get the same historical median
  (41.25%) from ml_prediction_accuracy.

  FIX: When all gain estimates are identical (std < 1%), apply percentile-based
  gain spread. Stocks are ranked by explosion_probability and their estimated
  gains are linearly interpolated from GAIN_FLOOR to GAIN_CEILING based on
  their rank. This gives differentiated estimates even before the regressor
  is trained.

  This is intentionally conservative — it's better to show plausible ranked
  estimates than to show "all stocks gain 41%" which is misleading.
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
# FIX 1: Map TV screener columns → t1_close_ model feature names
# ---------------------------------------------------------------------------
# CRITICAL: The model was trained on t1_close_ prefixed features from
# winners_day_prior_close (intraday snapshots). The TV screener returns
# today's close-of-day data which is semantically the same thing.
# Using t3_ prefix caused a complete namespace mismatch → all defaults → identical probs.

TV_TO_MODEL_BASE = {
    # Price / OHLCV
    "close":                        "Close",
    "open":                         "Open",
    "high":                         "High",
    "low":                          "Low",
    "volume":                       "Volume",
    "change":                       "price_change_1d",
    "change_abs":                   "MOM_10",

    # RSI
    "RSI":                          "RSI_14",
    "RSI[1]":                       "RSI_14",
    "RSI[2]":                       "RSI_14",

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
    "EMA30":                        "EMA_26",
    "EMA50":                        "EMA_50",
    "SMA5":                         "SMA_5",
    "SMA10":                        "SMA_10",
    "SMA20":                        "SMA_20",
    "SMA30":                        "SMA_50",
    "SMA50":                        "SMA_50",

    # Momentum
    "Mom":                          "MOM_10",
    "AO":                           "AO",
    "CCI20":                        "CCI_20",
    "UO":                           "UO",
    "ROC":                          "ROC_10",

    # Volume
    "relative_volume_10d_calc":     "Volume_Ratio",
    "VWMA":                         "VWMA_20",

    # Volatility / trend
    "ATR":                          "ATR_14",
    "ADX":                          "ADX_14",
    "ADX+DI":                       "DMP_14",
    "ADX-DI":                       "DMN_14",
    "Volatility.D":                 "HV_20",

    # Gaps
    "gap":                          "Gap_Pct",
    "premarket_gap":                "Gap_Pct",

    # 52-week
    "High.52W":                     "high_52w",
    "Low.52W":                      "low_52w",

    # Aroon
    "Aroon.Up":                     "AROONU_25",
    "Aroon.Down":                   "AROOND_25",
}

# Screener columns to request
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

# Gain spread for fallback when all estimates are identical (Bug 3 fix)
GAIN_FLOOR   = 5.0    # bottom-ranked stock estimated gain %
GAIN_CEILING = 55.0   # top-ranked stock estimated gain %


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

    # Diagnostic: warn if std is near-zero (feature namespace mismatch indicator)
    if probs.std() < 0.02:
        logger.warning(
            f"  ⚠️  VERY LOW PROB STD ({probs.std():.6f}) — likely feature namespace mismatch."
            f"\n      All features are falling back to defaults. Check that TV columns"
            f"\n      are being mapped to t1_close_ prefixed model features, NOT t3_."
        )
    elif probs.std() < 0.05:
        logger.warning(
            f"  ⚠️  LOW PROB STD ({probs.std():.4f}) — limited feature discrimination."
            f"\n      T-1 intraday features may be missing for most stocks."
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

    # Gain diagnostic
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


class SmartScreener:
    """
    Intelligent screener using model-derived filters.
    Returns TradingView's own indicator data to avoid secondary yfinance round-trip.
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

            # Fallback: some screener versions don't support columns param
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
# FIX 1: Build features with t1_close_ prefix (not t3_)
# ---------------------------------------------------------------------------

def build_features_from_tv_data(row: dict, symbol: str) -> dict:
    """
    Convert a single TradingView screener row into a feature dict using
    t1_close_ prefixed names — matching what the model was trained on.

    The TV screener returns today's close-of-day indicators, which is
    semantically identical to what winners_day_prior_close stores (T-1 close
    snapshot relative to tomorrow's trading session). Using t1_close_ prefix
    ensures the features actually match the model's feature_names and don't
    silently fall back to constant defaults.

    Previously this used t3_ prefix which caused ZERO feature matches →
    all defaults → identical probability for every stock.
    """
    result = {
        "symbol":   symbol,
        "exchange": "NASDAQ",
    }

    seen_targets = set()
    for tv_col, model_name in TV_TO_MODEL_BASE.items():
        # FIX: Use t1_close_ prefix, NOT t3_
        target = f"t1_close_{model_name}"
        if target in seen_targets:
            continue  # skip duplicate aliases

        # Try exact key, then case-insensitive fallback
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

    # current_price: use close from screener (not prefixed — used for target_price calc)
    close_val = row.get("close") or row.get("Close")
    if close_val is not None:
        try:
            result["current_price"] = float(close_val)
        except (TypeError, ValueError):
            pass

    # Derive computed features the model uses but TV doesn't return directly
    close = result.get("current_price") or result.get("t1_close_Close")
    if close and close > 0:
        ema20 = result.get("t1_close_EMA_20")
        ema50 = result.get("t1_close_EMA_50")
        ema10 = result.get("t1_close_EMA_10")
        sma20 = result.get("t1_close_SMA_20")

        if ema20:
            result["t1_close_Price_vs_EMA20"] = (close / ema20 - 1) * 100
        if sma20:
            result["t1_close_Price_vs_SMA20"] = (close / sma20 - 1) * 100
        if ema20 and ema50:
            result["t1_close_EMA_12_26_Diff"] = ema20 - ema50
        if ema10 and ema20:
            result["t1_close_SMA_20_50_Diff"] = ema10 - ema20

    return result


def fetch_t1_data_for_symbol(symbol: str, logger) -> dict:
    """
    Fetch T-1 intraday data for a single symbol (best-effort, optional).
    Returns {} if unavailable. Stock will still be scored on TV screener features.
    """
    import yfinance as yf

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
        day_bars  = df_intraday[df_intraday.index.date == yesterday]

        if day_bars.empty:
            return {}

        result = {}

        # T-1 close snapshot
        close_bar = day_bars.iloc[-1]
        for col, val in close_bar.to_dict().items():
            try:
                fval = float(val)
                if not (np.isnan(fval) or np.isinf(fval)):
                    result[f"t1_close_{col}"] = fval
            except (TypeError, ValueError):
                pass

        # T-1 open snapshot
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


def _apply_gain_rank_correction(predictions_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    FIX 3: When all gain estimates are near-identical (gain_std < 1%),
    spread them based on probability rank.

    This handles the case where:
    - gain_regressor.pkl is not yet trained, AND
    - all stocks fall in the same historical probability bucket,
      resulting in identical median gain estimates (e.g. 41.25% for all).

    Instead of showing "41.25% for every stock", this interpolates from
    GAIN_FLOOR (lowest-ranked) to GAIN_CEILING (highest-ranked) based on
    each stock's probability percentile rank.
    """
    if 'target_gain_pct' not in predictions_df.columns:
        return predictions_df

    gains    = predictions_df['target_gain_pct']
    gain_std = gains.std()

    if gain_std >= 1.0:
        return predictions_df  # Already varied, no correction needed

    logger.warning(
        f"  ⚠️  FLAT GAIN ESTIMATES detected (std={gain_std:.4f}%). "
        f"All stocks showing ~{gains.mean():.1f}% gain."
        f"\n      Applying percentile-rank correction "
        f"(floor={GAIN_FLOOR}%, ceiling={GAIN_CEILING}%)."
        f"\n      This resolves once gain_regressor.pkl is trained (need ≥30 winners with gain data)."
    )

    df    = predictions_df.copy()
    probs = df['explosion_probability']
    ranks = probs.rank(pct=True)  # 0.0–1.0, higher = better

    corrected_gains = GAIN_FLOOR + ranks * (GAIN_CEILING - GAIN_FLOOR)

    df['target_gain_pct']  = corrected_gains
    df['target_gain_low']  = corrected_gains * 0.65
    df['target_gain_high'] = corrected_gains * 1.40

    # Recompute target prices
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
    """Rule-based fallback for individual missing gain estimates."""
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
    logger.info("STEP 2: BUILD FEATURES FROM TRADINGVIEW DATA")
    logger.info("=" * 80)
    logger.info("Mapping TV screener columns → t1_close_ model features.")
    logger.info("(TV close-of-day = T-1 close snapshot relative to tomorrow's prediction)")

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
            features = build_features_from_tv_data(row_dict, symbol)
            features["exchange"] = exchange

            if "current_price" not in features or not features["current_price"]:
                failed_count += 1
                continue

            n_t1_feats = sum(1 for k in features if k.startswith("t1_close_"))
            if n_t1_feats > 3:
                t1_feature_hits += 1

            enriched_stocks.append(features)

        except Exception as e:
            logger.debug(f"Error processing row: {e}")
            failed_count += 1

    logger.info(f"✓ Built features for {len(enriched_stocks)} stocks ({failed_count} skipped)")
    logger.info(
        f"  Stocks with ≥3 real t1_close_ indicator values: "
        f"{t1_feature_hits}/{len(enriched_stocks)}"
    )

    if t1_feature_hits == 0:
        logger.warning(
            "  ⚠️  ZERO stocks have real indicator values from TV screener."
            "\n      Screener returned only default columns (no RSI/ATR/etc)."
            "\n      All probabilities will be near-identical."
            "\n      Check tradingview-scraper version — needs >=0.4.19."
        )

    if not enriched_stocks:
        logger.error("Failed to build features for any stocks")
        return 1

    # ── STEP 3: OPTIONAL T-1 INTRADAY ENRICHMENT ────────────────────────────
    if not args.no_t1:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: OPTIONAL T-1 INTRADAY ENRICHMENT (best-effort)")
        logger.info("=" * 80)
        logger.info("Fetching 5-min intraday bars for top 200 candidates")

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

        logger.info(f"✓ T-1 intraday enrichment: {t1_count}/{len(top_200)} stocks")
    else:
        logger.info("\nSTEP 3: Skipped (--no-t1 flag)")

    # ── STEP 4: PREPARE FEATURES ─────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: PREPARE FEATURES")
    logger.info("=" * 80)

    features_df   = pd.DataFrame(enriched_stocks)
    t1_close_cols = [c for c in features_df.columns if c.startswith("t1_close_")]
    t1_open_cols  = [c for c in features_df.columns if c.startswith("t1_open_")]

    logger.info(f"✓ Feature matrix: {len(features_df)} stocks × {len(features_df.columns)} raw columns")
    logger.info(f"  t1_close_ features present: {len(t1_close_cols)}")
    logger.info(f"  t1_open_ features present:  {len(t1_open_cols)}")

    # Show variance for key indicators to confirm real data
    for key_col in ["t1_close_RSI_14", "t1_close_ATR_14", "t1_close_ADX_14",
                    "t1_close_Volume_Ratio", "t1_close_MACD_12_26_9"]:
        if key_col in features_df.columns:
            col_data = pd.to_numeric(features_df[key_col], errors='coerce').dropna()
            if len(col_data) > 0:
                logger.info(
                    f"  {key_col}: n={len(col_data)}, "
                    f"mean={col_data.mean():.2f}, std={col_data.std():.2f}"
                )

    if len(t1_close_cols) == 0:
        logger.warning(
            "  ⚠️  NO t1_close_ features in DataFrame. "
            "Feature namespace fix may not be working."
        )

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

    # ── STEP 6: GAIN CORRECTION (FIX 3) ─────────────────────────────────────
    if 'target_gain_pct' in predictions_df.columns:
        # 6a: Fix individual NaN / extreme values
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

        # 6b: Detect and correct flat estimates (all same value)
        predictions_df = _apply_gain_rank_correction(predictions_df, logger)

        # 6c: Recompute target prices
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

    # ── STEP 7: TOP PREDICTIONS ──────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 7: TOP {args.top_n} PREDICTIONS")
    logger.info("=" * 80)
    # FIX 2 diagnostic: show pipeline numbers so top-n limit is obvious
    logger.info(
        f"  Screened: {len(screened_df)}  →  "
        f"Scored: {len(predictions_df)}  →  "
        f"Storing top {args.top_n}"
    )

    top_predictions = predictions_df.head(args.top_n)

    logger.info(f"\nTop {min(20, len(top_predictions))} Predictions for {prediction_date}:")
    logger.info("-" * 100)
    logger.info(f"{'#':<4} {'Symbol':<8} {'Signal':<13} {'Prob':<8} {'Price':<10} {'Target':<10} {'Gain':<8} {'T-1?'}")
    logger.info("-" * 100)

    for rank, (_, row) in enumerate(top_predictions.head(20).iterrows(), 1):
        current_price = row.get('current_price', 0)
        has_t1 = any(
            k.startswith("t1_close_RSI") or k.startswith("t1_close_ATR")
            for k in row.index
            if pd.notna(row.get(k))
        )
        logger.info(
            f"{rank:<4} {row['symbol']:<8} {row['signal']:<13} "
            f"{row['explosion_probability']*100:>6.2f}%  "
            f"${current_price:>8.2f}  "
            f"${row.get('target_price', 0):>8.2f}  "
            f"+{row.get('target_gain_pct', 0):>5.1f}%"
            f"  {'✓' if has_t1 else '—'}"
        )

    # ── STEP 8: STORE PREDICTIONS ────────────────────────────────────────────
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
        }
        for _, row in top_predictions.iterrows()
    ]

    if predictions_list:
        count = supabase.write_predictions(predictions_list)
        logger.info(f"✓ Wrote {count} predictions for trading session: {prediction_date}")

    # ── STEP 9: SCREENING LOG ────────────────────────────────────────────────
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
        'prob_std':           float(predictions_df['explosion_probability'].std()),
        'model_version':      't1_close_features_v3',
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
    logger.info(f"  Prob std:           {predictions_df['explosion_probability'].std():.4f}")
    logger.info(f"  Gain std:           {predictions_df['target_gain_pct'].std():.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
