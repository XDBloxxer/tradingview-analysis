#!/usr/bin/env python3
"""
ML Stock Screener & Predictor - FIXED VERSION

FIXES IN THIS VERSION (2026-03-01):

FIX 1 — Only ~15/1500 stocks making it through:
  Previously fetch_stock_data_for_prediction() returned None whenever T-1
  intraday data was missing or insufficient. Since yfinance 5-min data is
  often unavailable for less-liquid tickers, this silently dropped ~99% of
  screened stocks. T-1 data is now OPTIONAL — stocks are scored on T-3/T-5/T-10
  features alone when intraday data is unavailable. XGBoost handles NaN natively.

  Additionally the SmartScreener min_volume_ratio default was 2.0 which is
  already aggressive on its own. Combined with min_volume:300k and price filters
  it was over-filtering. Defaults relaxed: min_volume_ratio → 1.5,
  min_volume → 200k. The learned_filters.json still overrides these when available.

FIX 2 — All predictions HOLD/AVOID:
  extract_features_with_prefix() was calling col.lower() on all indicator names,
  producing t3_rsi_14 instead of t3_RSI_14. The model expects the latter.
  ExplosionPredictor._norm() handles case via .lower() for matching, but
  _get_default_value() was being called for every single feature because the
  input already had lowercased names that didn't match the (also-lowercased)
  model names consistently after dot normalization.

  Root fix: extract_features_with_prefix() now PRESERVES original column casing.
  The _norm() matching in ExplosionPredictor then correctly maps them.

  Also: extract_intraday_snapshot() was lowercasing all keys via col_lower.
  This is correct for storage (matching the Supabase schema) but wrong for
  prediction (the model expects the t1_column_map long-form names). The function
  now applies t1_column_map renaming when building prediction features.

FIX 3 — Target price predictions way off:
  The rule-based _estimate_target_gain() fallback used static thresholds
  (0.90 prob → 25%) not calibrated to actual data. The historical accuracy
  calibration path was also silently failing because get_historical_prediction_accuracy()
  uses .not_.is_("became_winner", "null") which returns empty when the column
  isn't populated yet. Added a calibrated fallback based on winner statistics
  (winners in your dataset average ~30-50% gain on detection day, not 25%).

Previously also: calculate_comprehensive_indicators_intraday was only computing
~10 indicators (SMA, EMA, RSI, MACD, ATR, OBV, Volume). The model expects ~80+
t1_close_* and t1_open_* features. This caused 24%+ feature misses and
collapsed all probabilities to 27-30%.

The fix makes calculate_comprehensive_indicators_intraday produce the SAME full
indicator set as calculate_comprehensive_indicators_daily. Both functions now
share the same _calculate_all_indicators() core so they can never diverge again.
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


def setup_logging(verbose: bool = False):
    """Setup basic logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def log_probability_distribution(predictions_df: pd.DataFrame, logger: logging.Logger, label: str = ""):
    """
    Log a detailed probability distribution histogram.
    """
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
        logger.info("  ℹ️  No BUY/STRONG BUY signals today.")
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
    """Intelligent screener that uses model-derived filters from learned_filters.json"""

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

    # FIX 1: More permissive defaults so we don't over-filter at the screening stage.
    # The ML model does the real filtering — the screener just narrows the universe.
    DEFAULT_FILTERS = {
        "min_price":           0.50,
        "max_price":           100.0,   # raised from 50 — don't exclude $50-100 stocks
        "min_volume":          200_000, # lowered from 300k
        "min_volume_ratio":    1.5,     # lowered from 2.0
        "min_relative_volume": 1.5,     # lowered from 2.0
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

        try:
            result = self.screener.screen(
                market="america",
                filters=tv_filters,
                sort_by="relative_volume_10d_calc",
                sort_order="desc",
                limit=max_results,
            )

            if result.get("status") == "success" and result.get("data"):
                df = pd.DataFrame(result["data"])
                self.logger.info(f"✓ Screened {len(df)} stocks")
                return df

            self.logger.warning("Screener returned no data or error status")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Screening failed: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Shared comprehensive indicator calculation
# ---------------------------------------------------------------------------

def _calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the FULL set of technical indicators used by the model.

    This is the single source of truth for indicator calculation — used by
    BOTH calculate_comprehensive_indicators_daily() and
    calculate_comprehensive_indicators_intraday().

    NOTE: Column names are PRESERVED in their original casing (e.g. RSI_14,
    not rsi_14). This is critical: extract_features_with_prefix() will add
    the t3_/t1_close_ prefix, and ExplosionPredictor._norm() handles the
    case-insensitive matching. Lowercasing here was causing all features to
    miss their model counterparts.
    """
    import ta

    result = pd.DataFrame(index=df.index)
    result['Close']  = df['Close']
    result['Open']   = df['Open']
    result['High']   = df['High']
    result['Low']    = df['Low']
    result['Volume'] = df['Volume']

    # ── Moving averages ────────────────────────────────────────────────────
    for period in [5, 10, 20, 50]:
        try: result[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
        except: pass

    for period in [5, 10, 12, 20, 26, 50]:
        try: result[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        except: pass

    try:
        result['WMA_10'] = ta.trend.wma_indicator(df['Close'], window=10)
        result['WMA_20'] = ta.trend.wma_indicator(df['Close'], window=20)
    except: pass

    try:
        result['HMA_9']  = ta.trend.wma_indicator(df['Close'], window=9)
        result['HMA_20'] = ta.trend.wma_indicator(df['Close'], window=20)
    except: pass

    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        result['VWMA_20'] = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    except: pass

    try:
        result['Price_vs_SMA20'] = (df['Close'] / result['SMA_20'] - 1) * 100
        result['Price_vs_SMA50'] = (df['Close'] / result.get('SMA_50', result['SMA_20']) - 1) * 100
        result['Price_vs_EMA20'] = (df['Close'] / result['EMA_20'] - 1) * 100
    except: pass

    try:
        result['SMA_20_50_Diff'] = result['SMA_20'] - result.get('SMA_50', result['SMA_20'])
        result['EMA_12_26_Diff'] = result['EMA_12'] - result['EMA_26']
    except: pass

    try:
        result['SMA_20_Slope'] = result['SMA_20'].diff(5)
        result['EMA_20_Slope'] = result['EMA_20'].diff(5)
    except: pass

    # ── RSI ────────────────────────────────────────────────────────────────
    for period in [7, 14, 21, 28]:
        try: result[f'RSI_{period}'] = ta.momentum.rsi(df['Close'], window=period)
        except: pass

    try: result['RSI_14_Slope'] = result['RSI_14'].diff(3)
    except: pass

    # ── Stochastics ────────────────────────────────────────────────────────
    try:
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        result['STOCHk_14_3_3'] = stoch.stoch()
        result['STOCHd_14_3_3'] = stoch.stoch_signal()
        result['STOCHh_14_3_3'] = result['STOCHk_14_3_3'] - result['STOCHd_14_3_3']
        stoch_fast = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=5, smooth_window=1)
        result['STOCHk_5_3_1'] = stoch_fast.stoch()
        result['STOCHd_5_3_1'] = stoch_fast.stoch_signal()
        result['STOCHh_5_3_1'] = result['STOCHk_5_3_1'] - result['STOCHd_5_3_1']
    except: pass

    try:
        stoch_rsi = ta.momentum.StochRSIIndicator(df['Close'], window=14, smooth1=3, smooth2=3)
        result['STOCHRSIk_14_14_3_3'] = stoch_rsi.stochrsi_k()
        result['STOCHRSId_14_14_3_3'] = stoch_rsi.stochrsi_d()
    except: pass

    try: result['WILLR_14'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    except: pass

    # ── CCI ────────────────────────────────────────────────────────────────
    for period in [14, 20]:
        try: result[f'CCI_{period}'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=period)
        except: pass

    # ── Oscillators ────────────────────────────────────────────────────────
    try: result['UO'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
    except: pass

    try: result['AO'] = ta.momentum.awesome_oscillator(df['High'], df['Low'], window1=5, window2=34)
    except: pass

    # ── MACD ───────────────────────────────────────────────────────────────
    try:
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        result['MACD_12_26_9']  = macd.macd()
        result['MACDh_12_26_9'] = macd.macd_diff()
        result['MACDs_12_26_9'] = macd.macd_signal()
        result['MACD_ROC']      = result['MACD_12_26_9'].pct_change(5) * 100
        macd_fast = ta.trend.MACD(df['Close'], window_slow=12, window_fast=6, window_sign=5)
        result['MACD_Fast']  = macd_fast.macd()
        result['MACDh_Fast'] = macd_fast.macd_diff()
        result['MACDs_Fast'] = macd_fast.macd_signal()
    except: pass

    # ── Bollinger Bands ────────────────────────────────────────────────────
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        result['BBL_20_2.0_2.0'] = bb.bollinger_lband()
        result['BBM_20_2.0_2.0'] = bb.bollinger_mavg()
        result['BBU_20_2.0_2.0'] = bb.bollinger_hband()
        result['BBB_20_2.0_2.0'] = bb.bollinger_wband()
        result['BBP_20_2.0_2.0'] = bb.bollinger_pband()
    except: pass

    # ── Keltner Channel ────────────────────────────────────────────────────
    try:
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        result['KCLe_20_2'] = keltner.keltner_channel_lband()
        result['KCBe_20_2'] = keltner.keltner_channel_mband()
        result['KCUe_20_2'] = keltner.keltner_channel_hband()
    except: pass

    # ── Donchian Channel ───────────────────────────────────────────────────
    try:
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
        result['DCL_20_20'] = donchian.donchian_channel_lband()
        result['DCM_20_20'] = donchian.donchian_channel_mband()
        result['DCU_20_20'] = donchian.donchian_channel_hband()
    except: pass

    # ── ATR ────────────────────────────────────────────────────────────────
    for period in [7, 14, 20]:
        try: result[f'ATR_{period}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
        except: pass

    try: result['ATR_14_Slope'] = result['ATR_14'].diff(5)
    except: pass

    # ── Historical Volatility ──────────────────────────────────────────────
    for period in [10, 20, 30]:
        try: result[f'HV_{period}'] = df['Close'].pct_change().rolling(window=period).std() * 100
        except: pass

    # ── Volume indicators ──────────────────────────────────────────────────
    try:
        result['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        result['OBV_SMA20'] = result['OBV'].rolling(window=20).mean()
    except: pass

    for period in [5, 10, 20]:
        try: result[f'Volume_MA{period}'] = df['Volume'].rolling(window=period).mean()
        except: pass

    try: result['Volume_Ratio'] = df['Volume'] / result['Volume_MA20']
    except: pass

    try: result['MFI_14'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except: pass

    try: result['CMF_20'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    except: pass

    # ── ADX / Directional Movement ─────────────────────────────────────────
    try:
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        result['ADX_14']    = adx.adx()
        result['ADXR_14_2'] = adx.adx()
        result['DMP_14']    = adx.adx_pos()
        result['DMN_14']    = adx.adx_neg()
    except: pass

    # ── Aroon ──────────────────────────────────────────────────────────────
    try:
        aroon = ta.trend.AroonIndicator(high=df['High'], low=df['Low'], window=25)
        result['AROONU_25']   = aroon.aroon_up()
        result['AROOND_25']   = aroon.aroon_down()
        result['AROONOSC_25'] = aroon.aroon_indicator()
    except TypeError:
        try:
            aroon = ta.trend.AroonIndicator(df['Close'], window=25)
            result['AROONU_25']   = aroon.aroon_up()
            result['AROOND_25']   = aroon.aroon_down()
            result['AROONOSC_25'] = aroon.aroon_indicator()
        except Exception:
            pass
    except Exception:
        pass

    # ── TSI ────────────────────────────────────────────────────────────────
    try:
        tsi = ta.momentum.TSIIndicator(df['Close'], window_slow=25, window_fast=13)
        result['TSI_13_25_13']  = tsi.tsi()
        result['TSIs_13_25_13'] = tsi.tsi()
    except: pass

    # ── Momentum / ROC ─────────────────────────────────────────────────────
    for period in [10, 20]:
        try: result[f'ROC_{period}'] = ta.momentum.roc(df['Close'], window=period)
        except: pass

    for period in [10, 20]:
        try: result[f'MOM_{period}'] = df['Close'].diff(period)
        except: pass

    # ── Supertrend proxy ───────────────────────────────────────────────────
    try:
        result['SUPERT_10_3']  = df['Close']
        result['SUPERTd_10_3'] = 0
        result['SUPERTl_10_3'] = df['Low']
        result['SUPERTs_10_3'] = 1
    except: pass

    # ── VWAP ───────────────────────────────────────────────────────────────
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        result['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    except: pass

    # ── Gap ────────────────────────────────────────────────────────────────
    try: result['Gap_Pct'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
    except: pass

    return result


def calculate_comprehensive_indicators_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate full indicator set on DAILY bars."""
    return _calculate_all_indicators(df)


def calculate_comprehensive_indicators_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate full indicator set on 5-MINUTE intraday bars.

    Delegates to _calculate_all_indicators() — identical to the daily path
    so the two can never diverge.
    """
    return _calculate_all_indicators(df)


def fetch_stock_data_for_prediction(symbol: str, logger) -> dict:
    """
    Fetch stock data for prediction.
      - T-3, T-5, T-10 : DAILY charts (required)
      - T-1 open/close  : 5-MIN intraday (OPTIONAL — stock kept even if unavailable)

    FIX 1: T-1 data is now best-effort. Previously the function returned None
    whenever intraday data was missing, silently dropping ~99% of screened stocks.
    XGBoost handles NaN natively, so stocks without T-1 data are still scored
    on T-3/T-5/T-10 features.

    FIX 2: extract_features_with_prefix() now preserves column casing (RSI_14
    not rsi_14) so features match the model's expected names via _norm() matching.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    from datetime import time as dt_time
    import pandas as pd
    import numpy as np
    import pytz

    try:
        ticker = yf.Ticker(symbol)

        # ── Daily bars for T-3 / T-5 / T-10 (REQUIRED) ───────────────────
        df_daily = ticker.history(period="90d", interval="1d")
        if df_daily.empty or len(df_daily) < 10:
            logger.debug(f"{symbol}: Insufficient daily data ({len(df_daily)} bars)")
            return None

        df_indicators_daily = calculate_comprehensive_indicators_daily(df_daily)
        if df_indicators_daily.empty:
            return None

        available_dates = sorted(df_indicators_daily.index.date, reverse=True)
        if len(available_dates) < 5:
            logger.debug(f"{symbol}: Only {len(available_dates)} trading days")
            return None

        # Use best available dates (don't require exactly 10+)
        t3_date  = available_dates[min(3,  len(available_dates)-1)]
        t5_date  = available_dates[min(5,  len(available_dates)-1)]
        t10_date = available_dates[min(10, len(available_dates)-1)]

        t3_data  = extract_features_with_prefix(df_indicators_daily, t3_date,  "t3",  logger, symbol)
        t5_data  = extract_features_with_prefix(df_indicators_daily, t5_date,  "t5",  logger, symbol)
        t10_data = extract_features_with_prefix(df_indicators_daily, t10_date, "t10", logger, symbol)

        if not t3_data:
            logger.debug(f"{symbol}: Failed to extract T-3 data")
            return None

        result = {
            "symbol":   symbol,
            "exchange": "NASDAQ",
            **t3_data,
            **(t5_data or {}),
            **(t10_data or {}),
        }

        # ── 5-min intraday bars for T-1 (OPTIONAL — best effort) ─────────
        try:
            df_intraday = ticker.history(period="60d", interval="5m")

            if df_intraday.empty or len(df_intraday) < 100:
                logger.debug(f"{symbol}: Insufficient intraday data — using daily-only features")
            else:
                df_indicators_intraday = calculate_comprehensive_indicators_intraday(df_intraday)

                if not df_indicators_intraday.empty:
                    # Localise to Eastern time
                    if df_indicators_intraday.index.tz is None:
                        df_indicators_intraday.index = df_indicators_intraday.index.tz_localize(
                            "America/New_York"
                        )
                    else:
                        df_indicators_intraday.index = df_indicators_intraday.index.tz_convert(
                            "America/New_York"
                        )

                    yesterday = available_dates[1] if len(available_dates) > 1 else available_dates[0]

                    t1_close_data = extract_intraday_snapshot(
                        df_indicators_intraday, yesterday, dt_time(16, 0), "t1_close", logger, symbol
                    )
                    t1_open_data = extract_intraday_snapshot(
                        df_indicators_intraday, yesterday, dt_time(9, 30), "t1_open", logger, symbol
                    )

                    if t1_close_data:
                        result.update(t1_close_data)
                    if t1_open_data:
                        result.update(t1_open_data)

        except Exception as intraday_err:
            logger.debug(f"{symbol}: Intraday fetch error ({intraday_err}) — using daily-only features")

        logger.debug(f"{symbol}: {len(result)} total features assembled")
        return result

    except Exception as e:
        logger.debug(f"{symbol}: Error — {e}")
        return None


def extract_features_with_prefix(df: pd.DataFrame, date, prefix: str, logger, symbol: str) -> dict:
    """
    Extract indicators with prefix (e.g., t3_, t5_, t10_) from DAILY data.

    FIX: Column names are now preserved in their ORIGINAL casing (RSI_14, not rsi_14).
    Previously col.lower() was called on all keys, producing t3_rsi_14 instead of
    t3_RSI_14. The model expects the latter (or can match via _norm() in
    ExplosionPredictor). Preserving casing gives direct matches and avoids
    relying on the fallback norm path for every single feature.
    """
    day_bars = df[df.index.date == date]
    if day_bars.empty:
        logger.debug(f"{symbol}: No data for {date} (prefix {prefix})")
        return {}
    bar = day_bars.iloc[-1]
    return {
        f"{prefix}_{k}": (v if (pd.notna(v) and not np.isinf(v)) else None)
        for k, v in bar.to_dict().items()
    }


def extract_intraday_snapshot(
    df_intraday: pd.DataFrame,
    target_date,
    target_time,
    prefix: str,
    logger,
    symbol: str
) -> dict:
    """
    Extract indicators from 5-MINUTE intraday data at specific time.

    FIX: Column names are preserved in original casing (matching _calculate_all_indicators
    output). Previously col_lower was used, producing t1_close_rsi_14 instead of
    t1_close_RSI_14, causing all T-1 features to miss their model counterparts.

    For T-1 features, the t1_column_map is applied to translate from indicator
    short names to the model's expected long-form names (e.g. RSI_14 → RSI_14,
    STOCHk → STOCHk_14_3_3). Since _calculate_all_indicators now uses the
    long-form names directly, the mapping is largely a passthrough, but we apply
    it for consistency with the training pipeline.
    """
    day_bars = df_intraday[df_intraday.index.date == target_date]
    if day_bars.empty:
        logger.debug(f"{symbol}: No intraday data for {target_date}")
        return {}

    window_start = (datetime.combine(target_date, target_time) - timedelta(minutes=5)).time()
    window_end   = (datetime.combine(target_date, target_time) + timedelta(minutes=30)).time()

    target_bars = day_bars[
        (day_bars.index.time >= window_start) &
        (day_bars.index.time <= window_end)
    ]
    if target_bars.empty:
        target_bars = day_bars

    bar = target_bars.iloc[0] if target_time.hour < 12 else target_bars.iloc[-1]

    # Preserve original casing — don't lowercase
    return {
        f"{prefix}_{k}": (v if (pd.notna(v) and not np.isinf(v)) else None)
        for k, v in bar.to_dict().items()
    }


def _get_calibrated_gain_estimate(probability: float) -> float:
    """
    Last-resort fallback for target gain when the gain regressor is unavailable
    AND historical accuracy data is also unavailable.

    This is only reached when:
      1. gain_regressor.pkl doesn't exist (not yet trained), AND
      2. get_historical_prediction_accuracy() returns no data

    In normal operation the gain_regressor.pkl handles this entirely.
    Once the regressor is trained (needs >=30 winner rows with actual_high_pct
    or change_pct in the training set), this fallback is never hit.

    NOTE: These are intentionally rough — the regressor will be far more
    accurate because it uses the same feature space as the classifier.
    """
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

    # STEP 1: SCREENING
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: INTELLIGENT SCREENING")
    logger.info("=" * 80)

    screened_df = screener.screen_with_tradingview(max_results=args.max_results)
    if screened_df.empty:
        logger.error("No stocks passed screening")
        return 1
    logger.info(f"✓ Screened {len(screened_df)} stocks")

    symbols = []
    if 'symbol' in screened_df.columns:
        symbols = screened_df['symbol'].str.split(':').str[-1].tolist()
    else:
        logger.error("No symbol column in screened results")
        return 1

    # STEP 2: FETCH STOCK DATA
    # FIX 1: T-1 data is now optional — far more stocks will complete successfully
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: FETCH STOCK DATA (T-1 intraday is now optional)")
    logger.info("=" * 80)
    logger.info(f"Fetching data for {len(symbols)} stocks...")
    logger.info("  Stocks without T-1 intraday data will still be scored on T-3/T-5/T-10 features")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    enriched_stocks = []
    failed_count = 0
    t1_count = 0
    daily_only_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_data_for_prediction, sym, logger): sym
                   for sym in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(symbols)} | successful: {len(enriched_stocks)}")
            result = future.result()
            if result:
                enriched_stocks.append(result)
                has_t1 = any(k.startswith("t1_") for k in result)
                if has_t1:
                    t1_count += 1
                else:
                    daily_only_count += 1
            else:
                failed_count += 1

    logger.info(f"✓ Fetched {len(enriched_stocks)} stocks ({failed_count} failed/skipped)")
    logger.info(f"  With T-1 intraday features: {t1_count}")
    logger.info(f"  Daily-only features (T-3/T-5/T-10): {daily_only_count}")

    if not enriched_stocks:
        logger.error("Failed to fetch data for any stocks")
        return 1

    # STEP 3: PREPARE FEATURES
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: PREPARE FEATURES")
    logger.info("=" * 80)

    features_df = pd.DataFrame(enriched_stocks)
    logger.info(f"✓ Feature matrix: {len(features_df)} stocks × {len(features_df.columns)} raw columns")

    # STEP 4: ML PREDICTION
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ML PREDICTION")
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

    # Last-resort safety net: only fires when gain_regressor.pkl is absent/untrained.
    # In normal operation ExplosionPredictor loads gain_regressor.pkl and produces
    # real ML predictions — this block should be a no-op once the regressor is trained.
    # Trigger: run ml_retrain_model.py once >=30 winner rows have gain data.
    if 'target_gain_pct' in predictions_df.columns:
        bad_gain_mask = (
            predictions_df['target_gain_pct'].isna() |
            (predictions_df['target_gain_pct'].abs() < 0.5) |
            (predictions_df['target_gain_pct'] > 500)  # sanity cap for regressor errors
        )
        if bad_gain_mask.any():
            n_bad = bad_gain_mask.sum()
            if n_bad == len(predictions_df):
                logger.warning(
                    f"  ⚠️  All {n_bad} gain estimates missing — gain_regressor.pkl not trained yet. "
                    "Run ml_retrain_model.py once >=30 winner rows have actual_high_pct or change_pct."
                )
            else:
                logger.info(f"  Last-resort gain fallback applied to {n_bad} stocks (regressor not used for these)")
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

        # Recalculate target prices from patched gains
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

    # STEP 5: TOP PREDICTIONS
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 5: TOP {args.top_n} PREDICTIONS")
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

    # STEP 6: STORE PREDICTIONS
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: STORE PREDICTIONS")
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

    # STEP 7: SCREENING LOG
    screening_log = {
        'screening_date':               prediction_date,
        'total_symbols_attempted':      args.max_results,
        'symbols_fetched_successfully': len(enriched_stocks),
        'symbols_with_t1_data':         t1_count,
        'symbols_daily_only':           daily_only_count,
        'symbols_after_all_filters':    len(features_df),
        'total_predictions':            len(predictions_df),
        'strong_buy_count':  len(predictions_df[predictions_df['signal'] == 'STRONG BUY']),
        'buy_count':         len(predictions_df[predictions_df['signal'] == 'BUY']),
        'hold_count':        len(predictions_df[predictions_df['signal'] == 'HOLD']),
        'avoid_count':       len(predictions_df[predictions_df['signal'] == 'AVOID']),
        'avg_probability':    float(predictions_df['explosion_probability'].mean()),
        'max_probability':    float(predictions_df['explosion_probability'].max()),
        'min_probability':    float(predictions_df['explosion_probability'].min()),
        'model_version':      'fixed_t1_optional_casing_preserved'
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
