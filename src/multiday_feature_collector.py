"""
MultidayFeatureCollector
========================
Computes T-3, T-5, and T-10 daily-bar features for a list of stocks on a
given detection_date and writes the results to Supabase.

Called at the END of both daily_winners_main.py and daily_non_winners_main.py,
after the T-1 intraday rows have already been written.  It mirrors the column
schema produced by backfill_multiday_features.py so that the retrain model
sees a consistent feature set across historical and live rows.

Table targets
-------------
  winners    →  winners_multiday
  non-winners →  non_winners_multiday

Both tables share the same schema:
  symbol, detection_date, <t3_* columns>, <t5_* columns>, <t10_* columns>
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping  (pandas_ta native name → base-CSV name)
# Matches the map used in backfill_multiday_features.py exactly.
# ---------------------------------------------------------------------------
PANDAS_TA_TO_BASE: Dict[str, str] = {
    # RSI variants
    "RSI_14": "rsi_14", "RSI_7": "rsi_7", "RSI_21": "rsi_21", "RSI_28": "rsi_28",
    # Moving averages
    "SMA_5": "sma_5", "SMA_10": "sma_10", "SMA_20": "sma_20", "SMA_50": "sma_50",
    "EMA_5": "ema_5", "EMA_10": "ema_10", "EMA_12": "ema_12",
    "EMA_20": "ema_20", "EMA_26": "ema_26", "EMA_50": "ema_50",
    # WMA / HMA / VWMA
    "WMA_10": "wma_10", "WMA_20": "wma_20",
    "HMA_9": "hma_9", "HMA_20": "hma_20",
    "VWMA_20": "vwma_20",
    # MACD — DB uses full suffix names
    "MACD_12_26_9": "macd_12_26_9", "MACDh_12_26_9": "macdh_12_26_9", "MACDs_12_26_9": "macds_12_26_9",
    "MACD_6_13_5": "macd_fast", "MACDh_6_13_5": "macdh_fast", "MACDs_6_13_5": "macds_fast",
    # Stochastic — pandas_ta_classic does NOT produce STOCHh; only k and d
    "STOCHk_14_3_3": "stochk_14_3_3", "STOCHd_14_3_3": "stochd_14_3_3",
    "STOCHk_5_3_1": "stochk_5_3_1", "STOCHd_5_3_1": "stochd_5_3_1",
    "STOCHh_14_3_3": "stochh_14_3_3", "STOCHh_5_3_1": "stochh_5_3_1",
    # Bollinger Bands — actual pandas_ta names use single _2.0 suffix
    "BBL_20_2.0": "bbl_20_2_0_2_0", "BBM_20_2.0": "bbm_20_2_0_2_0",
    "BBU_20_2.0": "bbu_20_2_0_2_0", "BBB_20_2.0": "bbb_20_2_0_2_0",
    "BBP_20_2.0": "bbp_20_2_0_2_0",
    # ATR
    "ATRr_14": "atr_14", "ATRr_7": "atr_7", "ATRr_20": "atr_20",
    # ADX (ADXR not available in pandas_ta_classic)
    "ADX_14": "adx_14", "DMP_14": "dmp_14", "DMN_14": "dmn_14",
    "ADXR_14_2": "adxr_14_2",
    # CCI (both periods)
    "CCI_14_0.015": "cci_14", "CCI_20_0.015": "cci_20",
    # Williams %R
    "WILLR_14": "willr_14",
    # Momentum / ROC (both periods)
    "MOM_10": "mom_10", "MOM_20": "mom_20",
    "ROC_10": "roc_10", "ROC_20": "roc_20",
    # Aroon
    "AROOND_25": "aroond_25", "AROONU_25": "aroonu_25", "AROONOSC_25": "aroonosc_25",
    # Awesome Oscillator
    "AO_5_34": "ao",
    # MFI
    "MFI_14": "mfi_14",
    # Ultimate Oscillator
    "UO_7_14_28": "uo",
    # TSI
    "TSI_13_25_13": "tsi_13_25_13", "TSIs_13_25_13": "tsis_13_25_13",
    # CMF
    "CMF_20": "cmf_20",
    # Donchian Channels
    "DCL_20_20": "dcl_20_20", "DCM_20_20": "dcm_20_20", "DCU_20_20": "dcu_20_20",
    # Keltner Channels — actual names use _2.0 suffix
    "KCLe_20_2.0": "kcle_20_2", "KCBe_20_2.0": "kcbe_20_2", "KCUe_20_2.0": "kcue_20_2",
    # OBV
    "OBV": "obv",
    # Supertrend
    "SUPERT_10_3.0": "supert_10_3", "SUPERTd_10_3.0": "supertd_10_3",
    "SUPERTs_10_3.0": "superts_10_3", "SUPERTl_10_3.0": "supertl_10_3",
    # Stoch RSI
    "STOCHRSIk_14_14_3_3": "stochrsik_14_14_3_3",
    "STOCHRSId_14_14_3_3": "stochrsid_14_14_3_3",
    # VWAP
    "VWAP_D": "vwap",
    # Manually computed fields (see helpers section below)
    # NOTE: raw OHLCV (open/high/low/close/volume) are intentionally excluded.
    # Price-level features are weak out-of-sample (splits, delistings, survivor
    # bias) and were shown to dominate feature importance (t3_high at 19.2%),
    # which indicates the model was learning price level rather than signal.
    # Only derived / normalised indicators are kept.
    "gap_pct": "gap_pct",
    "volume_ratio": "volume_ratio",
    "obv_sma20": "obv_sma20",
    "volume_ma5": "volume_ma5", "volume_ma10": "volume_ma10", "volume_ma20": "volume_ma20",
    "atr_14_slope": "atr_14_slope", "rsi_14_slope": "rsi_14_slope",
    "ema_20_slope": "ema_20_slope", "sma_20_slope": "sma_20_slope",
    "ema_12_26_diff": "ema_12_26_diff",
    "sma_20_50_diff": "sma_20_50_diff",
    "price_vs_sma20": "price_vs_sma20", "price_vs_sma50": "price_vs_sma50",
    "price_vs_ema20": "price_vs_ema20",
    "macd_roc": "macd_roc",
    "hv_10": "hv_10", "hv_20": "hv_20", "hv_30": "hv_30",
}

TIMEFRAMES = {
    "t3":  3,
    "t5":  5,
    "t10": 10,
}

# ---------------------------------------------------------------------------
# Columns that actually exist in the Supabase schema.
# Any key NOT in this set will be stripped before upsert to avoid PGRST204
# "column not found in schema cache" errors.
# Derived from the non_winners_multiday table export (non_winners_multiday_rows.csv).
# ---------------------------------------------------------------------------
DB_COLUMNS: set = {
    "symbol", "detection_date",
    # ── t10 ──────────────────────────────────────────────────────────────────
    "t10_adx_14", "t10_adxr_14_2", "t10_ao", "t10_aroond_25", "t10_aroonosc_25",
    "t10_aroonu_25", "t10_atr_14", "t10_atr_14_slope", "t10_atr_20", "t10_atr_7",
    "t10_bbb_20_2_0_2_0", "t10_bbl_20_2_0_2_0", "t10_bbm_20_2_0_2_0",
    "t10_bbp_20_2_0_2_0", "t10_bbu_20_2_0_2_0", "t10_cci_14", "t10_cci_20",
    "t10_cmf_20", "t10_dcl_20_20", "t10_dcm_20_20", "t10_dcu_20_20",
    "t10_dmn_14", "t10_dmp_14", "t10_ema_10", "t10_ema_12", "t10_ema_12_26_diff",
    "t10_ema_20", "t10_ema_20_slope", "t10_ema_26", "t10_ema_5", "t10_ema_50",
    "t10_gap_pct", "t10_hma_20", "t10_hma_9", "t10_hv_10", "t10_hv_20",
    "t10_hv_30", "t10_kcbe_20_2", "t10_kcle_20_2", "t10_kcue_20_2",
    "t10_macd_12_26_9", "t10_macd_fast", "t10_macd_roc", "t10_macdh_12_26_9",
    "t10_macdh_fast", "t10_macds_12_26_9", "t10_macds_fast", "t10_mfi_14",
    "t10_mom_10", "t10_mom_20", "t10_obv", "t10_obv_sma20",
    "t10_price_vs_ema20", "t10_price_vs_sma20", "t10_price_vs_sma50",
    "t10_roc_10", "t10_roc_20", "t10_rsi_14", "t10_rsi_14_slope", "t10_rsi_21",
    "t10_rsi_28", "t10_rsi_7", "t10_sma_10", "t10_sma_20", "t10_sma_20_50_diff",
    "t10_sma_20_slope", "t10_sma_5", "t10_sma_50", "t10_stochd_14_3_3",
    "t10_stochd_5_3_1", "t10_stochh_14_3_3", "t10_stochh_5_3_1",
    "t10_stochk_14_3_3", "t10_stochk_5_3_1", "t10_stochrsid_14_14_3_3",
    "t10_stochrsik_14_14_3_3", "t10_supert_10_3", "t10_supertd_10_3",
    "t10_supertl_10_3", "t10_superts_10_3", "t10_tsi_13_25_13", "t10_tsis_13_25_13",
    "t10_uo", "t10_volume_ma10", "t10_volume_ma20", "t10_volume_ma5",
    "t10_volume_ratio", "t10_vwap", "t10_vwma_20", "t10_willr_14", "t10_wma_10",
    "t10_wma_20",
    # ── t3 ───────────────────────────────────────────────────────────────────
    "t3_adx_14", "t3_adxr_14_2", "t3_ao", "t3_aroond_25", "t3_aroonosc_25",
    "t3_aroonu_25", "t3_atr_14", "t3_atr_14_slope", "t3_atr_20", "t3_atr_7",
    "t3_bbb_20_2_0_2_0", "t3_bbl_20_2_0_2_0", "t3_bbm_20_2_0_2_0",
    "t3_bbp_20_2_0_2_0", "t3_bbu_20_2_0_2_0", "t3_cci_14", "t3_cci_20",
    "t3_cmf_20", "t3_dcl_20_20", "t3_dcm_20_20", "t3_dcu_20_20",
    "t3_dmn_14", "t3_dmp_14", "t3_ema_10", "t3_ema_12", "t3_ema_12_26_diff",
    "t3_ema_20", "t3_ema_20_slope", "t3_ema_26", "t3_ema_5", "t3_ema_50",
    "t3_gap_pct", "t3_hma_20", "t3_hma_9", "t3_hv_10", "t3_hv_20",
    "t3_hv_30", "t3_kcbe_20_2", "t3_kcle_20_2", "t3_kcue_20_2",
    "t3_macd_12_26_9", "t3_macd_fast", "t3_macd_roc", "t3_macdh_12_26_9",
    "t3_macdh_fast", "t3_macds_12_26_9", "t3_macds_fast", "t3_mfi_14",
    "t3_mom_10", "t3_mom_20", "t3_obv", "t3_obv_sma20",
    "t3_price_vs_ema20", "t3_price_vs_sma20", "t3_price_vs_sma50",
    "t3_roc_10", "t3_roc_20", "t3_rsi_14", "t3_rsi_14_slope", "t3_rsi_21",
    "t3_rsi_28", "t3_rsi_7", "t3_sma_10", "t3_sma_20", "t3_sma_20_50_diff",
    "t3_sma_20_slope", "t3_sma_5", "t3_sma_50", "t3_stochd_14_3_3",
    "t3_stochd_5_3_1", "t3_stochh_14_3_3", "t3_stochh_5_3_1",
    "t3_stochk_14_3_3", "t3_stochk_5_3_1", "t3_stochrsid_14_14_3_3",
    "t3_stochrsik_14_14_3_3", "t3_supert_10_3", "t3_supertd_10_3",
    "t3_supertl_10_3", "t3_superts_10_3", "t3_tsi_13_25_13", "t3_tsis_13_25_13",
    "t3_uo", "t3_volume_ma10", "t3_volume_ma20", "t3_volume_ma5",
    "t3_volume_ratio", "t3_vwap", "t3_vwma_20", "t3_willr_14", "t3_wma_10",
    "t3_wma_20",
    # ── t5 ───────────────────────────────────────────────────────────────────
    "t5_adx_14", "t5_adxr_14_2", "t5_ao", "t5_aroond_25", "t5_aroonosc_25",
    "t5_aroonu_25", "t5_atr_14", "t5_atr_14_slope", "t5_atr_20", "t5_atr_7",
    "t5_bbb_20_2_0_2_0", "t5_bbl_20_2_0_2_0", "t5_bbm_20_2_0_2_0",
    "t5_bbp_20_2_0_2_0", "t5_bbu_20_2_0_2_0", "t5_cci_14", "t5_cci_20",
    "t5_cmf_20", "t5_dcl_20_20", "t5_dcm_20_20", "t5_dcu_20_20",
    "t5_dmn_14", "t5_dmp_14", "t5_ema_10", "t5_ema_12", "t5_ema_12_26_diff",
    "t5_ema_20", "t5_ema_20_slope", "t5_ema_26", "t5_ema_5", "t5_ema_50",
    "t5_gap_pct", "t5_hma_20", "t5_hma_9", "t5_hv_10", "t5_hv_20",
    "t5_hv_30", "t5_kcbe_20_2", "t5_kcle_20_2", "t5_kcue_20_2",
    "t5_macd_12_26_9", "t5_macd_fast", "t5_macd_roc", "t5_macdh_12_26_9",
    "t5_macdh_fast", "t5_macds_12_26_9", "t5_macds_fast", "t5_mfi_14",
    "t5_mom_10", "t5_mom_20", "t5_obv", "t5_obv_sma20",
    "t5_price_vs_ema20", "t5_price_vs_sma20", "t5_price_vs_sma50",
    "t5_roc_10", "t5_roc_20", "t5_rsi_14", "t5_rsi_14_slope", "t5_rsi_21",
    "t5_rsi_28", "t5_rsi_7", "t5_sma_10", "t5_sma_20", "t5_sma_20_50_diff",
    "t5_sma_20_slope", "t5_sma_5", "t5_sma_50", "t5_stochd_14_3_3",
    "t5_stochd_5_3_1", "t5_stochh_14_3_3", "t5_stochh_5_3_1",
    "t5_stochk_14_3_3", "t5_stochk_5_3_1", "t5_stochrsid_14_14_3_3",
    "t5_stochrsik_14_14_3_3", "t5_supert_10_3", "t5_supertd_10_3",
    "t5_supertl_10_3", "t5_superts_10_3", "t5_tsi_13_25_13", "t5_tsis_13_25_13",
    "t5_uo", "t5_volume_ma10", "t5_volume_ma20", "t5_volume_ma5",
    "t5_volume_ratio", "t5_vwap", "t5_vwma_20", "t5_willr_14", "t5_wma_10",
    "t5_wma_20",
}

MAX_WORKERS = 5
LOOKBACK_BARS = 260   # ~1 year of daily bars — enough for SMA_200

# Minimum trading-day history thresholds
MIN_BARS_FULL    = 50   # Need this many bars for SMA_50 / EMA_50
MIN_BARS_VIABLE  = 20   # Below this, most multi-period indicators are meaningless
MIN_BARS_WARN    = 50   # Emit a warning and set insufficient_history flag

# Map each strategy indicator to the minimum number of bars it needs.
# Indicators not listed here have a min-bar of 1 (they work on any length).
# These minimums match the indicator's own look-back period (period + warmup).
_INDICATOR_MIN_BARS: Dict[str, int] = {
    # RSI variants (need period + 1 warmup)
    "rsi_14":        15,
    "rsi_7":          8,
    "rsi_21":        22,
    "rsi_28":        29,
    # SMAs
    "sma_5":          5,
    "sma_10":        10,
    "sma_20":        20,
    "sma_50":        50,
    # EMAs (technically work sooner but values are unreliable before period bars)
    "ema_5":          5,
    "ema_10":        10,
    "ema_12":        12,
    "ema_20":        20,
    "ema_26":        26,
    "ema_50":        50,
    # WMA / HMA / VWMA
    "wma_10":        10,
    "wma_20":        20,
    "hma_9":          9,
    "hma_20":        20,
    "vwma_20":       20,
    # MACD needs slow EMA + signal warmup
    "macd_12_26_9":  35,   # 26 + 9
    "macd_6_13_5":   18,   # 13 + 5
    # Stochastic (k period + smooth_k + d)
    "stoch_14_3_3":  20,   # 14 + 3 + 3
    "stoch_5_3_1":    9,   # 5  + 3 + 1
    # Bollinger Bands
    "bbands_20":     20,
    # ATR
    "atr_14":        14,
    "atr_7":          7,
    "atr_20":        20,
    # ADX (needs 2× period internally)
    "adx_14":        28,
    # CCI
    "cci_14":        14,
    "cci_20":        20,
    # Williams %R
    "willr_14":      14,
    # Momentum / ROC
    "mom_10":        10,
    "mom_20":        20,
    "roc_10":        10,
    "roc_20":        20,
    # Aroon
    "aroon_25":      25,
    # Awesome Oscillator (uses 34-period SMA internally)
    "ao":            34,
    # MFI
    "mfi_14":        14,
    # Ultimate Oscillator
    "uo":            28,   # longest period is 28
    # TSI (slow + signal)
    "tsi_13_25_13":  38,   # 25 + 13
    # CMF
    "cmf_20":        20,
    # Donchian / Keltner
    "donchian_20":   20,
    "kc_20":         20,
    # OBV — no min
    # Supertrend
    "supertrend_10": 10,
    # Stoch RSI (rsi_length + stoch k + d)
    "stochrsi_14":   31,   # 14 + 14 + 3
    # VWAP — daily anchor, works with 1 bar
}


# ---------------------------------------------------------------------------
# Core indicator calculation  (daily bars → single-row snapshot)
# ---------------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run pandas_ta on a daily OHLCV DataFrame and return a flat dict with
    *base-CSV-style* column names (without any timeframe prefix).

    `df` should already be sliced so that its LAST row is the snapshot day.
    """
    if len(df) < 5:
        return {}

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure the standard column names yfinance uses
    for src, dst in [("adj close", "close"), ("adj_close", "close")]:
        if src in df.columns and "close" not in df.columns:
            df.rename(columns={src: "close"}, inplace=True)

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.debug(f"Missing OHLCV columns: {missing}")
        return {}

    df["volume"] = df["volume"].replace(0, np.nan)

    n_bars = len(df)

    # ── Build the indicator list, skipping any whose look-back exceeds n_bars.
    #
    # Root fix for the pandas_ta "iloc cannot enlarge its target object" bug:
    # when a DataFrame is shorter than an indicator's period, pandas_ta tries
    # to write results into a pre-allocated (too-small) slice and raises an
    # IndexError.  Instead of catching the error after the fact (which silently
    # drops ALL indicators), we exclude under-qualified indicators up front so
    # the remaining ones compute cleanly.
    # -------------------------------------------------------------------------
    def _meets_min_bars(spec: dict) -> bool:
        """Return True if n_bars satisfies the indicator's minimum look-back."""
        kind   = spec.get("kind", "")
        length = spec.get("length", spec.get("slow", spec.get("upper_length", 1)))
        for key, min_b in _INDICATOR_MIN_BARS.items():
            if key.startswith(kind) and key.endswith(str(length)):
                return n_bars >= min_b
        # Generic fallback: require at least the dominant period.
        return n_bars >= max(int(length), 1)

    all_indicators = [
        # RSI
        {"kind": "rsi",       "length": 14},
        {"kind": "rsi",       "length": 7},
        {"kind": "rsi",       "length": 21},
        {"kind": "rsi",       "length": 28},
        # SMA / EMA / WMA / HMA / VWMA
        {"kind": "sma",       "length": 5},
        {"kind": "sma",       "length": 10},
        {"kind": "sma",       "length": 20},
        {"kind": "sma",       "length": 50},
        {"kind": "ema",       "length": 5},
        {"kind": "ema",       "length": 10},
        {"kind": "ema",       "length": 12},
        {"kind": "ema",       "length": 20},
        {"kind": "ema",       "length": 26},
        {"kind": "ema",       "length": 50},
        {"kind": "wma",       "length": 10},
        {"kind": "wma",       "length": 20},
        {"kind": "hma",       "length": 9},
        {"kind": "hma",       "length": 20},
        {"kind": "vwma",      "length": 20},
        # MACD (standard + fast)
        {"kind": "macd",      "fast": 12, "slow": 26, "signal": 9},
        {"kind": "macd",      "fast": 6,  "slow": 13, "signal": 5},
        # Stochastic (two param sets)
        {"kind": "stoch",     "k": 14, "d": 3, "smooth_k": 3},
        {"kind": "stoch",     "k": 5,  "d": 3, "smooth_k": 1},
        # Bollinger Bands
        {"kind": "bbands",    "length": 20, "std": 2.0},
        # ATR
        {"kind": "atr",       "length": 14},
        {"kind": "atr",       "length": 7},
        {"kind": "atr",       "length": 20},
        # ADX
        {"kind": "adx",       "length": 14},
        # CCI (two periods)
        {"kind": "cci",       "length": 14},
        {"kind": "cci",       "length": 20},
        # Williams %R
        {"kind": "willr",     "length": 14},
        # Momentum / ROC (two periods)
        {"kind": "mom",       "length": 10},
        {"kind": "mom",       "length": 20},
        {"kind": "roc",       "length": 10},
        {"kind": "roc",       "length": 20},
        # Aroon
        {"kind": "aroon",     "length": 25},
        # Awesome Oscillator
        {"kind": "ao"},
        # MFI
        {"kind": "mfi",       "length": 14},
        # Ultimate Oscillator
        {"kind": "uo"},
        # TSI
        {"kind": "tsi",       "fast": 13, "slow": 25, "signal": 13},
        # CMF
        {"kind": "cmf",       "length": 20},
        # Donchian Channels
        {"kind": "donchian",  "lower_length": 20, "upper_length": 20},
        # Keltner Channels
        {"kind": "kc",        "length": 20, "scalar": 2},
        # OBV
        {"kind": "obv"},
        # Supertrend
        {"kind": "supertrend","length": 10, "multiplier": 3.0},
        # Stoch RSI
        {"kind": "stochrsi",  "length": 14, "rsi_length": 14, "k": 3, "d": 3},
        # VWAP
        {"kind": "vwap"},
    ]

    eligible = [s for s in all_indicators if _meets_min_bars(s)]
    skipped  = [s["kind"] for s in all_indicators if not _meets_min_bars(s)]
    if skipped:
        logger.debug(
            f"_compute_indicators: {n_bars} bars available — skipping "
            f"{len(skipped)} indicator(s) that need more history: {skipped}"
        )

    # ── Run pandas_ta strategy (only eligible indicators) ───────────────────────
    if eligible:
        try:
            df.ta.strategy(
                ta.Strategy("multiday", eligible),
                verbose=False,
            )
        except Exception as exc:
            # The iloc bug should no longer fire because we pre-filtered, but
            # keep a catch-all so any other unexpected error degrades gracefully.
            logger.warning(
                f"pandas_ta strategy error ({n_bars} bars, "
                f"{len(eligible)} indicators): {exc}"
            )

    # ── Local series references (used for derived indicators below) ──────────
    # Raw OHLCV are NOT stored as features (see PANDAS_TA_TO_BASE); they are
    # only used here to compute normalised / derived signals.
    close  = df["close"]
    volume = df["volume"]
    open_  = df["open"]

    # ── Volume helpers ───────────────────────────────────────────────────────
    df["volume_ma5"]  = volume.rolling(5).mean()
    df["volume_ma10"] = volume.rolling(10).mean()
    df["volume_ma20"] = volume.rolling(20).mean()
    vol_ma20 = df["volume_ma20"]
    df["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # ── OBV SMA ──────────────────────────────────────────────────────────────
    if "OBV" in df.columns:
        df["obv_sma20"] = df["OBV"].rolling(20).mean()

    # ── Gap pct ──────────────────────────────────────────────────────────────
    if len(df) > 1:
        df["gap_pct"] = (open_ - close.shift(1)) / close.shift(1) * 100
    else:
        df["gap_pct"] = np.nan

    # ── Derived price metrics ─────────────────────────────────────────────────
    # price_vs_* are computed here from raw dollar MAs — result is already %.
    # sma_20_slope, ema_20_slope, ema_12_26_diff, sma_20_50_diff are computed
    # here as intermediate dollar values but will be OVERWRITTEN later in the
    # normalisation block (after MAs are converted to % of close), so their
    # final stored values will be in %-point-per-bar / %-point-spread units.
    if "SMA_20" in df.columns:
        df["price_vs_sma20"]  = (close - df["SMA_20"])  / df["SMA_20"]  * 100
        df["sma_20_slope"]    = df["SMA_20"].diff(1)      # overwritten after normalisation
        df["sma_20_50_diff"]  = df["SMA_20"] - df.get("SMA_50", np.nan)  # overwritten
    if "SMA_50" in df.columns:
        df["price_vs_sma50"]  = (close - df["SMA_50"])  / df["SMA_50"]  * 100
    if "EMA_20" in df.columns:
        df["price_vs_ema20"]  = (close - df["EMA_20"])  / df["EMA_20"]  * 100
        df["ema_20_slope"]    = df["EMA_20"].diff(1)      # overwritten after normalisation
    if "EMA_12" in df.columns and "EMA_26" in df.columns:
        df["ema_12_26_diff"]  = df["EMA_12"] - df["EMA_26"]  # overwritten after normalisation

    # ── Slopes ───────────────────────────────────────────────────────────────
    if "ATRr_14" in df.columns:
        df["atr_14_slope"] = df["ATRr_14"].diff(1)
    if "RSI_14" in df.columns:
        df["rsi_14_slope"] = df["RSI_14"].diff(1)

    # ── MACD ROC ─────────────────────────────────────────────────────────────
    if "MACD_12_26_9" in df.columns:
        df["macd_roc"] = df["MACD_12_26_9"].pct_change(1) * 100

    # ── Historical Volatility ─────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    df["hv_10"] = log_ret.rolling(10).std() * np.sqrt(252) * 100
    df["hv_20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["hv_30"] = log_ret.rolling(30).std() * np.sqrt(252) * 100

    # ── ADXR (pandas_ta does not output ADXR; compute as 2-period SMA of ADX) ──────
    if "ADX_14" in df.columns:
        df["ADXR_14_2"] = df["ADX_14"].rolling(2).mean()

    # ── STOCHh (pandas_ta stoch only outputs k/d; h = rolling max of smoothed k) ────
    if "STOCHk_14_3_3" in df.columns:
        df["STOCHh_14_3_3"] = df["STOCHk_14_3_3"].rolling(3).max()
    if "STOCHk_5_3_1" in df.columns:
        df["STOCHh_5_3_1"] = df["STOCHk_5_3_1"].rolling(1).max()

    # ── SUPERTs/SUPERTl (pandas_ta doesn't output these; reconstruct from direction) ──
    if "SUPERTd_10_3.0" in df.columns and "SUPERT_10_3.0" in df.columns:
        # Upper band (resistance): SUPERT value when bearish (direction == -1)
        upper = df["SUPERT_10_3.0"].where(df["SUPERTd_10_3.0"] == -1)
        df["SUPERTs_10_3.0"] = upper.ffill()
        # Lower band (support): SUPERT value when bullish (direction == 1)
        lower = df["SUPERT_10_3.0"].where(df["SUPERTd_10_3.0"] == 1)
        df["SUPERTl_10_3.0"] = lower.ffill()


    # ── Normalise price-level indicators ─────────────────────────────────────
    # Any indicator whose value is in dollar terms will act as a price-level
    # proxy: a $2 stock always has lower band values near $1, a $50 stock near
    # $40, so the model learns "price range" rather than the intended signal.
    # We already exclude raw OHLCV (t3_close etc.) for the same reason.
    #
    # Strategy per category:
    #   DOLLAR bands/lines  → (value / close - 1) * 100   = % distance from close
    #   SIGNED dollar diffs → value / close * 100          = % of close
    #   Volume absolutes    → value / vol_ma20             = ratio (already done
    #                         for volume_ratio; apply same to MAs and OBV)
    #
    # NOTE: atr_14_slope is diff(ATRr_14) where ATRr is already % of close,
    # so its slope is a change-in-% — no further normalisation needed.
    # Slopes of dollar MAs (ema_20_slope, sma_20_slope) are normalised below.
    # ─────────────────────────────────────────────────────────────────────────

    _close = df["close"]
    _safe_close = _close.replace(0, np.nan)

    # ── Moving averages (SMA / EMA / WMA / HMA / VWMA) → % distance from close ──
    for _ma_col in [
        "SMA_5", "SMA_10", "SMA_20", "SMA_50",
        "EMA_5", "EMA_10", "EMA_12", "EMA_20", "EMA_26", "EMA_50",
        "WMA_10", "WMA_20",
        "HMA_9", "HMA_20",
        "VWMA_20",
    ]:
        if _ma_col in df.columns:
            df[_ma_col] = (df[_ma_col] / _safe_close - 1) * 100

    # ── Bollinger Bands: lower/middle/upper → % distance from close ──────────
    # BBB (bandwidth) and BBP (%B) are already unitless — leave them alone.
    for _bb_col in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]:
        if _bb_col in df.columns:
            df[_bb_col] = (df[_bb_col] / _safe_close - 1) * 100

    # ── Donchian Channels → % distance from close ────────────────────────────
    for _dc_col in ["DCL_20_20", "DCM_20_20", "DCU_20_20"]:
        if _dc_col in df.columns:
            df[_dc_col] = (df[_dc_col] / _safe_close - 1) * 100

    # ── Keltner Channels → % distance from close ─────────────────────────────
    for _kc_col in ["KCLe_20_2.0", "KCBe_20_2.0", "KCUe_20_2.0"]:
        if _kc_col in df.columns:
            df[_kc_col] = (df[_kc_col] / _safe_close - 1) * 100

    # ── Supertrend bands → % distance from close ─────────────────────────────
    # SUPERTd (direction ±1) is already unitless.
    for _st_col in ["SUPERT_10_3.0", "SUPERTs_10_3.0", "SUPERTl_10_3.0"]:
        if _st_col in df.columns:
            df[_st_col] = (df[_st_col] / _safe_close - 1) * 100

    # ── VWAP → % distance from close ─────────────────────────────────────────
    if "VWAP_D" in df.columns:
        df["VWAP_D"] = (df["VWAP_D"] / _safe_close - 1) * 100

    # ── MACD lines & signal → % of close ─────────────────────────────────────
    # MACD = EMA_fast - EMA_slow, so it's a dollar difference.
    # Dividing by close gives a scale-free momentum measure.
    for _macd_col in [
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "MACD_6_13_5",  "MACDh_6_13_5",  "MACDs_6_13_5",
    ]:
        if _macd_col in df.columns:
            df[_macd_col] = df[_macd_col] / _safe_close * 100

    # ── Momentum (MOM) → % of close ──────────────────────────────────────────
    # MOM_n = close - close[n], a dollar difference.
    for _mom_col in ["MOM_10", "MOM_20"]:
        if _mom_col in df.columns:
            df[_mom_col] = df[_mom_col] / _safe_close * 100

    # ── Awesome Oscillator → % of close ──────────────────────────────────────
    # AO = SMA5(midprice) - SMA34(midprice), a dollar difference.
    if "AO_5_34" in df.columns:
        df["AO_5_34"] = df["AO_5_34"] / _safe_close * 100

    # ── MA slopes → % of close ───────────────────────────────────────────────
    # These are computed as .diff(1) of dollar MAs above; re-derive after
    # the MAs have been normalised so slopes are in %-point-per-bar units.
    if "EMA_20" in df.columns:
        df["ema_20_slope"] = df["EMA_20"].diff(1)   # already in % after normalisation
    if "SMA_20" in df.columns:
        df["sma_20_slope"] = df["SMA_20"].diff(1)

    # ── EMA / SMA cross-spreads → %-point spread ──────────────────────────────
    # After normalising each MA to % of close, the diff is a %-point spread.
    if "EMA_12" in df.columns and "EMA_26" in df.columns:
        df["ema_12_26_diff"] = df["EMA_12"] - df["EMA_26"]   # both now % of close
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        df["sma_20_50_diff"] = df["SMA_20"] - df["SMA_50"]   # both now % of close

    # ── OBV and its SMA → ratio to vol_ma20 ──────────────────────────────────
    # Raw OBV is a cumulative volume number that varies wildly across stocks
    # (a high-float $50 stock vs a thinly-traded $2 stock).  Dividing by
    # vol_ma20 expresses it as "how many average-day-volumes is the OBV?",
    # making it comparable across market-cap tiers.
    _safe_vol_ma20 = df["volume_ma20"].replace(0, np.nan) if "volume_ma20" in df.columns else None
    if "OBV" in df.columns and _safe_vol_ma20 is not None:
        df["OBV"] = df["OBV"] / _safe_vol_ma20
    if "obv_sma20" in df.columns and _safe_vol_ma20 is not None:
        df["obv_sma20"] = df["obv_sma20"] / _safe_vol_ma20

    # ── Volume MAs → ratio to vol_ma20 ───────────────────────────────────────
    # These are intermediate values used to compute volume_ratio; keeping them
    # as raw share counts makes them price-cap proxies (high-float stocks have
    # larger volume in absolute terms).  Express as ratio to 20-day avg volume.
    if _safe_vol_ma20 is not None:
        if "volume_ma5" in df.columns:
            df["volume_ma5"]  = df["volume_ma5"]  / _safe_vol_ma20
        if "volume_ma10" in df.columns:
            df["volume_ma10"] = df["volume_ma10"] / _safe_vol_ma20
        # vol_ma20 / vol_ma20 = 1.0 always — useful as a sanity-check constant,
        # but not informative.  Zero it out so it adds no noise.
        df["volume_ma20"] = 1.0

    # ── Extract last row and map column names ───────────────────────────────
    last = df.iloc[-1]
    result: Dict[str, Any] = {}

    for raw_col, base_name in PANDAS_TA_TO_BASE.items():
        if raw_col in df.columns:
            val = last.get(raw_col, np.nan)
            result[base_name] = None if (pd.isna(val) or np.isinf(val)) else float(val)

    # Internal metadata — not written to DB (stripped by _sanitize), but used
    # by _process_symbol to detect stocks with insufficient trading history.
    result["_bar_count"] = n_bars

    return result


def _snapshot_for_offset(
    daily_df: pd.DataFrame,
    detection_date: pd.Timestamp,
    offset_days: int,
) -> Dict[str, Any]:
    """
    Return the indicator snapshot for the trading day that is `offset_days`
    calendar days before `detection_date`.

    We include all bars UP TO AND INCLUDING that day so that moving averages
    have history.
    """
    target = detection_date - timedelta(days=offset_days)

    # Find the closest available bar on or before `target`
    available = daily_df[daily_df.index <= target]
    if available.empty:
        return {}

    # Slice up to and including that bar
    snap_date = available.index[-1]
    slice_df = daily_df[daily_df.index <= snap_date].copy()

    return _compute_indicators(slice_df)


# ---------------------------------------------------------------------------
# Per-symbol worker
# ---------------------------------------------------------------------------

def _process_symbol(
    symbol: str,
    detection_date: pd.Timestamp,
) -> Optional[Dict[str, Any]]:
    """
    Fetch daily bars for `symbol`, then compute t3/t5/t10 snapshots.
    Returns a flat dict ready to upsert, or None on failure.
    """
    try:
        fetch_start = detection_date - timedelta(days=LOOKBACK_BARS + 20)
        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start=fetch_start.strftime("%Y-%m-%d"),
            end=(detection_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )

        if raw is None or raw.empty:
            logger.debug(f"{symbol}: no daily bar data")
            return None

        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        # ── Detect stocks with insufficient trading history ─────────────────
        # Count bars up to detection_date (excluding the future).
        bars_available = int((raw.index <= detection_date).sum())
        is_short_history = bars_available < MIN_BARS_WARN
        if is_short_history:
            logger.warning(
                f"{symbol}: only {bars_available} trading day(s) of history "
                f"(need {MIN_BARS_WARN} for full feature set). "
                f"Long-period indicators (SMA_50, EMA_50, MACD, ADX, ...) will "
                f"be NaN/absent.  Model score should be treated as LOW CONFIDENCE."
            )

        row: Dict[str, Any] = {
            "symbol":               symbol,
            "detection_date":       detection_date.date().isoformat(),
        }

        for prefix, offset in TIMEFRAMES.items():
            snap = _snapshot_for_offset(raw, detection_date, offset)
            for base_name, val in snap.items():
                # Strip internal metadata keys — they are not DB columns.
                if base_name.startswith("_"):
                    continue
                row[f"{prefix}_{base_name}"] = val

        return row

    except Exception as exc:
        logger.warning(f"{symbol}: failed to compute multiday features — {exc}")
        return None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MultidayFeatureCollector:
    """
    Compute and persist T-3/T-5/T-10 daily-bar features for a batch of stocks.

    Usage
    -----
    collector = MultidayFeatureCollector(config)

    # For winners:
    collector.collect_and_write(symbols_and_dates, table="winners_multiday")

    # For non-winners:
    collector.collect_and_write(symbols_and_dates, table="non_winners_multiday")
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set."
            )
        self.client: Client = create_client(supabase_url, supabase_key)
        self.logger.info("MultidayFeatureCollector: Supabase connected.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _existing_symbols(self, table: str, detection_date: str) -> set:
        try:
            resp = (
                self.client.table(table)
                .select("symbol")
                .eq("detection_date", detection_date)
                .execute()
            )
            return {r["symbol"] for r in (resp.data or [])}
        except Exception as exc:
            self.logger.debug(f"Could not check existing rows in {table}: {exc}")
            return set()

    @staticmethod
    def _sanitize(row: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}
        skipped = []
        for k, v in row.items():
            if k not in DB_COLUMNS:
                skipped.append(k)
                continue
            if v is None:
                clean[k] = None
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v) if not (np.isnan(v) or np.isinf(v)) else None
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            else:
                clean[k] = v
        if skipped:
            logger.debug(
                f"_sanitize: dropped {len(skipped)} key(s) not in DB schema: {skipped}"
            )
        return clean

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def collect_and_write(
        self,
        stocks: List[Dict[str, Any]],
        table: str,
        allow_append: bool = False,
    ) -> int:
        """
        Compute multiday features for every stock in `stocks` and upsert
        to `table`.

        Parameters
        ----------
        stocks : list of dicts, each with at minimum:
            - "symbol"         : str
            - "detection_date" : str  (YYYY-MM-DD)  OR  datetime/Timestamp
        table  : Supabase table name
            e.g. "winners_multiday" or "non_winners_multiday"
        allow_append : bool, default False
            When False (the safe default for scheduled runs), symbols that
            already exist for the date are skipped.  Set to True when running
            manually to allow writing new stocks into a date that already has
            records in the database.

        Returns
        -------
        int  number of rows written
        """
        if not stocks:
            self.logger.warning("MultidayFeatureCollector: no stocks to process.")
            return 0

        # Normalise detection_date to a single Timestamp per symbol.
        # If all stocks share the same date (the normal daily-run case) we
        # only need one _existing_symbols check.
        date_str = str(stocks[0]["detection_date"])[:10]

        if allow_append:
            self.logger.info(
                f"MultidayFeatureCollector: allow_append=True — skipping "
                f"duplicate check for {table} on {date_str}"
            )
            existing: set = set()
        else:
            existing = self._existing_symbols(table, date_str)

        tasks = []
        for s in stocks:
            sym = s.get("symbol")
            if not sym:
                continue
            if sym in existing:
                self.logger.debug(f"  {sym}: already in {table}, skipping.")
                continue
            det = pd.Timestamp(str(s["detection_date"])[:10])
            tasks.append((sym, det))

        if not tasks:
            self.logger.info(f"MultidayFeatureCollector: all rows already in {table}.")
            return 0

        self.logger.info(
            f"MultidayFeatureCollector: computing multiday features for "
            f"{len(tasks)} symbol(s) → {table}"
        )

        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_process_symbol, sym, det): sym
                for sym, det in tasks
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    result = fut.result()
                    if result:
                        rows.append(self._sanitize(result))
                        self.logger.debug(f"  ✓ {sym}")
                    else:
                        self.logger.debug(f"  ✗ {sym}: no data returned")
                except Exception as exc:
                    self.logger.warning(f"  ✗ {sym}: {exc}")

        if not rows:
            self.logger.warning("MultidayFeatureCollector: no rows to write.")
            return 0

        # Insert in chunks of 500 to stay under request-size limits.
        # When allow_append=True, use upsert with ignore_duplicates=True so any
        # symbols that already exist for this date are silently skipped rather
        # than hard-erroring on the unique constraint.
        written = 0
        chunk_size = 500
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            try:
                if allow_append:
                    self.client.table(table).upsert(
                        chunk,
                        ignore_duplicates=True,
                        on_conflict="symbol,detection_date",
                    ).execute()
                else:
                    self.client.table(table).insert(chunk).execute()
                written += len(chunk)
            except Exception as exc:
                self.logger.error(
                    f"MultidayFeatureCollector: insert error for chunk "
                    f"{i}–{i+len(chunk)}: {exc}"
                )

        self.logger.info(
            f"MultidayFeatureCollector: wrote {written} row(s) to {table}."
        )
        return written
