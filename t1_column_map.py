"""
t1_column_map.py  —  T-1 Intraday Column Name Translator

The intraday data collector (intraday_data_collector.py) stores T-1 snapshots
using lowercase short-form column names from the `ta` library:
    rsi, stoch.k, macd.macd, ema20, atr, obv, ...

The model was trained on long-form names matching the base CSV:
    RSI_14, STOCHk_14_3_3, MACD_12_26_9, EMA_20, ATR_14, OBV, ...

This module provides a single mapping used by:
  - ml_retrain_model.py   (when loading T-1 data from Supabase)
  - explosion_predictor.py (when building features for prediction)

USAGE:
    from t1_column_map import rename_t1_columns

    df = rename_t1_columns(df, prefix="t1_close")
    # renames "rsi" → "t1_close_RSI_14", "stoch.k" → "t1_close_STOCHk_14_3_3", etc.
"""

# ---------------------------------------------------------------------------
# Core mapping: intraday short name → model long name
# ---------------------------------------------------------------------------
# Keys are EXACTLY as they appear in the Supabase T-1 tables (lowercase).
# Values are EXACTLY as they appear in model_metadata.json features list
# (after stripping the t1_close_ / t1_open_ prefix).

INTRADAY_TO_MODEL: dict[str, str] = {
    # ── OHLCV ──────────────────────────────────────────────────────────────
    "close":            "Close",
    "open":             "Open",
    "high":             "High",
    "low":              "Low",
    "volume":           "Volume",

    # ── MOVING AVERAGES ────────────────────────────────────────────────────
    "sma5":             "SMA_5",
    "sma10":            "SMA_10",
    "sma20":            "SMA_20",
    "sma50":            "SMA_50",
    "ema5":             "EMA_5",
    "ema10":            "EMA_10",
    "ema12":            "EMA_12",
    "ema20":            "EMA_20",
    "ema26":            "EMA_26",
    "ema50":            "EMA_50",
    "wma10":            "WMA_10",
    "wma20":            "WMA_20",
    "hma9":             "HMA_9",
    "hma20":            "HMA_20",
    "vwma20":           "VWMA_20",

    # ── MA DERIVED ─────────────────────────────────────────────────────────
    "price_vs_sma20":       "Price_vs_SMA20",
    "price_vs_sma50":       "Price_vs_SMA50",
    "price_vs_ema20":       "Price_vs_EMA20",
    "sma_20_50_diff":       "SMA_20_50_Diff",
    "ema_12_26_diff":       "EMA_12_26_Diff",
    "sma20_slope":          "SMA_20_Slope",
    "ema20_slope":          "EMA_20_Slope",

    # ── MACD ───────────────────────────────────────────────────────────────
    "macd.macd":            "MACD_12_26_9",
    "macd_diff":            "MACDh_12_26_9",
    "macd.signal":          "MACDs_12_26_9",
    "macd_roc":             "MACD_ROC",
    "macd_fast":            "MACD_Fast",
    "macdh_fast":           "MACDh_Fast",
    "macds_fast":           "MACDs_Fast",

    # ── RSI ────────────────────────────────────────────────────────────────
    "rsi":              "RSI_14",
    "rsi7":             "RSI_7",
    "rsi14":            "RSI_14",
    "rsi21":            "RSI_21",
    "rsi28":            "RSI_28",
    "rsi_14_slope":     "RSI_14_Slope",
    "rsi[1]":           "RSI_14",   # shifted — map to same base feature

    # ── STOCHASTIC ─────────────────────────────────────────────────────────
    "stoch.k":          "STOCHk_14_3_3",
    "stoch.d":          "STOCHd_14_3_3",
    "stochk_14_3_3":    "STOCHk_14_3_3",
    "stochd_14_3_3":    "STOCHd_14_3_3",
    "stochh_14_3_3":    "STOCHh_14_3_3",
    "stochk_5_3_1":     "STOCHk_5_3_1",
    "stochd_5_3_1":     "STOCHd_5_3_1",
    "stochh_5_3_1":     "STOCHh_5_3_1",

    # ── STOCH RSI ──────────────────────────────────────────────────────────
    "stochrsi_k":       "STOCHRSIk_14_14_3_3",
    "stochrsi_d":       "STOCHRSId_14_14_3_3",
    "stochrsik_14_14_3_3": "STOCHRSIk_14_14_3_3",
    "stochrsid_14_14_3_3": "STOCHRSId_14_14_3_3",

    # ── OSCILLATORS ────────────────────────────────────────────────────────
    "w.r":              "WILLR_14",
    "cci20":            "CCI_20",
    "cci":              "CCI_14",
    "uo":               "UO",
    "ao":               "AO",
    "tsi":              "TSI_13_25_13",
    "kst":              "TSIs_13_25_13",   # closest equivalent

    # ── BOLLINGER BANDS ────────────────────────────────────────────────────
    "bb.lower":         "BBL_20_2.0_2.0",
    "bb.middle":        "BBM_20_2.0_2.0",
    "bb.upper":         "BBU_20_2.0_2.0",
    "bb_width":         "BBB_20_2.0_2.0",
    "bbpower":          "BBP_20_2.0_2.0",

    # ── KELTNER / DONCHIAN ─────────────────────────────────────────────────
    "keltner_lower":    "KCLe_20_2",
    "keltner_middle":   "KCBe_20_2",
    "keltner_upper":    "KCUe_20_2",
    "donchian_lower":   "DCL_20_20",
    "donchian_middle":  "DCM_20_20",
    "donchian_upper":   "DCU_20_20",

    # ── ATR / VOLATILITY ───────────────────────────────────────────────────
    "atr":              "ATR_14",
    "atr7":             "ATR_7",
    "atr14":            "ATR_14",
    "atr20":            "ATR_20",
    "atr_pct":          "ATR_14_Slope",   # no exact match — closest
    "volatility_10d":   "HV_10",
    "volatility_20d":   "HV_20",
    "volatility_30d":   "HV_30",

    # ── VOLUME ─────────────────────────────────────────────────────────────
    "volume_sma5":      "Volume_MA5",
    "volume_sma10":     "Volume_MA10",
    "volume_sma20":     "Volume_MA20",
    "volume_ratio":     "Volume_Ratio",
    "obv":              "OBV",
    "obv_sma20":        "OBV_SMA20",

    # ── TREND / ADX ────────────────────────────────────────────────────────
    "adx":              "ADX_14",
    "adx+di":           "DMP_14",
    "adx-di":           "DMN_14",
    "adx_pos":          "DMP_14",
    "adx_neg":          "DMN_14",
    "adxr":             "ADXR_14_2",

    # ── AROON ──────────────────────────────────────────────────────────────
    "aroon_down":       "AROOND_25",
    "aroon_up":         "AROONU_25",
    "aroon_indicator":  "AROONOSC_25",

    # ── SUPERTREND ─────────────────────────────────────────────────────────
    "supert":           "SUPERT_10_3",
    "supert_d":         "SUPERTd_10_3",
    "supert_l":         "SUPERTl_10_3",
    "supert_s":         "SUPERTs_10_3",

    # ── MISC ───────────────────────────────────────────────────────────────
    "vwap":             "VWAP",
    "cmf":              "CMF_20",
    "mfi":              "MFI_14",
    "roc":              "ROC_10",
    "roc10":            "ROC_10",
    "roc20":            "ROC_20",
    "mom":              "MOM_10",
    "mom10":            "MOM_10",
    "mom20":            "MOM_20",
    "gap_%":            "Gap_Pct",
    "gap_pct":          "Gap_Pct",
}

# Columns that are metadata — never renamed, just preserved or dropped
METADATA_COLS = {
    "id", "created_at", "updated_at", "symbol", "exchange",
    "detection_date", "snapshot_type", "snapshot_time", "snapshot_date",
    "label", "source", "sample_weight",
}


def rename_t1_columns(df, prefix: str) -> "pd.DataFrame":
    """
    Rename a T-1 DataFrame's feature columns from intraday short names
    to model long names, then add the given prefix.

    Example:
        prefix = "t1_close"
        "rsi" → "t1_close_RSI_14"
        "stoch.k" → "t1_close_STOCHk_14_3_3"
        "volume" → "t1_close_Volume"

    Metadata columns (symbol, detection_date, etc.) are left unchanged.
    Columns with no mapping are silently dropped — they were never used
    by the model anyway.

    Args:
        df:     DataFrame as loaded from Supabase T-1 table
        prefix: Column prefix to add ("t1_close" or "t1_open")

    Returns:
        DataFrame with renamed feature columns + original metadata columns
    """
    import pandas as pd

    rename_map = {}
    drop_cols = []

    for col in df.columns:
        if col in METADATA_COLS:
            continue  # keep as-is
        col_lower = col.lower()
        if col_lower in INTRADAY_TO_MODEL:
            new_name = f"{prefix}_{INTRADAY_TO_MODEL[col_lower]}"
            rename_map[col] = new_name
        else:
            drop_cols.append(col)

    result = df.drop(columns=drop_cols, errors="ignore")
    result = result.rename(columns=rename_map)
    return result


def get_t1_model_feature_names(prefix: str) -> list[str]:
    """
    Return the full list of model feature names this mapping can produce
    for a given prefix. Useful for validating coverage.

    Args:
        prefix: "t1_close" or "t1_open"

    Returns:
        List of expected model column names
    """
    seen = set()
    names = []
    for model_name in INTRADAY_TO_MODEL.values():
        full = f"{prefix}_{model_name}"
        if full not in seen:
            seen.add(full)
            names.append(full)
    return names
