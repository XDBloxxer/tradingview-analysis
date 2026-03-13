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
#
# IMPORTANT — duplicate-target handling:
#   Several source keys map to the same model name (e.g. "rsi" and "rsi14"
#   both → "RSI_14").  rename_t1_columns() resolves this by keeping only the
#   FIRST matching source column it encounters in the DataFrame (iterating in
#   dict-insertion order) and dropping all subsequent ones.  This is safe
#   because both source columns carry the same information; we only need one.

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
    # "rsi" is the canonical source; "rsi14" and "rsi[1]" are aliases.
    # rename_t1_columns keeps whichever appears first in the DataFrame.
    "rsi":              "RSI_14",
    "rsi14":            "RSI_14",       # alias — deduped at runtime
    "rsi[1]":           "RSI_14",       # shifted bar — deduped at runtime
    "rsi7":             "RSI_7",
    "rsi21":            "RSI_21",
    "rsi28":            "RSI_28",
    "rsi_14_slope":     "RSI_14_Slope",

    # ── STOCHASTIC ─────────────────────────────────────────────────────────
    # Short forms ("stoch.k/d") are canonical; long forms are aliases.
    "stoch.k":              "STOCHk_14_3_3",
    "stochk_14_3_3":        "STOCHk_14_3_3",   # alias — deduped at runtime
    "stoch.d":              "STOCHd_14_3_3",
    "stochd_14_3_3":        "STOCHd_14_3_3",   # alias — deduped at runtime
    "stochh_14_3_3":        "STOCHh_14_3_3",
    "stochk_5_3_1":         "STOCHk_5_3_1",
    "stochd_5_3_1":         "STOCHd_5_3_1",
    "stochh_5_3_1":         "STOCHh_5_3_1",

    # ── STOCH RSI ──────────────────────────────────────────────────────────
    "stochrsi_k":           "STOCHRSIk_14_14_3_3",
    "stochrsik_14_14_3_3":  "STOCHRSIk_14_14_3_3",  # alias — deduped at runtime
    "stochrsi_d":           "STOCHRSId_14_14_3_3",
    "stochrsid_14_14_3_3":  "STOCHRSId_14_14_3_3",  # alias — deduped at runtime

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
    # "atr" is canonical; "atr14" is an alias — deduped at runtime.
    "atr":              "ATR_14",
    "atr14":            "ATR_14",       # alias — deduped at runtime
    "atr7":             "ATR_7",
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
    # "adx+di" / "adx-di" are canonical; "_pos" / "_neg" are aliases.
    "adx":              "ADX_14",
    "adx+di":           "DMP_14",
    "adx_pos":          "DMP_14",       # alias — deduped at runtime
    "adx-di":           "DMN_14",
    "adx_neg":          "DMN_14",       # alias — deduped at runtime
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
    # "roc" / "roc10" and "mom" / "mom10" and "gap_%" / "gap_pct" are aliases.
    "vwap":             "VWAP",
    "cmf":              "CMF_20",
    "mfi":              "MFI_14",
    "roc":              "ROC_10",
    "roc10":            "ROC_10",       # alias — deduped at runtime
    "roc20":            "ROC_20",
    "mom":              "MOM_10",
    "mom10":            "MOM_10",       # alias — deduped at runtime
    "mom20":            "MOM_20",
    "gap_%":            "Gap_Pct",
    "gap_pct":          "Gap_Pct",      # alias — deduped at runtime
    # ── NEW: previously missing ──────────────────────────────────────────
    "hma9":                 "HMA_9",
    "hma20":                "HMA_20",
    "price_vs_sma50":       "Price_vs_SMA50",
    "sma_20_slope":         "SMA_20_Slope",
    "ema_20_slope":         "EMA_20_Slope",
    "macd_roc":             "MACD_ROC",
    "macd_fast":            "MACD_Fast",
    "macdh_fast":           "MACDh_Fast",
    "macds_fast":           "MACDs_Fast",
    "rsi_14_slope":         "RSI_14_Slope",
    "stochh_14_3_3":        "STOCHh_14_3_3",
    "stochk_5_3_1":         "STOCHk_5_3_1",
    "stochd_5_3_1":         "STOCHd_5_3_1",
    "stochh_5_3_1":         "STOCHh_5_3_1",
    "stochrsik_14_14_3_3":  "STOCHRSIk_14_14_3_3",
    "stochrsid_14_14_3_3":  "STOCHRSId_14_14_3_3",
    "cci":                  "CCI_14",
    "obv_sma20":            "OBV_SMA20",
    "adxr":                 "ADXR_14_2",
    "supert":               "SUPERT_10_3",
    "supert_d":             "SUPERTd_10_3",
    "supert_l":             "SUPERTl_10_3",
    "supert_s":             "SUPERTs_10_3",
    "mfi":                  "MFI_14",
    "roc20":                "ROC_20",
    "mom20":                "MOM_20",
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
        "rsi"  → "t1_close_RSI_14"
        "stoch.k" → "t1_close_STOCHk_14_3_3"
        "volume"  → "t1_close_Volume"

    Metadata columns (symbol, detection_date, etc.) are left unchanged.

    Duplicate-target resolution:
        When two source columns map to the same model name (e.g. "rsi" and
        "rsi14" both → "RSI_14"), only the first one found in the DataFrame
        is kept; all subsequent aliases are dropped before renaming.  This
        prevents pandas.errors.InvalidIndexError during pd.concat.

    Columns with no mapping entry are silently dropped — they were never
    used by the model.

    Args:
        df:     DataFrame as loaded from a Supabase T-1 table
        prefix: Column prefix to add ("t1_close" or "t1_open")

    Returns:
        DataFrame with renamed feature columns + original metadata columns
    """
    import pandas as pd

    rename_map: dict[str, str] = {}
    drop_cols: list[str] = []
    seen_targets: set[str] = set()   # tracks model-name targets already claimed

    for col in df.columns:
        if col in METADATA_COLS:
            continue  # preserve metadata as-is

        col_lower = col.lower()

        if col_lower in INTRADAY_TO_MODEL:
            model_name = INTRADAY_TO_MODEL[col_lower]
            target = f"{prefix}_{model_name}"

            if target in seen_targets:
                # A prior column already claimed this target — drop the alias
                drop_cols.append(col)
            else:
                rename_map[col] = target
                seen_targets.add(target)
        else:
            # No mapping exists — column is not used by the model
            drop_cols.append(col)

    result = df.drop(columns=drop_cols, errors="ignore")
    result = result.rename(columns=rename_map)

    # Belt-and-suspenders: if any duplicates slipped through, keep first occurrence
    if result.columns.duplicated().any():
        result = result.loc[:, ~result.columns.duplicated(keep="first")]

    return result


def get_t1_model_feature_names(prefix: str) -> list[str]:
    """
    Return the full list of model feature names this mapping can produce
    for a given prefix.  Deduplicates targets so each model name appears
    only once.  Useful for validating coverage.

    Args:
        prefix: "t1_close" or "t1_open"

    Returns:
        List of expected model column names (unique, insertion-ordered)
    """
    seen: set[str] = set()
    names: list[str] = []
    for model_name in INTRADAY_TO_MODEL.values():
        full = f"{prefix}_{model_name}"
        if full not in seen:
            seen.add(full)
            names.append(full)
    return names
