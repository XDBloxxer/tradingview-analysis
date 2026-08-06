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
    # WMA_10/20, HMA_9/20, VWMA_20, EMA_12/26 pruned (2026-08-06): no code in
    # intraday_data_collector.py ever computes these — they were permanently
    # all-NaN for every T-1 row. If any of these are wanted as real T-1
    # features, implement the computation in intraday_data_collector.py
    # first, then add the mapping back here.
    "sma5":             "SMA_5",
    "sma10":            "SMA_10",
    "sma20":            "SMA_20",
    "sma50":            "SMA_50",
    "ema5":             "EMA_5",
    "ema10":            "EMA_10",
    "ema20":            "EMA_20",
    "ema50":            "EMA_50",

    # ── MA DERIVED ─────────────────────────────────────────────────────────
    # EMA_12_26_Diff pruned along with EMA_12/EMA_26 above (same reason).
    "price_vs_sma20":       "Price_vs_SMA20",
    "price_vs_sma50":       "Price_vs_SMA50",
    "price_vs_ema20":       "Price_vs_EMA20",
    "sma_20_50_diff":       "SMA_20_50_Diff",
    "sma20_slope":          "SMA_20_Slope",
    "ema20_slope":          "EMA_20_Slope",

    # ── MACD ───────────────────────────────────────────────────────────────
    # MACD_ROC and the fast-MACD trio (MACD_Fast/MACDh_Fast/MACDs_Fast)
    # pruned (2026-08-06): never computed in intraday_data_collector.py —
    # no 5/13/1 MACD instance exists there, and macd_roc is only computed
    # live in ml_screen_and_predict.py (a different pipeline), not stored.
    "macd.macd":            "MACD_12_26_9",
    "macd_diff":            "MACDh_12_26_9",
    "macd.signal":          "MACDs_12_26_9",

    # ── RSI ────────────────────────────────────────────────────────────────
    # "rsi" is the canonical source; "rsi14" and "rsi[1]" are aliases.
    # rename_t1_columns keeps whichever appears first in the DataFrame.
    # RSI_7/21/28 and RSI_14_Slope pruned (2026-08-06): only RSI_14 is
    # actually computed here.
    "rsi":              "RSI_14",
    "rsi14":            "RSI_14",       # alias — deduped at runtime
    "rsi[1]":           "RSI_14",       # shifted bar — deduped at runtime

    # ── STOCHASTIC ─────────────────────────────────────────────────────────
    # Short forms ("stoch.k/d") are canonical; long forms are aliases.
    # STOCHh_14_3_3 (histogram never derived) and the entire 5/3/1 variant
    # (STOCHk/d/h_5_3_1) pruned (2026-08-06): not computed here.
    "stoch.k":              "STOCHk_14_3_3",
    "stochk_14_3_3":        "STOCHk_14_3_3",   # alias — deduped at runtime
    "stoch.d":              "STOCHd_14_3_3",
    "stochd_14_3_3":        "STOCHd_14_3_3",   # alias — deduped at runtime

    # ── STOCH RSI ──────────────────────────────────────────────────────────
    # Pruned entirely (2026-08-06): no StochRSI import/computation exists in
    # intraday_data_collector.py.

    # ── OSCILLATORS ────────────────────────────────────────────────────────
    # CCI_14 pruned (2026-08-06): only CCI_20 is computed here.
    "w.r":              "WILLR_14",
    "cci20":            "CCI_20",
    "uo":               "UO",
    "ao":               "AO",
    "tsi":              "TSI_13_25_13",
    # NOTE: "kst" is intentionally NOT mapped here. intraday_data_collector.py
    # computes a genuine KST (Know Sure Thing) indicator, which is a
    # structurally different calculation from TSI (True Strength Index) —
    # they are not interchangeable. Previously "kst" was mapped to
    # "TSIs_13_25_13" as a "closest equivalent," which meant the trained
    # feature TSIs_13_25_13 held real TSI-signal values for base-sourced rows
    # but real KST values for T-1-sourced rows: two different indicators
    # sharing one feature name. Per this module's own convention, source
    # columns with no mapping entry are safely dropped by rename_t1_columns,
    # so kst/kst_signal are simply excluded rather than mislabeled. If KST is
    # wanted as a feature, it should be added as its own named feature to
    # BOTH multiday_feature_collector.py (base training) and here — not
    # aliased onto an existing TSI feature.

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
    # ATR_7/ATR_20 pruned (2026-08-06): only the 14-period ATR is computed.
    "atr":              "ATR_14",
    "atr14":            "ATR_14",       # alias — deduped at runtime
    "atr_pct":          "ATR_14_Slope",   # no exact match — closest
    "volatility_10d":   "HV_10",
    "volatility_20d":   "HV_20",
    "volatility_30d":   "HV_30",

    # ── VOLUME ─────────────────────────────────────────────────────────────
    # OBV_SMA20 pruned (2026-08-06): never computed here.
    "volume_sma5":      "Volume_MA5",
    "volume_sma10":     "Volume_MA10",
    "volume_sma20":     "Volume_MA20",
    "volume_ratio":     "Volume_Ratio",
    "obv":              "OBV",

    # ── TREND / ADX ────────────────────────────────────────────────────────
    # "adx+di" / "adx-di" are canonical; "_pos" / "_neg" are aliases.
    # ADXR_14_2 pruned (2026-08-06): never computed here.
    "adx":              "ADX_14",
    "adx+di":           "DMP_14",
    "adx_pos":          "DMP_14",       # alias — deduped at runtime
    "adx-di":           "DMN_14",
    "adx_neg":          "DMN_14",       # alias — deduped at runtime

    # ── AROON ──────────────────────────────────────────────────────────────
    "aroon_down":       "AROOND_25",
    "aroon_up":         "AROONU_25",
    "aroon_indicator":  "AROONOSC_25",

    # ── SUPERTREND ─────────────────────────────────────────────────────────
    # Pruned entirely (2026-08-06): no Supertrend computation exists in
    # intraday_data_collector.py.

    # ── MISC ───────────────────────────────────────────────────────────────
    # "roc" / "roc10" and "mom" / "mom10" and "gap_%" / "gap_pct" are aliases.
    # MFI_14, ROC_20, MOM_20 pruned (2026-08-06): never computed here — only
    # the 10-period ROC/MOM exist (see the separate ROC window=12-vs-10 fix
    # elsewhere; that's a different, already-addressed bug).
    "vwap":             "VWAP",
    "cmf":              "CMF_20",
    "roc":              "ROC_10",
    "roc10":            "ROC_10",       # alias — deduped at runtime
    "mom":              "MOM_10",
    "mom10":            "MOM_10",       # alias — deduped at runtime
    "gap_%":            "Gap_Pct",
    "gap_pct":          "Gap_Pct",      # alias — deduped at runtime
}

# ---------------------------------------------------------------------------
# Intentionally-unmapped T-1 source columns
# ---------------------------------------------------------------------------
# intraday_data_collector.py also computes kama, vortex_pos, vortex_neg,
# mass_index, dpo, psar, psar_up, and psar_down for every T-1 snapshot (some
# even pass through its normalization loop). None of these have an entry
# above.
#
# This is intentional, not an oversight: none of these indicators exist in
# the base/multiday feature set either (checked against the t3_/t5_/t10_
# column names produced by multiday_feature_collector.py — no kama, vortex,
# mass_index, dpo, or psar variant appears there). Per this module's own
# convention (see the "kst" note above), a source column with no live
# counterpart in the model's feature vocabulary is safely dropped by
# rename_t1_columns() rather than mapped to a same-window-but-wrong-indicator
# stand-in or a dead target no other row ever populates (see t1_column_map.py
# bug fix 2026-08-06: "cci" → "CCI_14" was exactly this mistake).
#
# If any of these are wanted as real model features, they need to be added
# to BOTH multiday_feature_collector.py (base training) and here — not
# mapped one-sided from T-1 alone.

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
