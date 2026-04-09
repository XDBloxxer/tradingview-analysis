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
    "SMA_5": "sma_5", "SMA_10": "sma_10", "SMA_20": "sma_20",
    "SMA_50": "sma_50", "SMA_100": "sma_100", "SMA_200": "sma_200",
    "EMA_5": "ema_5", "EMA_10": "ema_10", "EMA_12": "ema_12",
    "EMA_20": "ema_20", "EMA_26": "ema_26", "EMA_50": "ema_50",
    "EMA_100": "ema_100", "EMA_200": "ema_200",
    # MACD (pandas_ta native names)
    "MACD_12_26_9": "macd", "MACDh_12_26_9": "macdh", "MACDs_12_26_9": "macds",
    "MACD_6_13_5": "macd_fast", "MACDh_6_13_5": "macdh_fast", "MACDs_6_13_5": "macds_fast",
    # Stochastic
    "STOCHk_14_3_3": "stochk_14_3_3", "STOCHd_14_3_3": "stochd_14_3_3",
    "STOCHh_14_3_3": "stochh_14_3_3",
    # Bollinger Bands
    "BBL_20_2.0_2.0": "bbl_20_2_0", "BBM_20_2.0_2.0": "bbm_20_2_0",
    "BBU_20_2.0_2.0": "bbu_20_2_0", "BBB_20_2.0_2.0": "bbb_20_2_0",
    "BBP_20_2.0_2.0": "bbp_20_2_0",
    # ATR / volatility
    "ATRr_14": "atr_14", "ATRr_7": "atr_7", "ATRr_20": "atr_20",
    # ADX
    "ADX_14": "adx_14", "DMP_14": "dmp_14", "DMN_14": "dmn_14",
    # CCI
    "CCI_14_0.015": "cci_14",
    # Williams %R
    "WILLR_14": "willr_14",
    # Momentum / ROC
    "MOM_10": "mom_10", "ROC_10": "roc_10",
    # OBV / volume
    "OBV": "obv",
    # Supertrend
    "SUPERT_10_3.0": "supert_10_3", "SUPERTd_10_3.0": "supertd_10_3",
    "SUPERTs_10_3.0": "superts_10_3", "SUPERTl_10_3.0": "supertl_10_3",
    # Stoch RSI
    "STOCHRSIk_14_14_3_3": "stochrsik_14_14_3_3",
    "STOCHRSId_14_14_3_3": "stochrsid_14_14_3_3",
    # VWAP
    "VWAP_D": "vwap",
    # Price-change helpers (computed manually, see below)
    "price_change_3d": "price_change_3d",
    "price_change_5d": "price_change_5d",
    "price_change_10d": "price_change_10d",
    "gap_pct": "gap_pct",
    "volume_ratio": "volume_ratio",
    "high_52w_pct": "high_52w_pct",
    "low_52w_pct": "low_52w_pct",
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
    "t10_close", "t10_cmf_20", "t10_dcl_20_20", "t10_dcm_20_20", "t10_dcu_20_20",
    "t10_dmn_14", "t10_dmp_14", "t10_ema_10", "t10_ema_12", "t10_ema_12_26_diff",
    "t10_ema_20", "t10_ema_20_slope", "t10_ema_26", "t10_ema_5", "t10_ema_50",
    "t10_gap_pct", "t10_high", "t10_hma_20", "t10_hma_9", "t10_hv_10", "t10_hv_20",
    "t10_hv_30", "t10_kcbe_20_2", "t10_kcle_20_2", "t10_kcue_20_2", "t10_low",
    "t10_macd_12_26_9", "t10_macd_fast", "t10_macd_roc", "t10_macdh_12_26_9",
    "t10_macdh_fast", "t10_macds_12_26_9", "t10_macds_fast", "t10_mfi_14",
    "t10_mom_10", "t10_mom_20", "t10_obv", "t10_obv_sma20", "t10_open",
    "t10_price_vs_ema20", "t10_price_vs_sma20", "t10_price_vs_sma50",
    "t10_roc_10", "t10_roc_20", "t10_rsi_14", "t10_rsi_14_slope", "t10_rsi_21",
    "t10_rsi_28", "t10_rsi_7", "t10_sma_10", "t10_sma_20", "t10_sma_20_50_diff",
    "t10_sma_20_slope", "t10_sma_5", "t10_sma_50", "t10_stochd_14_3_3",
    "t10_stochd_5_3_1", "t10_stochh_14_3_3", "t10_stochh_5_3_1",
    "t10_stochk_14_3_3", "t10_stochk_5_3_1", "t10_stochrsid_14_14_3_3",
    "t10_stochrsik_14_14_3_3", "t10_supert_10_3", "t10_supertd_10_3",
    "t10_supertl_10_3", "t10_superts_10_3", "t10_tsi_13_25_13", "t10_tsis_13_25_13",
    "t10_uo", "t10_volume", "t10_volume_ma10", "t10_volume_ma20", "t10_volume_ma5",
    "t10_volume_ratio", "t10_vwap", "t10_vwma_20", "t10_willr_14", "t10_wma_10",
    "t10_wma_20",
    # ── t3 ───────────────────────────────────────────────────────────────────
    "t3_adx_14", "t3_adxr_14_2", "t3_ao", "t3_aroond_25", "t3_aroonosc_25",
    "t3_aroonu_25", "t3_atr_14", "t3_atr_14_slope", "t3_atr_20", "t3_atr_7",
    "t3_bbb_20_2_0_2_0", "t3_bbl_20_2_0_2_0", "t3_bbm_20_2_0_2_0",
    "t3_bbp_20_2_0_2_0", "t3_bbu_20_2_0_2_0", "t3_cci_14", "t3_cci_20",
    "t3_close", "t3_cmf_20", "t3_dcl_20_20", "t3_dcm_20_20", "t3_dcu_20_20",
    "t3_dmn_14", "t3_dmp_14", "t3_ema_10", "t3_ema_12", "t3_ema_12_26_diff",
    "t3_ema_20", "t3_ema_20_slope", "t3_ema_26", "t3_ema_5", "t3_ema_50",
    "t3_gap_pct", "t3_high", "t3_hma_20", "t3_hma_9", "t3_hv_10", "t3_hv_20",
    "t3_hv_30", "t3_kcbe_20_2", "t3_kcle_20_2", "t3_kcue_20_2", "t3_low",
    "t3_macd_12_26_9", "t3_macd_fast", "t3_macd_roc", "t3_macdh_12_26_9",
    "t3_macdh_fast", "t3_macds_12_26_9", "t3_macds_fast", "t3_mfi_14",
    "t3_mom_10", "t3_mom_20", "t3_obv", "t3_obv_sma20", "t3_open",
    "t3_price_vs_ema20", "t3_price_vs_sma20", "t3_price_vs_sma50",
    "t3_roc_10", "t3_roc_20", "t3_rsi_14", "t3_rsi_14_slope", "t3_rsi_21",
    "t3_rsi_28", "t3_rsi_7", "t3_sma_10", "t3_sma_20", "t3_sma_20_50_diff",
    "t3_sma_20_slope", "t3_sma_5", "t3_sma_50", "t3_stochd_14_3_3",
    "t3_stochd_5_3_1", "t3_stochh_14_3_3", "t3_stochh_5_3_1",
    "t3_stochk_14_3_3", "t3_stochk_5_3_1", "t3_stochrsid_14_14_3_3",
    "t3_stochrsik_14_14_3_3", "t3_supert_10_3", "t3_supertd_10_3",
    "t3_supertl_10_3", "t3_superts_10_3", "t3_tsi_13_25_13", "t3_tsis_13_25_13",
    "t3_uo", "t3_volume", "t3_volume_ma10", "t3_volume_ma20", "t3_volume_ma5",
    "t3_volume_ratio", "t3_vwap", "t3_vwma_20", "t3_willr_14", "t3_wma_10",
    "t3_wma_20",
    # ── t5 ───────────────────────────────────────────────────────────────────
    "t5_adx_14", "t5_adxr_14_2", "t5_ao", "t5_aroond_25", "t5_aroonosc_25",
    "t5_aroonu_25", "t5_atr_14", "t5_atr_14_slope", "t5_atr_20", "t5_atr_7",
    "t5_bbb_20_2_0_2_0", "t5_bbl_20_2_0_2_0", "t5_bbm_20_2_0_2_0",
    "t5_bbp_20_2_0_2_0", "t5_bbu_20_2_0_2_0", "t5_cci_14", "t5_cci_20",
    "t5_close", "t5_cmf_20", "t5_dcl_20_20", "t5_dcm_20_20", "t5_dcu_20_20",
    "t5_dmn_14", "t5_dmp_14", "t5_ema_10", "t5_ema_12", "t5_ema_12_26_diff",
    "t5_ema_20", "t5_ema_20_slope", "t5_ema_26", "t5_ema_5", "t5_ema_50",
    "t5_gap_pct", "t5_high", "t5_hma_20", "t5_hma_9", "t5_hv_10", "t5_hv_20",
    "t5_hv_30", "t5_kcbe_20_2", "t5_kcle_20_2", "t5_kcue_20_2", "t5_low",
    "t5_macd_12_26_9", "t5_macd_fast", "t5_macd_roc", "t5_macdh_12_26_9",
    "t5_macdh_fast", "t5_macds_12_26_9", "t5_macds_fast", "t5_mfi_14",
    "t5_mom_10", "t5_mom_20", "t5_obv", "t5_obv_sma20", "t5_open",
    "t5_price_vs_ema20", "t5_price_vs_sma20", "t5_price_vs_sma50",
    "t5_roc_10", "t5_roc_20", "t5_rsi_14", "t5_rsi_14_slope", "t5_rsi_21",
    "t5_rsi_28", "t5_rsi_7", "t5_sma_10", "t5_sma_20", "t5_sma_20_50_diff",
    "t5_sma_20_slope", "t5_sma_5", "t5_sma_50", "t5_stochd_14_3_3",
    "t5_stochd_5_3_1", "t5_stochh_14_3_3", "t5_stochh_5_3_1",
    "t5_stochk_14_3_3", "t5_stochk_5_3_1", "t5_stochrsid_14_14_3_3",
    "t5_stochrsik_14_14_3_3", "t5_supert_10_3", "t5_supertd_10_3",
    "t5_supertl_10_3", "t5_superts_10_3", "t5_tsi_13_25_13", "t5_tsis_13_25_13",
    "t5_uo", "t5_volume", "t5_volume_ma10", "t5_volume_ma20", "t5_volume_ma5",
    "t5_volume_ratio", "t5_vwap", "t5_vwma_20", "t5_willr_14", "t5_wma_10",
    "t5_wma_20",
}

MAX_WORKERS = 5
LOOKBACK_BARS = 260   # ~1 year of daily bars — enough for SMA_200


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

    # ── Run pandas_ta strategy ──────────────────────────────────────────────
    try:
        df.ta.strategy(
            ta.Strategy(
                "multiday",
                [
                    {"kind": "rsi",       "length": 14},
                    {"kind": "rsi",       "length": 7},
                    {"kind": "rsi",       "length": 21},
                    {"kind": "rsi",       "length": 28},
                    {"kind": "sma",       "length": 5},
                    {"kind": "sma",       "length": 10},
                    {"kind": "sma",       "length": 20},
                    {"kind": "sma",       "length": 50},
                    {"kind": "sma",       "length": 100},
                    {"kind": "sma",       "length": 200},
                    {"kind": "ema",       "length": 5},
                    {"kind": "ema",       "length": 10},
                    {"kind": "ema",       "length": 12},
                    {"kind": "ema",       "length": 20},
                    {"kind": "ema",       "length": 26},
                    {"kind": "ema",       "length": 50},
                    {"kind": "ema",       "length": 100},
                    {"kind": "ema",       "length": 200},
                    {"kind": "macd",      "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "macd",      "fast": 6,  "slow": 13, "signal": 5},
                    {"kind": "stoch",     "k": 14, "d": 3, "smooth_k": 3},
                    {"kind": "bbands",    "length": 20, "std": 2.0},
                    {"kind": "atr",       "length": 14},
                    {"kind": "atr",       "length": 7},
                    {"kind": "atr",       "length": 20},
                    {"kind": "adx",       "length": 14},
                    {"kind": "cci",       "length": 14},
                    {"kind": "willr",     "length": 14},
                    {"kind": "mom",       "length": 10},
                    {"kind": "roc",       "length": 10},
                    {"kind": "obv"},
                    {"kind": "supertrend","length": 10, "multiplier": 3.0},
                    {"kind": "stochrsi",  "length": 14, "rsi_length": 14,
                     "k": 3, "d": 3},
                    {"kind": "vwap"},
                ]
            ),
            verbose=False,
        )
    except Exception as exc:
        logger.debug(f"pandas_ta strategy error: {exc}")

    # ── Price-change helpers ─────────────────────────────────────────────────
    close = df["close"]
    volume = df["volume"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]

    for n, col in [(3, "price_change_3d"), (5, "price_change_5d"),
                   (10, "price_change_10d")]:
        if len(close) > n:
            df[col] = (close - close.shift(n)) / close.shift(n) * 100
        else:
            df[col] = np.nan

    if len(df) > 1:
        df["gap_pct"] = (open_ - close.shift(1)) / close.shift(1) * 100
    else:
        df["gap_pct"] = np.nan

    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    if len(close) >= 252:
        df["high_52w_pct"] = (close - high.rolling(252).max()) / high.rolling(252).max() * 100
        df["low_52w_pct"]  = (close - low.rolling(252).min())  / low.rolling(252).min()  * 100
    else:
        df["high_52w_pct"] = np.nan
        df["low_52w_pct"]  = np.nan

    # ── Extract last row and map column names ───────────────────────────────
    last = df.iloc[-1]
    result: Dict[str, Any] = {}

    for raw_col, base_name in PANDAS_TA_TO_BASE.items():
        if raw_col in df.columns:
            val = last.get(raw_col, np.nan)
            result[base_name] = None if (pd.isna(val) or np.isinf(val)) else float(val)

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

        row: Dict[str, Any] = {
            "symbol":         symbol,
            "detection_date": detection_date.date().isoformat(),
        }

        for prefix, offset in TIMEFRAMES.items():
            snap = _snapshot_for_offset(raw, detection_date, offset)
            for base_name, val in snap.items():
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

        # Supabase upsert in chunks of 500 to stay under request-size limits
        written = 0
        chunk_size = 500
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            try:
                resp = self.client.table(table).upsert(
                    chunk,
                    on_conflict="symbol,detection_date",
                ).execute()
                written += len(resp.data or chunk)
            except Exception as exc:
                self.logger.error(
                    f"MultidayFeatureCollector: upsert error for chunk "
                    f"{i}–{i+len(chunk)}: {exc}"
                )

        self.logger.info(
            f"MultidayFeatureCollector: wrote {written} row(s) to {table}."
        )
        return written
