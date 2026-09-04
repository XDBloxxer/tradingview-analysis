#!/usr/bin/env python3
"""
Comprehensive ML Accuracy Tracker with MODEL-DRIVEN LEARNING

FIXES vs previous version:
  1. Case mismatch in compute_model_driven_filters: SCREENER_FEATURE_MAP keys
     were mixed-case ('HV_10', 'RSI_14') but top_features from
     load_feature_importance() are lowercase after prefix stripping ('hv_10').
     The `if base_feat not in top_features` check therefore ALWAYS skipped
     every entry → HV/RSI/ATR/ADX filters were never derived.
     Fix: normalize both sides to lowercase before comparison.

  2. analyze_missed_opportunities used wrong actual_high_pct denominator:
     (winner_row['high'] / winner_row['price'] - 1) uses same-day price,
     giving near-0% for all stocks. Fixed to use yfinance prev_close like
     analyze_prediction_accuracy does.

  3. HV filter over-aggressiveness: min_hv10/min_hv20 were derived from
     winner p10 percentiles only, producing values like 56%+ that excluded
     ~98% of the market at screen time. Added HARD_CAPS["max_min_hv10/20"]
     and per-filter clamping in compute_model_driven_filters so the screener
     always passes enough stocks for the ML model to rank.

  4. (NEW) KeyError 'symbol' crash when winners_df is empty for the check
     date (e.g. a date with predictions but no recorded winners yet).
     analyze_prediction_accuracy now guards winners_set construction with an
     empty-DataFrame check so the script completes and writes accuracy records
     even when winners data is absent.

  5. (NEW) Signal inversion fix: predicted_positive was derived from the raw
     XGBoost binary (prediction == 1, i.e. probability > 0.5).  After
     scale_pos_weight was raised to up to 10x, virtually every post-screener
     stock scores above 0.5, making prediction=1 for almost everyone and
     rendering the binary useless as a BUY/AVOID discriminator.  Meanwhile
     the published signal is percentile-ranked within the batch, so a stock
     can be prediction=1 (raw above 0.5) yet signal=AVOID (bottom 20% of
     the batch).  This caused AVOID stocks that won big (e.g. VCIG +118%)
     to be counted as true negatives (prediction_correct=True) in the
     accuracy tracker, silently inflating reported accuracy.
     Fix: predicted_positive is now True only when signal ∈ {BUY, STRONG BUY},
     which matches exactly what the published recommendation said.

  6. (NEW 2026-09) ml_accuracy_details / ml_missed_opportunities writes
     bypassed the NaN/Infinity sanitizer that write_accuracy_records uses,
     calling self.client.table(...).upsert(...) directly. A single NaN or
     Infinity float anywhere in the batch (e.g. one symbol yfinance
     couldn't resolve) made postgrest's JSON encoder reject the WHOLE
     batch upsert with "Out of range float values are not JSON compliant"
     — so all rows for the run silently failed to write, not just the bad
     one. Fixed by routing every write through the new sanitized
     write_records_upsert() helper on MLPredictionSupabaseClient.

  7. (NEW 2026-09) Added pre-market tracking. The yfinance actual-gain
     fetch previously only pulled interval="1d" regular-session bars,
     which cannot reflect pre-market moves (Yahoo's daily OHLC is
     regular-session only, 09:30-16:00 ET). Added a second intraday
     (5m, prepost=True) fetch that isolates the 04:00-09:30 ET window
     and computes actual_premarket_high_pct / actual_premarket_low_pct
     relative to the same prev_close used for the regular-session figures,
     so premarket spikes that reverse before the open are captured.

  8. (NEW 2026-09) Backfill hardening: --date already existed for
     re-running a specific past date, but a partial/failed prior run
     (e.g. hit by bug #6) could leave ml_accuracy_details/
     ml_missed_opportunities empty for that date. Since every table
     write now upserts on the natural key (symbol, prediction_date) /
     (symbol, detection_date), simply re-running with --date on the same
     date now fully repairs prior partial writes instead of skipping them.
     Also added --backfill-yesterday as a convenience shortcut.

IMPROVEMENTS (carried from previous version):
1. Finds most recent prediction date automatically
2. Validates data exists before fetching
3. Early exit if no data
4. MODEL-DRIVEN filter learning from feature_importance.csv + winner stats
5. Fetches actual_gain_pct via yfinance for ALL predicted symbols
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import yaml
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - stdlib zoneinfo should exist on py3.9+
    _ET = None

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_IMPORTANCE_PATH = Path("ml_models/feature_importance.csv")
LEARNED_FILTERS_PATH    = Path("ml_models/learned_filters.json")

LOWER_PCT = 10
UPPER_PCT = 90

FILTER_LOOKBACK_DAYS = 90
MIN_SAMPLES_FOR_FILTER = 20
YFINANCE_MAX_WORKERS = 10

# A stock is considered a winner if its intraday high on prediction date
# reached this threshold above the prior close — regardless of where it closed.
# CLAUDE FIX (2026-08-19): kept in sync with ml_retrain_model.py's
# INTRADAY_WIN_THRESHOLD (20.0 -> 15.0). These two constants must match —
# otherwise this tracker would grade the model against a "winner" bar
# different from what it was actually trained to hit, silently mis-scoring
# performance again in the other direction.
INTRADAY_WIN_THRESHOLD = 15.0

# FIX 1: Keys are now lowercase to match what load_feature_importance() produces
# after stripping the t3_/t5_/t10_ prefix from lowercase model feature names.
SCREENER_FEATURE_MAP = {
    "hv_10":        ("min_hv10",        "max_hv10"),
    "hv_20":        ("min_hv20",        "max_hv20"),
    "volume_ratio": ("min_volume_ratio", None),
    "rsi_14":       ("min_rsi",         "max_rsi"),
    "rsi_7":        ("min_rsi7",        "max_rsi7"),
    "atr_14":       ("min_atr14",       None),
    "adx_14":       ("min_adx",         None),
}

# FIX 3: Hard caps prevent any single filter from excluding the bulk of the market.
HARD_CAPS = {
    "min_price":           0.50,
    "max_price":           50.0,
    # FIX 6: max_price previously had a ceiling (50.0) but no floor. Because
    # daily_winners_detector.py feeds this same learned max_price BACK IN as
    # the price ceiling for tomorrow's winner-candidate screen (see its
    # _load_learned_filters), a narrow/unlucky winner sample can pull
    # max_price down to a few dollars — which then makes it structurally
    # impossible for the screener to ever find a winner above that price
    # again, ratcheting max_price down further on each subsequent run with
    # no way to recover. min_max_price stops that ratchet: no matter how low
    # the derived p90*1.2 comes out, max_price is never clamped below this
    # floor, so the screener always keeps looking above it.
    "min_max_price":       15.0,
    "min_volume":          100_000,
    "min_relative_volume": 1.0,
    "max_min_hv10":        30.0,
    "max_min_hv20":        30.0,
    "max_min_volume_ratio": 3.0,
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# yfinance actual-gain fetcher
# ---------------------------------------------------------------------------

def _fetch_premarket_extremes(symbol: str, prediction_date: str, prev_close: Optional[float]) -> dict:
    """
    FIX 7: regular-session daily bars (interval="1d") cannot reflect
    pre-market activity — Yahoo's daily OHLC only covers the 09:30-16:00 ET
    regular session regardless of the `prepost` flag. To capture pre-market
    moves we need a separate intraday (5m) fetch with prepost=True, then
    isolate bars whose Eastern-time timestamp falls in the 04:00-09:30 ET
    pre-market window on the prediction date.

    Returns {} (empty extras) on any failure — this is best-effort
    enrichment and must never break the regular-session accuracy numbers.
    """
    empty = {
        "actual_premarket_high_pct": None,
        "actual_premarket_low_pct":  None,
        "actual_premarket_volume":   None,
    }
    if _ET is None or prev_close is None or prev_close == 0:
        return empty
    try:
        import yfinance as yf

        target = datetime.strptime(prediction_date, "%Y-%m-%d").date()
        # yfinance only serves ~30 days of intraday history at 5m resolution,
        # so a wide window costs nothing extra and keeps this robust for
        # near-term backfills; older dates will simply return no data.
        start = (target - timedelta(days=1)).isoformat()
        end   = (target + timedelta(days=1)).isoformat()

        ticker   = yf.Ticker(symbol)
        intraday = ticker.history(
            start=start, end=end, interval="5m", prepost=True, auto_adjust=True
        )
        if intraday.empty:
            return empty

        idx = intraday.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx_et = idx.tz_convert(_ET)

        et_dates = idx_et.date
        et_times = idx_et.time
        premarket_mask = (et_dates == target) & (et_times >= dt_time(4, 0)) & (et_times < dt_time(9, 30))

        pm = intraday.loc[premarket_mask]
        if pm.empty:
            return empty

        pm_high   = float(pm["High"].max())
        pm_low    = float(pm["Low"].min())
        pm_volume = int(pm["Volume"].sum()) if "Volume" in pm.columns else None

        return {
            "actual_premarket_high_pct": round((pm_high - prev_close) / prev_close * 100, 4),
            "actual_premarket_low_pct":  round((pm_low  - prev_close) / prev_close * 100, 4),
            "actual_premarket_volume":   pm_volume,
        }
    except Exception as e:
        logging.getLogger(__name__).debug(
            f"premarket fetch failed for {symbol} on {prediction_date}: {e}"
        )
        return empty


def _fetch_actual_gain_for_symbol(
    symbol: str, prediction_date: str, include_premarket: bool = True
) -> dict:
    try:
        import yfinance as yf

        target = datetime.strptime(prediction_date, "%Y-%m-%d").date()
        start  = (target - timedelta(days=5)).isoformat()
        end    = (target + timedelta(days=2)).isoformat()

        ticker = yf.Ticker(symbol)
        hist   = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)

        if hist.empty or len(hist) < 2:
            return {"symbol": symbol}

        hist.index = pd.to_datetime(hist.index).date

        if target not in hist.index:
            return {"symbol": symbol}

        target_idx = list(hist.index).index(target)
        if target_idx == 0:
            return {"symbol": symbol}

        today_bar  = hist.iloc[target_idx]
        prev_close = float(hist.iloc[target_idx - 1]["Close"])

        if prev_close == 0:
            return {"symbol": symbol}

        close      = float(today_bar["Close"])
        day_high   = float(today_bar["High"])
        day_low    = float(today_bar["Low"])
        day_open   = float(today_bar["Open"])
        day_volume = int(today_bar["Volume"])

        result = {
            "symbol":          symbol,
            "actual_gain_pct": round((close    - prev_close) / prev_close * 100, 4),
            "actual_high_pct": round((day_high - prev_close) / prev_close * 100, 4),
            "actual_close":    close,
            "actual_open":     day_open,
            "actual_high":     day_high,
            "actual_low":      day_low,
            "actual_volume":   day_volume,
            "actual_premarket_high_pct": None,
            "actual_premarket_low_pct":  None,
            "actual_premarket_volume":   None,
        }

        if include_premarket:
            result.update(_fetch_premarket_extremes(symbol, prediction_date, prev_close))

        return result

    except Exception as e:
        logging.getLogger(__name__).debug(
            f"yfinance fetch failed for {symbol} on {prediction_date}: {e}"
        )
        return {"symbol": symbol}


def fetch_actual_gains_for_all_symbols(
    symbols: list,
    prediction_date: str,
    logger: logging.Logger,
    max_workers: int = YFINANCE_MAX_WORKERS,
    include_premarket: bool = True,
) -> dict:
    logger.info(
        f"Fetching actual gain data from yfinance for {len(symbols)} symbols "
        f"on {prediction_date} (max_workers={max_workers}, premarket={include_premarket})..."
    )

    results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(
                _fetch_actual_gain_for_symbol, sym, prediction_date, include_premarket
            ): sym
            for sym in symbols
        }
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                results[sym] = future.result()
            except Exception as e:
                logger.warning(f"Unexpected error fetching {sym}: {e}")
                results[sym] = {"symbol": sym}

    found = sum(1 for r in results.values() if r.get("actual_gain_pct") is not None)
    pm_found = sum(1 for r in results.values() if r.get("actual_premarket_high_pct") is not None)
    logger.info(f"✓ yfinance fetch complete: {found}/{len(symbols)} symbols returned gain data")
    if include_premarket:
        logger.info(f"✓ premarket fetch complete: {pm_found}/{len(symbols)} symbols returned premarket data")
    return results


# ---------------------------------------------------------------------------
# Helpers — data validation
# ---------------------------------------------------------------------------

def get_most_recent_prediction_date(tracker) -> Optional[str]:
    try:
        response = (
            tracker.client.table("ml_explosion_predictions")
            .select("prediction_date")
            .order("prediction_date", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return response.data[0]["prediction_date"]
    except Exception as e:
        tracker.logger.error(f"Error finding most recent prediction date: {e}")
        return None


def validate_data_exists(tracker, check_date: str) -> dict:
    result = {
        "predictions_exist": False,
        "winners_exist":     False,
        "should_proceed":    False,
        "prediction_count":  0,
        "winner_count":      0,
    }
    try:
        pred_response = (
            tracker.client.table("ml_explosion_predictions")
            .select("*", count="exact")
            .eq("prediction_date", check_date)
            .limit(1)
            .execute()
        )
        result["prediction_count"] = pred_response.count or 0
        result["predictions_exist"] = result["prediction_count"] > 0

        if not result["predictions_exist"]:
            tracker.logger.warning(f"⚠️ No predictions found for {check_date}")
            return result

        tracker.logger.info(f"✓ Found {result['prediction_count']} predictions for {check_date}")

        winner_response = (
            tracker.client.table("daily_winners")
            .select("*", count="exact")
            .eq("detection_date", check_date)
            .limit(1)
            .execute()
        )
        result["winner_count"] = winner_response.count or 0
        result["winners_exist"] = result["winner_count"] > 0

        if not result["winners_exist"]:
            tracker.logger.warning(f"⚠️ No winners found for {check_date}")
            result["should_proceed"] = True
            return result

        tracker.logger.info(f"✓ Found {result['winner_count']} winners for {check_date}")
        result["should_proceed"] = True
        return result

    except Exception as e:
        tracker.logger.error(f"Error validating data: {e}")
        return result


# ---------------------------------------------------------------------------
# Model-driven filter learning
# ---------------------------------------------------------------------------

def load_feature_importance(path: Path, top_n: int = 30) -> list:
    """
    Returns lowercase base feature names (prefix stripped) sorted by importance.
    Lowercase ensures consistent comparison with SCREENER_FEATURE_MAP keys.
    """
    if not path.exists():
        return []
    try:
        fi = pd.read_csv(path)
        fi = fi[fi["importance"] > 0].sort_values("importance", ascending=False)

        base_names = []
        seen = set()
        for feat in fi["feature"]:
            for prefix in ("t3_", "t5_", "t10_", "t1_close_", "t1_open_"):
                if feat.startswith(prefix):
                    feat = feat[len(prefix):]
                    break
            if feat.startswith("t1_"):
                continue
            feat_lower = feat.lower()
            if feat_lower not in seen:
                seen.add(feat_lower)
                base_names.append(feat_lower)
            if len(base_names) >= top_n:
                break
        return base_names
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load feature importance: {e}")
        return []


def fetch_winner_t1_snapshots(client, lookback_days: int) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    start_date = (datetime.now().date() - timedelta(days=lookback_days)).isoformat()

    rows = []
    offset = 0
    page_size = 1000
    while True:
        try:
            resp = (
                client.table("winners_day_prior_close")
                .select("*")
                .gte("detection_date", start_date)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        except Exception as e:
            logger.warning(f"Error fetching winner snapshots: {e}")
            break

    if not rows:
        logger.warning("No winner T-1 snapshots found for filter learning.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"Fetched {len(df)} winner T-1 snapshots for filter computation.")
    return df


def fetch_non_winner_t1_snapshots(client, lookback_days: int) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    start_date = (datetime.now().date() - timedelta(days=lookback_days)).isoformat()

    rows = []
    offset = 0
    page_size = 1000
    while True:
        try:
            resp = (
                client.table("non_winners_day_prior_close")
                .select("*")
                .gte("detection_date", start_date)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        except Exception as e:
            logger.warning(f"Error fetching non-winner snapshots: {e}")
            break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _col_variants(base: str) -> list:
    """
    Return candidate column names for a base feature name.
    base is already lowercase (normalized by load_feature_importance).
    """
    variants = [base, base.replace("_", "")]
    special = {
        "hv_10":        ["volatility_10d", "hv_10", "hv10"],
        "hv_20":        ["volatility_20d", "hv_20", "hv20"],
        "hv_30":        ["volatility_30d", "hv_30", "hv30"],
        "volume_ratio": ["volume_ratio"],
        "rsi_14":       ["rsi", "rsi14"],
        "rsi_7":        ["rsi7"],
        "atr_14":       ["atr", "atr14"],
        "adx_14":       ["adx"],
        "obv":          ["obv"],
        "close":        ["close"],
        "volume":       ["volume"],
    }
    if base in special:
        variants = special[base] + variants
    return variants


def find_col(df: pd.DataFrame, base: str) -> Optional[str]:
    """Find the actual column in df matching base (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for v in _col_variants(base.lower()):
        if v in df_cols_lower:
            return df_cols_lower[v]
    return None


def compute_model_driven_filters(
    winner_df: pd.DataFrame,
    non_winner_df: pd.DataFrame,
    top_features: list,
    logger: logging.Logger,
) -> dict:
    filters = {}

    if winner_df.empty or len(winner_df) < MIN_SAMPLES_FOR_FILTER:
        logger.warning(
            f"Only {len(winner_df)} winner samples — need {MIN_SAMPLES_FOR_FILTER}. "
            "Using conservative defaults."
        )
        return _conservative_defaults()

    top_features_set = set(top_features)

    # ── Price range ──────────────────────────────────────────────────────────
    price_col = find_col(winner_df, "close")
    if price_col:
        prices = pd.to_numeric(winner_df[price_col], errors="coerce").dropna()
        if len(prices) >= MIN_SAMPLES_FOR_FILTER:
            p10 = float(prices.quantile(LOWER_PCT / 100))
            p90 = float(prices.quantile(UPPER_PCT / 100))
            filters["min_price"] = max(HARD_CAPS["min_price"], round(p10 * 0.8, 2))
            # FIX 6: floor max_price at HARD_CAPS["min_max_price"] so a
            # narrow/low-priced winner sample can't ratchet the ceiling down
            # below a sane floor (see HARD_CAPS comment — this value feeds
            # back into tomorrow's winner screen, so under-flooring here
            # permanently blinds the screener to higher-priced winners).
            filters["max_price"] = min(
                HARD_CAPS["max_price"],
                max(HARD_CAPS["min_max_price"], round(p90 * 1.2, 2)),
            )
            logger.info(
                f"  Price range from winners: ${prices.min():.2f}–${prices.max():.2f} | "
                f"10th–90th: ${p10:.2f}–${p90:.2f} → "
                f"filters: ${filters['min_price']}–${filters['max_price']}"
                + (
                    f" (max_price floored at ${HARD_CAPS['min_max_price']})"
                    if round(p90 * 1.2, 2) < HARD_CAPS["min_max_price"]
                    else ""
                )
            )

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_col = find_col(winner_df, "volume")
    if vol_col:
        vols = pd.to_numeric(winner_df[vol_col], errors="coerce").dropna()
        if len(vols) >= MIN_SAMPLES_FOR_FILTER:
            p10_vol = float(vols.quantile(LOWER_PCT / 100))
            filters["min_volume"] = max(
                HARD_CAPS["min_volume"],
                int(round(p10_vol * 0.7, -3))
            )
            logger.info(
                f"  Volume 10th pct from winners: {p10_vol:,.0f} → "
                f"min_volume filter: {filters['min_volume']:,}"
            )

    # ── Screener-relevant features ────────────────────────────────────────────
    for base_feat, (min_key, max_key) in SCREENER_FEATURE_MAP.items():
        if base_feat not in top_features_set:
            logger.debug(f"  {base_feat}: not in top features, skipping")
            continue

        col = find_col(winner_df, base_feat)
        if col is None:
            logger.debug(f"  {base_feat}: no column found in winner snapshot — skipping")
            continue

        w_vals = pd.to_numeric(winner_df[col], errors="coerce").dropna()
        if len(w_vals) < MIN_SAMPLES_FOR_FILTER:
            continue

        p10_w = float(w_vals.quantile(LOWER_PCT / 100))
        p90_w = float(w_vals.quantile(UPPER_PCT / 100))

        # ── Discriminativeness check ─────────────────────────────────────────
        discriminative = True
        if not non_winner_df.empty:
            nw_col = find_col(non_winner_df, base_feat)
            if nw_col:
                nw_vals = pd.to_numeric(non_winner_df[nw_col], errors="coerce").dropna()
                if len(nw_vals) >= MIN_SAMPLES_FOR_FILTER:
                    nw_median = float(nw_vals.median())
                    w_median  = float(w_vals.median())
                    if nw_median != 0 and abs(w_median - nw_median) / abs(nw_median) < 0.10:
                        discriminative = False
                        logger.debug(
                            f"  {base_feat}: winner median {w_median:.2f} vs "
                            f"non-winner {nw_median:.2f} — not discriminative, skipping"
                        )
        if not discriminative:
            continue

        # ── FIX 3: Apply hard caps ───────────────────────────────────────────
        min_val = round(p10_w, 4)

        if min_key in ("min_hv10",):
            cap = HARD_CAPS["max_min_hv10"]
            if min_val > cap:
                logger.info(
                    f"  {base_feat}: raw p10={min_val:.2f} exceeds cap {cap} "
                    f"→ clamping {min_key} to {cap}"
                )
                min_val = cap

        elif min_key in ("min_hv20",):
            cap = HARD_CAPS["max_min_hv20"]
            if min_val > cap:
                logger.info(
                    f"  {base_feat}: raw p10={min_val:.2f} exceeds cap {cap} "
                    f"→ clamping {min_key} to {cap}"
                )
                min_val = cap

        elif min_key in ("min_volume_ratio", "min_relative_volume"):
            cap = HARD_CAPS["max_min_volume_ratio"]
            if min_val > cap:
                logger.info(
                    f"  {base_feat}: raw p10={min_val:.2f} exceeds cap {cap} "
                    f"→ clamping {min_key} to {cap}"
                )
                min_val = cap

        filters[min_key] = min_val
        if max_key and p90_w > p10_w:
            filters[max_key] = round(p90_w, 4)

        logger.info(
            f"  {base_feat}: winner 10th–90th = {p10_w:.2f}–{p90_w:.2f} "
            f"→ {min_key}={filters.get(min_key)}"
            + (f", {max_key}={filters.get(max_key)}" if max_key else "")
        )

    # ── Relative volume ───────────────────────────────────────────────────────
    rv_col = find_col(winner_df, "volume_ratio")
    if rv_col:
        rv_vals = pd.to_numeric(winner_df[rv_col], errors="coerce").dropna()
        if len(rv_vals) >= MIN_SAMPLES_FOR_FILTER:
            p10_rv = float(rv_vals.quantile(LOWER_PCT / 100))
            min_rv = max(HARD_CAPS["min_relative_volume"], round(p10_rv * 0.8, 2))
            min_rv = min(min_rv, HARD_CAPS["max_min_volume_ratio"])
            filters["min_relative_volume"] = min_rv
            filters["min_volume_ratio"]    = min_rv
            logger.info(
                f"  volume_ratio 10th pct from winners: {p10_rv:.2f} → "
                f"min_relative_volume filter: {min_rv}"
            )

    filters["_derived_from_n_winners"] = int(len(winner_df))
    filters["_derived_at"]             = datetime.now().isoformat()
    filters["_lookback_days"]          = FILTER_LOOKBACK_DAYS

    return filters


def _conservative_defaults() -> dict:
    return {
        "min_price":               1.00,
        "max_price":               50.0,
        "min_volume":              300_000,
        "min_volume_ratio":        2.0,
        "min_relative_volume":     2.0,
        "_derived_from_n_winners": 0,
        "_derived_at":             datetime.now().isoformat(),
        "_note":                   "conservative defaults — not enough winner samples yet",
    }


def learn_and_write_filters(client, logger: logging.Logger) -> dict:
    logger.info("")
    logger.info("=" * 60)
    logger.info("MODEL-DRIVEN FILTER LEARNING")
    logger.info("=" * 60)

    top_features = load_feature_importance(FEATURE_IMPORTANCE_PATH, top_n=40)
    if top_features:
        logger.info(f"Top features (first 10): {top_features[:10]}")
        screener_hits = [f for f in top_features if f in SCREENER_FEATURE_MAP]
        logger.info(f"Screener-relevant top features: {screener_hits}")
    else:
        logger.warning("No feature importance file — using conservative defaults")
        filters = _conservative_defaults()
        _write_filters(filters, logger)
        return filters

    winner_df     = fetch_winner_t1_snapshots(client, FILTER_LOOKBACK_DAYS)
    non_winner_df = fetch_non_winner_t1_snapshots(client, FILTER_LOOKBACK_DAYS)

    logger.info(f"\nComputing filters from {len(winner_df)} winners, "
                f"{len(non_winner_df)} non-winners...")
    filters = compute_model_driven_filters(winner_df, non_winner_df, top_features, logger)

    if LEARNED_FILTERS_PATH.exists():
        try:
            with open(LEARNED_FILTERS_PATH) as f:
                existing = json.load(f)
            for k, v in existing.items():
                if k.startswith("_") and k not in filters:
                    filters[k] = v
        except Exception:
            pass

    _write_filters(filters, logger)
    return filters


def _write_filters(filters: dict, logger: logging.Logger) -> None:
    LEARNED_FILTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_FILTERS_PATH, "w") as f:
        json.dump(filters, f, indent=2)
    logger.info(f"✓ Wrote learned filters → {LEARNED_FILTERS_PATH}")
    logger.info(f"  Fields: {[k for k in filters if not k.startswith('_')]}")


# ---------------------------------------------------------------------------
# Accuracy tracker
# ---------------------------------------------------------------------------

class ComprehensiveAccuracyTracker:
    def __init__(self, config: dict):
        self.logger  = logging.getLogger(__name__)
        self.config  = config
        self.supabase = MLPredictionSupabaseClient(config)
        self.client  = self.supabase.client

    def get_predictions_for_date(self, check_date: str) -> pd.DataFrame:
        self.logger.info(f"Fetching predictions for {check_date}...")
        response = (
            self.client.table("ml_explosion_predictions")
            .select("*")
            .eq("prediction_date", check_date)
            .execute()
        )
        if not response.data:
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        self.logger.info(f"Found {len(df)} predictions")
        return df

    def get_actual_winners_for_date(self, check_date: str) -> pd.DataFrame:
        self.logger.info(f"Fetching actual winners for {check_date}...")
        response = (
            self.client.table("daily_winners")
            .select("symbol,change_pct,price,volume,high,low,open,close")
            .eq("detection_date", check_date)
            .execute()
        )
        if not response.data:
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        self.logger.info(f"Found {len(df)} actual winners")
        return df

    def get_actual_non_winners_for_date(self, check_date: str) -> pd.DataFrame:
        response = (
            self.client.table("daily_non_winners")
            .select("symbol,change_pct,price,volume")
            .eq("detection_date", check_date)
            .execute()
        )
        if not response.data:
            return pd.DataFrame()
        return pd.DataFrame(response.data)

    def analyze_prediction_accuracy(
        self,
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame,
        yfinance_gains: dict,
    ) -> tuple:
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ANALYZING PREDICTION ACCURACY")
        self.logger.info("=" * 60)

        if not winners_df.empty:
            self.logger.info(f"Winner columns: {winners_df.columns.tolist()}")

        # winners_set is used only to look up supplementary price/volume data
        # from the daily_winners table — it no longer determines became_winner.
        # became_winner is now derived purely from actual_high_pct ≥
        # INTRADAY_WIN_THRESHOLD so that stocks which exploded intraday but
        # pulled back before close are still counted as winners.
        if not winners_df.empty and "symbol" in winners_df.columns:
            winners_set = set(winners_df["symbol"].tolist())
        else:
            winners_set = set()
            if winners_df.empty:
                self.logger.warning(
                    "winners_df is empty — daily_winners price data unavailable. "
                    "became_winner will be determined from yfinance actual_high_pct only."
                )
            else:
                self.logger.warning(
                    "winners_df has no 'symbol' column — cannot look up winner price data. "
                    f"Available columns: {winners_df.columns.tolist()}"
                )

        yf_populated = sum(
            1 for r in yfinance_gains.values()
            if r.get("actual_gain_pct") is not None
        )
        self.logger.info(
            f"yfinance gain data available for {yf_populated}/{len(yfinance_gains)} symbols"
        )

        accuracy_records = []
        details_records  = []
        true_positives = false_positives = true_negatives = false_negatives = 0

        for _, pred in predictions_df.iterrows():
            symbol = pred["symbol"]
            # Use signal (percentile-ranked label) rather than the raw XGBoost binary.
            # See FIX 5 in the module docstring for full rationale.
            predicted_positive = pred.get("signal", "") in ("BUY", "STRONG BUY")

            yf_data = yfinance_gains.get(symbol, {})

            # Resolve actual price data — prefer yfinance (uses prev_close denominator)
            # and fall back to the daily_winners row where available.
            in_winners_table = symbol in winners_set
            if in_winners_table and not winners_df.empty:
                winner_row   = winners_df[winners_df["symbol"] == symbol].iloc[0]
                actual_gain  = float(winner_row["change_pct"])
                actual_price = float(winner_row["price"])
                actual_volume = (
                    int(winner_row["volume"])
                    if pd.notna(winner_row.get("volume"))
                    else yf_data.get("actual_volume")
                )
            else:
                actual_gain   = yf_data.get("actual_gain_pct")
                actual_price  = yf_data.get("actual_close") or pred.get("current_price", 0)
                actual_volume = yf_data.get("actual_volume")

            # Always prefer yfinance for actual_high_pct (prev_close denominator).
            # Fall back to the daily_winners high column only if yfinance has no data.
            #
            # FIX2 (denominator tagging): the two branches below compute
            # actual_high_pct on DIFFERENT bases — yfinance uses prev_close,
            # the daily_winners fallback uses actual_price (same-day price).
            # Previously this distinction was computed and then discarded, so
            # nothing downstream (ml_retrain_model.py's gain regressor) could
            # tell a prev_close-based value apart from a same-day-price one.
            # That silent mixing is exactly what produced the ~60pp mean-gain
            # divergence flagged by the FIX2 diagnostic in ml_retrain_model.py.
            # We now record which base was used in actual_high_pct_source so
            # downstream consumers can filter/weight accordingly.
            if yf_data.get("actual_high_pct") is not None:
                actual_high_pct = yf_data["actual_high_pct"]
                actual_high_pct_source = "yfinance_prev_close"
            elif in_winners_table and not winners_df.empty:
                winner_row  = winners_df[winners_df["symbol"] == symbol].iloc[0]
                w_high      = winner_row.get("high", actual_price)
                actual_high_pct = (
                    ((float(w_high) / actual_price) - 1) * 100
                    if actual_price and actual_price > 0 else None
                )
                # NOTE: this base is same-day actual_price, NOT prev_close — it
                # is the fallback flagged as noisy in the module docstring (FIX2).
                actual_high_pct_source = (
                    "winners_table_same_day_price" if actual_high_pct is not None else None
                )
            else:
                actual_high_pct = None
                actual_high_pct_source = None

            # ── Winner definition ────────────────────────────────────────────
            # A stock is a winner if its intraday high on prediction date reached
            # INTRADAY_WIN_THRESHOLD (20%) above the prior close.  Close price is
            # irrelevant — stocks that exploded and then pulled back still count.
            became_winner = (
                actual_high_pct is not None
                and actual_high_pct >= INTRADAY_WIN_THRESHOLD
            )

            # ── Correctness ──────────────────────────────────────────────────
            # Correct = signal matched the outcome:
            #   BUY / STRONG BUY  → stock hit ≥20% intraday high  (true positive)
            #   HOLD / AVOID      → stock did NOT hit ≥20% intraday high  (true negative)
            prediction_correct = (predicted_positive and became_winner) or (
                not predicted_positive and not became_winner
            )

            # Gain error — compare predicted target gain vs actual intraday high
            # (the figure that matters for momentum traders entering at open).
            predicted_gain = pred.get("target_gain_pct") or 0
            if predicted_gain and predicted_gain > 0 and actual_high_pct is not None:
                gain_error       = abs(predicted_gain - actual_high_pct)
                gain_error_ratio = gain_error / actual_high_pct if actual_high_pct != 0 else 0
            else:
                gain_error = gain_error_ratio = None

            if predicted_positive and became_winner:
                outcome_type = "true_positive"
                true_positives += 1
            elif predicted_positive and not became_winner:
                outcome_type = "false_positive"
                false_positives += 1
            elif not predicted_positive and not became_winner:
                outcome_type = "true_negative"
                true_negatives += 1
            else:
                outcome_type = "false_negative"
                false_negatives += 1

            accuracy_records.append({
                "symbol":                 symbol,
                "prediction_date":        pred["prediction_date"],
                "predicted_probability":  pred["explosion_probability"],
                "predicted_signal":       pred["signal"],
                "predicted_target_gain":  pred.get("target_gain_pct"),
                "predicted_target_price": pred.get("target_price"),
                "became_winner":          became_winner,
                "actual_gain_pct":        actual_gain,
                "actual_high_pct":        actual_high_pct,
                # FIX2: tags which denominator actual_high_pct was computed with
                # ("yfinance_prev_close" or "winners_table_same_day_price"), so
                # ml_retrain_model.py can exclude/downweight the noisier
                # same-day-price fallback instead of silently blending both
                # bases into one gain-regressor target. Requires a matching
                # `actual_high_pct_source text` column on ml_prediction_accuracy;
                # if that column doesn't exist yet in Supabase, add it via
                # migration before this starts populating (upsert will otherwise
                # either error or silently drop the field depending on client
                # config — check before relying on it downstream).
                "actual_high_pct_source": actual_high_pct_source,
                "actual_price":           actual_price,
                "prediction_correct":     prediction_correct,
                "gain_error_pct":         gain_error,
                "gain_error_ratio":       gain_error_ratio,
                # FIX 7: pre-market extremes, relative to the same prev_close
                # basis as actual_high_pct. None if yfinance had no intraday
                # data for this symbol/date (common for older backfills,
                # since 5m bars are only retained for ~30 days).
                # Requires actual_premarket_high_pct / actual_premarket_low_pct
                # / actual_premarket_volume columns on ml_prediction_accuracy —
                # add via migration before relying on them downstream (same
                # caveat as actual_high_pct_source above).
                "actual_premarket_high_pct": yf_data.get("actual_premarket_high_pct"),
                "actual_premarket_low_pct":  yf_data.get("actual_premarket_low_pct"),
                "actual_premarket_volume":   yf_data.get("actual_premarket_volume"),
                "actual_recorded_at":     datetime.now().isoformat(),
            })

            details_records.append({
                "symbol":                symbol,
                "prediction_date":       pred["prediction_date"],
                "predicted_probability": pred["explosion_probability"],
                "predicted_signal":      pred["signal"],
                "outcome_type":          outcome_type,
                "became_winner":         became_winner,
                "actual_gain_pct":       actual_gain,
                "actual_high_pct":       actual_high_pct,
                "actual_premarket_high_pct": yf_data.get("actual_premarket_high_pct"),
                "actual_premarket_low_pct":  yf_data.get("actual_premarket_low_pct"),
                "actual_volume":         actual_volume,
                "failure_reason":        None,
            })

        total          = len(predictions_df)
        predicted_buys = true_positives + false_positives
        all_correct    = true_positives + true_negatives
        accuracy       = (all_correct    / total * 100) if total > 0 else 0
        precision      = (true_positives / predicted_buys * 100) if predicted_buys > 0 else 0
        recall         = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0

        gain_populated = sum(1 for r in accuracy_records if r.get("actual_gain_pct") is not None)
        high_populated = sum(1 for r in accuracy_records if r.get("actual_high_pct") is not None)

        self.logger.info(f"\nPrediction Accuracy (winner = intraday high ≥ {INTRADAY_WIN_THRESHOLD}%):")
        self.logger.info(f"  Total predictions:         {total}")
        self.logger.info(f"  True Positives:            {true_positives}  ← BUY/STRONG BUY + hit {INTRADAY_WIN_THRESHOLD}%+ intraday")
        self.logger.info(f"  False Positives:           {false_positives}  ← BUY/STRONG BUY + did NOT hit {INTRADAY_WIN_THRESHOLD}%+")
        self.logger.info(f"  True Negatives:            {true_negatives}  ← HOLD/AVOID + did NOT hit {INTRADAY_WIN_THRESHOLD}%+")
        self.logger.info(f"  False Negatives:           {false_negatives}  ← HOLD/AVOID + DID hit {INTRADAY_WIN_THRESHOLD}%+")
        self.logger.info(f"  Accuracy:                  {accuracy:.2f}%")
        self.logger.info(f"  Precision:                 {precision:.2f}%")
        self.logger.info(f"  Recall:                    {recall:.2f}%")
        self.logger.info(f"  actual_gain_pct populated: {gain_populated}/{total} ({gain_populated/total*100:.1f}%)")
        self.logger.info(f"  actual_high_pct populated: {high_populated}/{total} ({high_populated/total*100:.1f}%)")

        return accuracy_records, details_records

    def analyze_missed_opportunities(
        self,
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame,
        check_date: str,
        yfinance_gains: dict = None,
    ) -> list:
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ANALYZING MISSED OPPORTUNITIES")
        self.logger.info("=" * 60)

        # Guard: if winners_df is empty or missing symbol column, nothing to compare
        if winners_df.empty or "symbol" not in winners_df.columns:
            self.logger.info("No winner data available — skipping missed opportunity analysis.")
            return []

        predicted_symbols = set(predictions_df["symbol"].tolist())
        winner_symbols    = set(winners_df["symbol"].tolist())
        missed_symbols    = winner_symbols - predicted_symbols

        self.logger.info(f"Missed {len(missed_symbols)} winners not in prediction set")

        yf = yfinance_gains or {}

        missed_records = []
        for symbol in missed_symbols:
            winner_data  = winners_df[winners_df["symbol"] == symbol].iloc[0]
            actual_price = float(winner_data["price"])

            # FIX 2: use yfinance actual_high_pct (prev_close denominator) when available
            yf_data = yf.get(symbol, {})
            if yf_data.get("actual_high_pct") is not None:
                actual_high_pct = yf_data["actual_high_pct"]
            else:
                w_high = winner_data.get("high", actual_price)
                actual_high_pct = (
                    ((float(w_high) / actual_price) - 1) * 100
                    if actual_price and actual_price > 0 else None
                )
                if actual_high_pct is not None:
                    self.logger.debug(
                        f"{symbol}: missed opp actual_high_pct using same-day price "
                        "(no yfinance data) — value will be near 0%"
                    )

            # Volume can be NaN for some rows — int(NaN) raises ValueError,
            # which previously could abort analysis for the whole date.
            raw_volume = winner_data.get("volume")
            actual_volume = int(raw_volume) if pd.notna(raw_volume) else None

            missed_records.append({
                "symbol":                   symbol,
                "detection_date":           check_date,
                "exchange":                 winner_data.get("exchange", "UNKNOWN"),
                "actual_gain_pct":          float(winner_data["change_pct"]),
                "actual_high_pct":          actual_high_pct,
                "actual_premarket_high_pct": yf_data.get("actual_premarket_high_pct"),
                "actual_premarket_low_pct":  yf_data.get("actual_premarket_low_pct"),
                "actual_price":             actual_price,
                "actual_volume":            actual_volume,
                "was_screened":             False,
                "screening_failure_reason": self._determine_screening_failure(winner_data),
                "predicted_probability":    None,
                "predicted_signal":         None,
            })

        return missed_records

    def _determine_screening_failure(self, winner_data: pd.Series) -> str:
        price  = winner_data.get("price", 0)
        volume = winner_data.get("volume", 0)
        if price < 0.50:
            return "price_too_low"
        elif price > 500.0:
            return "price_too_high"
        elif volume < 100_000:
            return "volume_too_low"
        return "not_in_screener_results"

    def write_all_records(
        self,
        accuracy_records: list,
        details_records: list,
        missed_records: list,
    ):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WRITING RECORDS TO DATABASE")
        self.logger.info("=" * 60)

        if accuracy_records:
            self.logger.info(f"Writing {len(accuracy_records)} accuracy records...")
            self.supabase.write_accuracy_records(accuracy_records)

        if details_records:
            self.logger.info(f"Writing {len(details_records)} detail records...")
            try:
                # FIX 6: was self.client.table(...).upsert(...) directly, which
                # sends raw numpy/NaN values and fails the WHOLE batch (all
                # rows, not just the bad one) with "Out of range float values
                # are not JSON compliant" the moment any single row has a
                # NaN/Infinity. write_records_upsert() sanitizes every row
                # first, same as the main accuracy table.
                self.supabase.write_records_upsert(
                    "ml_accuracy_details", details_records, on_conflict="symbol,prediction_date"
                )
            except Exception as e:
                self.logger.error(f"Failed to write details: {e}")

        if missed_records:
            self.logger.info(f"Writing {len(missed_records)} missed opportunity records...")
            try:
                self.supabase.write_records_upsert(
                    "ml_missed_opportunities", missed_records, on_conflict="symbol,detection_date"
                )
            except Exception as e:
                self.logger.error(f"Failed to write missed opportunities: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ML accuracy tracking with model-driven filter learning"
    )
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--date",    type=str, help="Date to check (YYYY-MM-DD)")
    parser.add_argument(
        "--backfill-yesterday", action="store_true",
        help=(
            "Convenience shortcut for --date <yesterday, local date>. Useful "
            "for repairing a run that failed partway (e.g. wrote accuracy "
            "records but not details/missed) — re-running with the same "
            "date upserts on (symbol, prediction_date)/(symbol, detection_date), "
            "so it fully overwrites/repairs the prior partial data rather "
            "than duplicating or skipping it."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--filters-only", action="store_true",
        help="Skip accuracy analysis — only recompute learned_filters.json",
    )
    parser.add_argument(
        "--yfinance-workers", type=int, default=YFINANCE_MAX_WORKERS,
        help=f"Parallel workers for yfinance fetches (default: {YFINANCE_MAX_WORKERS})",
    )
    parser.add_argument(
        "--no-premarket", action="store_true",
        help=(
            "Skip the extra pre-market (5m, prepost=True) yfinance fetch. "
            "Pre-market fetching doubles yfinance calls per symbol; disable "
            "if you hit rate limits and only need regular-session accuracy."
        ),
    )

    args = parser.parse_args()

    if args.date and args.backfill_yesterday:
        parser.error("--date and --backfill-yesterday are mutually exclusive")

    if args.backfill_yesterday:
        args.date = (datetime.now().date() - timedelta(days=1)).isoformat()

    config    = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger    = setup_logging(log_level)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ML ACCURACY TRACKING WITH MODEL-DRIVEN LEARNING")
    logger.info("=" * 80)

    tracker = ComprehensiveAccuracyTracker(config)

    learned_filters = learn_and_write_filters(tracker.client, logger)

    if args.filters_only:
        logger.info("\n--filters-only flag set. Skipping accuracy analysis.")
        return 0

    if args.date:
        check_date = args.date
        source = "--backfill-yesterday" if args.backfill_yesterday else "manually specified"
        logger.info(f"Using {source} date: {check_date}")
    else:
        check_date = get_most_recent_prediction_date(tracker)
        if not check_date:
            logger.warning("⚠️ No predictions found in database. Nothing to track.")
            return 0
        logger.info(f"✓ Most recent prediction date: {check_date}")

    validation = validate_data_exists(tracker, check_date)
    if not validation["should_proceed"]:
        logger.warning("DATA VALIDATION FAILED — no predictions found, exiting.")
        return 0

    predictions_df = tracker.get_predictions_for_date(check_date)
    winners_df     = tracker.get_actual_winners_for_date(check_date)

    all_symbols = predictions_df["symbol"].tolist() if not predictions_df.empty else []

    yfinance_gains: dict = {}
    if all_symbols:
        yfinance_gains = fetch_actual_gains_for_all_symbols(
            symbols=all_symbols,
            prediction_date=check_date,
            logger=logger,
            max_workers=args.yfinance_workers,
            include_premarket=not args.no_premarket,
        )
    else:
        logger.warning("No predicted symbols found — skipping yfinance fetch.")

    accuracy_records, details_records = tracker.analyze_prediction_accuracy(
        predictions_df, winners_df, yfinance_gains
    )

    missed_records = tracker.analyze_missed_opportunities(
        predictions_df, winners_df, check_date,
        yfinance_gains=yfinance_gains,
    )

    tracker.write_all_records(accuracy_records, details_records, missed_records)

    gain_populated = sum(
        1 for r in accuracy_records if r.get("actual_gain_pct") is not None
    )
    logger.info("\n" + "=" * 80)
    logger.info("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    logger.info("=" * 80)
    premarket_populated = sum(
        1 for r in accuracy_records if r.get("actual_premarket_high_pct") is not None
    )
    logger.info(f"  Accuracy records written      : {len(accuracy_records)}")
    logger.info(f"  actual_gain_pct populated     : {gain_populated}/{len(accuracy_records)}")
    logger.info(f"  actual_premarket_high_pct pop.: {premarket_populated}/{len(accuracy_records)}")
    logger.info(f"  Detail records written        : {len(details_records)}")
    logger.info(f"  Missed records written        : {len(missed_records)}")
    logger.info(f"\n  Learned filters updated       : {LEARNED_FILTERS_PATH}")
    derived_fields = [k for k in learned_filters if not k.startswith("_")]
    logger.info(f"  Filter fields derived         : {derived_fields}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
