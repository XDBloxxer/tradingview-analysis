#!/usr/bin/env python3
"""
Comprehensive ML Accuracy Tracker with MODEL-DRIVEN LEARNING

IMPROVEMENTS over previous version:
1. ✅ Finds most recent prediction date automatically
2. ✅ Validates data exists before fetching (saves egress)
3. ✅ Early exit if no data to process
4. ✅ MODEL-DRIVEN filter learning from feature_importance.csv + winner stats
5. ✅ Tightens screening population (price/volume/volatility) using actual winner data
6. ✅ Computes percentile ranges of top features across historical winners

HOW FILTER LEARNING WORKS:
  - Load feature_importance.csv to find the top non-t1 features
  - Fetch historical T-1 close snapshots for actual winners from Supabase
  - Compute 10th–90th percentile ranges of key screener-relevant features
    (HV_10, Volume_Ratio, RSI_14, price range, etc.)
  - Write those as hard filters to learned_filters.json
  - SmartScreener reads learned_filters.json at startup — no other changes needed

WHY THIS MATTERS:
  The original learned_filters.json had only 6 fields and never updated
  meaningfully. A stock screened with min_volume=100k and max_price=500
  returns thousands of large-caps that look NOTHING like the training data
  (small-cap, high-vol, elevated relative volume). This fix makes the
  screened population match the distribution the model was actually
  trained on.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import yaml
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_predictor.ml_supabase_client import MLPredictionSupabaseClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_IMPORTANCE_PATH = Path("ml_models/feature_importance.csv")
LEARNED_FILTERS_PATH    = Path("ml_models/learned_filters.json")

# Percentile bounds for filter ranges
LOWER_PCT = 10   # 10th percentile → min filter value
UPPER_PCT = 90   # 90th percentile → max filter value

# How many days of winner T-1 snapshots to use when computing filter ranges
FILTER_LOOKBACK_DAYS = 90

# Minimum winner samples needed before we trust the derived filters
MIN_SAMPLES_FOR_FILTER = 20

# Features in the model that map directly to TradingView screener fields.
# Key   = model feature base name (without t3_/t5_/t10_ prefix)
# Value = learned_filters.json key that SmartScreener reads
SCREENER_FEATURE_MAP = {
    "HV_10":        ("min_hv10",        "max_hv10"),
    "HV_20":        ("min_hv20",        "max_hv20"),
    "Volume_Ratio": ("min_volume_ratio", None),
    "RSI_14":       ("min_rsi",         "max_rsi"),
    "RSI_7":        ("min_rsi7",        "max_rsi7"),
    "ATR_14":       ("min_atr14",       None),
    "ADX_14":       ("min_adx",         None),
}

# Hard caps — regardless of data, never go beyond these
HARD_CAPS = {
    "min_price":           0.50,
    "max_price":           500.0,
    "min_volume":          100_000,
    "min_relative_volume": 1.0,
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
# Helpers — data validation (unchanged from previous version)
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
        "winners_exist": False,
        "should_proceed": False,
        "prediction_count": 0,
        "winner_count": 0,
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

def load_feature_importance(path: Path, top_n: int = 30) -> list[str]:
    """
    Load feature_importance.csv and return the top-N non-t1 feature base names.
    Strips the t3_/t5_/t10_ prefix so we get base names like 'HV_10', 'EMA_10'.
    """
    if not path.exists():
        return []
    try:
        fi = pd.read_csv(path)
        fi = fi[fi["importance"] > 0].sort_values("importance", ascending=False)

        base_names = []
        seen = set()
        for feat in fi["feature"]:
            # Strip time prefix
            for prefix in ("t3_", "t5_", "t10_", "t1_close_", "t1_open_"):
                if feat.startswith(prefix):
                    feat = feat[len(prefix):]
                    break
            # Skip t1 features — they're training features, not screener features
            if feat.startswith("t1_"):
                continue
            if feat not in seen:
                seen.add(feat)
                base_names.append(feat)
            if len(base_names) >= top_n:
                break
        return base_names
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load feature importance: {e}")
        return []


def fetch_winner_t1_snapshots(client, lookback_days: int) -> pd.DataFrame:
    """
    Fetch winners_day_prior_close rows for the last `lookback_days` days.
    These are T-1 snapshots of stocks that actually exploded — the ground-truth
    population we want to screen for.
    """
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
    """Fetch non_winners_day_prior_close for comparison."""
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


def _col_variants(base: str) -> list[str]:
    """
    Return all column name variants to try for a given base feature name.
    The T-1 Supabase tables use intraday short names (lowercase).
    We try several forms to find the column.
    """
    variants = [
        base,                            # RSI_14
        base.lower(),                    # rsi_14
        base.replace("_", "").lower(),   # rsi14
    ]
    # Special overrides for known mismatches
    special = {
        "HV_10":        ["volatility_10d", "hv_10", "hv10"],
        "HV_20":        ["volatility_20d", "hv_20", "hv20"],
        "HV_30":        ["volatility_30d", "hv_30", "hv30"],
        "Volume_Ratio": ["volume_ratio"],
        "RSI_14":       ["rsi", "rsi14"],
        "RSI_7":        ["rsi7"],
        "ATR_14":       ["atr", "atr14"],
        "ADX_14":       ["adx"],
        "OBV":          ["obv"],
        "Close":        ["close"],
        "Volume":       ["volume"],
    }
    if base in special:
        variants = special[base] + variants
    return variants


def find_col(df: pd.DataFrame, base: str) -> Optional[str]:
    """Find the actual column name in df for a base feature name."""
    for v in _col_variants(base):
        if v in df.columns:
            return v
    return None


def compute_model_driven_filters(
    winner_df: pd.DataFrame,
    non_winner_df: pd.DataFrame,
    top_features: list[str],
    logger: logging.Logger,
) -> dict:
    """
    Core filter learning logic.

    For each screener-relevant feature (from SCREENER_FEATURE_MAP),
    compute percentile ranges across winner snapshots. Then compare
    to non-winners to confirm the filter is discriminative.

    Also derives price/volume ranges from the winner distribution.

    Returns:
        dict of learned_filters ready to write to JSON
    """
    filters = {}

    if winner_df.empty or len(winner_df) < MIN_SAMPLES_FOR_FILTER:
        logger.warning(
            f"Only {len(winner_df)} winner samples — need {MIN_SAMPLES_FOR_FILTER} "
            "for reliable filter derivation. Using conservative defaults."
        )
        return _conservative_defaults()

    # ── 1. Price range from actual winners ─────────────────────────────────
    price_col = find_col(winner_df, "Close")
    if price_col:
        prices = pd.to_numeric(winner_df[price_col], errors="coerce").dropna()
        if len(prices) >= MIN_SAMPLES_FOR_FILTER:
            p10 = float(prices.quantile(LOWER_PCT / 100))
            p90 = float(prices.quantile(UPPER_PCT / 100))
            # Apply hard caps
            filters["min_price"] = max(HARD_CAPS["min_price"], round(p10 * 0.8, 2))
            filters["max_price"] = min(HARD_CAPS["max_price"], round(p90 * 1.2, 2))
            logger.info(
                f"  Price range from winners: ${prices.min():.2f}–${prices.max():.2f} | "
                f"10th–90th: ${p10:.2f}–${p90:.2f} → "
                f"filters: ${filters['min_price']}–${filters['max_price']}"
            )

    # ── 2. Volume range from actual winners ────────────────────────────────
    vol_col = find_col(winner_df, "Volume")
    if vol_col:
        vols = pd.to_numeric(winner_df[vol_col], errors="coerce").dropna()
        if len(vols) >= MIN_SAMPLES_FOR_FILTER:
            p10_vol = float(vols.quantile(LOWER_PCT / 100))
            filters["min_volume"] = max(
                HARD_CAPS["min_volume"],
                int(round(p10_vol * 0.7, -3))  # round to nearest 1000
            )
            logger.info(
                f"  Volume 10th pct from winners: {p10_vol:,.0f} → "
                f"min_volume filter: {filters['min_volume']:,}"
            )

    # ── 3. Screener-relevant features (HV, RSI, Volume_Ratio, etc.) ────────
    for base_feat, (min_key, max_key) in SCREENER_FEATURE_MAP.items():
        # Only derive if this feature is in the top-N important
        if base_feat not in top_features:
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

        # Compare to non-winners — only use filter if there's real separation
        discriminative = True
        if not non_winner_df.empty:
            nw_col = find_col(non_winner_df, base_feat)
            if nw_col:
                nw_vals = pd.to_numeric(non_winner_df[nw_col], errors="coerce").dropna()
                if len(nw_vals) >= MIN_SAMPLES_FOR_FILTER:
                    nw_median = float(nw_vals.median())
                    w_median  = float(w_vals.median())
                    # If medians are within 10% of each other, feature isn't discriminative
                    if nw_median != 0 and abs(w_median - nw_median) / abs(nw_median) < 0.10:
                        discriminative = False
                        logger.debug(
                            f"  {base_feat}: winner median {w_median:.2f} vs "
                            f"non-winner {nw_median:.2f} — not discriminative, skipping"
                        )

        if not discriminative:
            continue

        # Write min filter
        filters[min_key] = round(p10_w, 4)

        # Write max filter if defined and meaningful
        if max_key and p90_w > p10_w:
            filters[max_key] = round(p90_w, 4)

        logger.info(
            f"  {base_feat}: winner 10th–90th = {p10_w:.2f}–{p90_w:.2f} "
            f"→ {min_key}={filters.get(min_key)} "
            + (f", {max_key}={filters.get(max_key)}" if max_key else "")
        )

    # ── 4. Relative volume (high-importance proxy) ─────────────────────────
    rv_col = find_col(winner_df, "Volume_Ratio")
    if rv_col:
        rv_vals = pd.to_numeric(winner_df[rv_col], errors="coerce").dropna()
        if len(rv_vals) >= MIN_SAMPLES_FOR_FILTER:
            p10_rv = float(rv_vals.quantile(LOWER_PCT / 100))
            min_rv = max(HARD_CAPS["min_relative_volume"], round(p10_rv * 0.8, 2))
            filters["min_relative_volume"] = min_rv
            filters["min_volume_ratio"]    = min_rv   # SmartScreener reads this key
            logger.info(
                f"  Volume_Ratio 10th pct from winners: {p10_rv:.2f} → "
                f"min_relative_volume filter: {min_rv}"
            )

    # ── 5. Metadata ────────────────────────────────────────────────────────
    filters["_derived_from_n_winners"] = int(len(winner_df))
    filters["_derived_at"] = datetime.now().isoformat()
    filters["_lookback_days"] = FILTER_LOOKBACK_DAYS

    return filters


def _conservative_defaults() -> dict:
    """Fallback filters when not enough data to derive from winners."""
    return {
        "min_price":           1.00,
        "max_price":           50.0,
        "min_volume":          300_000,
        "min_volume_ratio":    2.0,
        "min_relative_volume": 2.0,
        "_derived_from_n_winners": 0,
        "_derived_at": datetime.now().isoformat(),
        "_note": "conservative defaults — not enough winner samples yet",
    }


def learn_and_write_filters(client, logger: logging.Logger) -> dict:
    """
    Full pipeline: load importance → fetch snapshots → derive filters → write JSON.

    Returns the filter dict that was written.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("MODEL-DRIVEN FILTER LEARNING")
    logger.info("=" * 60)

    # 1. Top features from importance file
    top_features = load_feature_importance(FEATURE_IMPORTANCE_PATH, top_n=40)
    if top_features:
        logger.info(f"Top features (first 10): {top_features[:10]}")
    else:
        logger.warning("No feature importance file — using conservative defaults")
        filters = _conservative_defaults()
        _write_filters(filters, logger)
        return filters

    # 2. Winner T-1 snapshots
    winner_df = fetch_winner_t1_snapshots(client, FILTER_LOOKBACK_DAYS)

    # 3. Non-winner T-1 snapshots (for discrimination check)
    non_winner_df = fetch_non_winner_t1_snapshots(client, FILTER_LOOKBACK_DAYS)

    # 4. Compute filters
    logger.info(f"\nComputing filters from {len(winner_df)} winners, "
                f"{len(non_winner_df)} non-winners...")
    filters = compute_model_driven_filters(winner_df, non_winner_df, top_features, logger)

    # 5. Merge with any existing non-derived fields (preserve manual overrides)
    if LEARNED_FILTERS_PATH.exists():
        try:
            with open(LEARNED_FILTERS_PATH) as f:
                existing = json.load(f)
            # Only preserve fields that start with _ (manual notes) or aren't computed
            for k, v in existing.items():
                if k.startswith("_") and k not in filters:
                    filters[k] = v
        except Exception:
            pass

    # 6. Write
    _write_filters(filters, logger)
    return filters


def _write_filters(filters: dict, logger: logging.Logger) -> None:
    LEARNED_FILTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_FILTERS_PATH, "w") as f:
        json.dump(filters, f, indent=2)
    logger.info(f"✓ Wrote learned filters → {LEARNED_FILTERS_PATH}")
    logger.info(f"  Fields: {[k for k in filters if not k.startswith('_')]}")


# ---------------------------------------------------------------------------
# Accuracy tracker (existing logic, minimally changed)
# ---------------------------------------------------------------------------

class ComprehensiveAccuracyTracker:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.supabase = MLPredictionSupabaseClient(config)
        self.client = self.supabase.client

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
        self, predictions_df: pd.DataFrame, winners_df: pd.DataFrame
    ) -> tuple:
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ANALYZING PREDICTION ACCURACY")
        self.logger.info("=" * 60)

        if not winners_df.empty:
          self.logger.info(f"Winner columns: {winners_df.columns.tolist()}")
          self.logger.info(f"Sample winner: {winners_df.iloc[0].to_dict()}")
        winners_set = set(winners_df["symbol"].tolist())
        accuracy_records = []
        details_records = []
        true_positives = false_positives = true_negatives = 0

        for _, pred in predictions_df.iterrows():
            symbol = pred["symbol"]
            predicted_positive = pred["prediction"] == 1
            became_winner = symbol in winners_set

            if became_winner:
                winner_data = winners_df[winners_df["symbol"] == symbol].iloc[0]
                actual_gain = winner_data["change_pct"]
                actual_price = winner_data["price"]
                actual_high = winner_data.get("high", actual_price)
                actual_high_pct = (
                    ((actual_high / actual_price) - 1) * 100 if actual_price > 0 else 0
                )
            else:
                actual_gain = 0
                actual_price = pred.get("current_price", 0)
                actual_high_pct = 0

            prediction_correct = (predicted_positive and became_winner) or (
                not predicted_positive and not became_winner
            )

            predicted_gain = pred.get("target_gain_pct", 0)
            if became_winner and predicted_gain and predicted_gain > 0:
                gain_error = abs(predicted_gain - actual_gain)
                gain_error_ratio = gain_error / actual_gain if actual_gain != 0 else 0
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

            accuracy_records.append(
                {
                    "symbol": symbol,
                    "prediction_date": pred["prediction_date"],
                    "predicted_probability": pred["explosion_probability"],
                    "predicted_signal": pred["signal"],
                    "predicted_target_gain": pred.get("target_gain_pct"),
                    "predicted_target_price": pred.get("target_price"),
                    "became_winner": became_winner,
                    "actual_gain_pct": actual_gain if became_winner else None,
                    "actual_high_pct": actual_high_pct if became_winner else None,
                    "actual_price": actual_price,
                    "prediction_correct": prediction_correct,
                    "gain_error_pct": gain_error,
                    "gain_error_ratio": gain_error_ratio,
                    "actual_recorded_at": datetime.now().isoformat(),
                }
            )

            # Safely extract volume — int(NaN) throws ValueError which would
            # silently abort the entire loop and leave all remaining records null.
            actual_volume = None
            if became_winner:
                try:
                    vol_series = winners_df[winners_df["symbol"] == symbol]["volume"]
                    if not vol_series.empty and pd.notna(vol_series.iloc[0]):
                        actual_volume = int(vol_series.iloc[0])
                except (ValueError, TypeError, IndexError) as e:
                    self.logger.warning(f"Could not extract volume for {symbol}: {e}")
            
            details_records.append(
                {
                    "symbol": symbol,
                    "prediction_date": pred["prediction_date"],
                    "predicted_probability": pred["explosion_probability"],
                    "predicted_signal": pred["signal"],
                    "outcome_type": outcome_type,
                    "became_winner": became_winner,
                    "actual_gain_pct": actual_gain if became_winner else None,
                    "actual_high_pct": actual_high_pct if became_winner else None,
                    "actual_volume": actual_volume,
                    "failure_reason": None,
                }
            )

        total = len(predictions_df)
        correct = true_positives + true_negatives
        accuracy_pct = (correct / total * 100) if total > 0 else 0
        predicted_winners = true_positives + false_positives
        precision = (
            (true_positives / predicted_winners * 100) if predicted_winners > 0 else 0
        )

        self.logger.info(f"\nPrediction Accuracy:")
        self.logger.info(f"  Total:           {total}")
        self.logger.info(f"  True Positives:  {true_positives}")
        self.logger.info(f"  False Positives: {false_positives}")
        self.logger.info(f"  True Negatives:  {true_negatives}")
        self.logger.info(f"  Accuracy:        {accuracy_pct:.2f}%")
        self.logger.info(f"  Precision:       {precision:.2f}%")

        return accuracy_records, details_records

    def analyze_missed_opportunities(
        self,
        predictions_df: pd.DataFrame,
        winners_df: pd.DataFrame,
        check_date: str,
    ) -> list:
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ANALYZING MISSED OPPORTUNITIES")
        self.logger.info("=" * 60)

        predicted_symbols = set(predictions_df["symbol"].tolist())
        winner_symbols    = set(winners_df["symbol"].tolist())
        missed_symbols    = winner_symbols - predicted_symbols

        self.logger.info(f"Missed {len(missed_symbols)} winners not in prediction set")

        missed_records = []
        for symbol in missed_symbols:
            winner_data = winners_df[winners_df["symbol"] == symbol].iloc[0]
            missed_records.append(
                {
                    "symbol": symbol,
                    "detection_date": check_date,
                    "exchange": winner_data.get("exchange", "UNKNOWN"),
                    "actual_gain_pct": winner_data["change_pct"],
                    "actual_high_pct": (
                        (winner_data.get("high", winner_data["price"]) / winner_data["price"] - 1)
                        * 100
                    ),
                    "actual_price": winner_data["price"],
                    "actual_volume": int(winner_data["volume"]),
                    "was_screened": False,
                    "screening_failure_reason": self._determine_screening_failure(winner_data),
                    "predicted_probability": None,
                    "predicted_signal": None,
                }
            )

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
                self.client.table("ml_accuracy_details").upsert(
                    details_records, on_conflict="symbol,prediction_date"
                ).execute()
            except Exception as e:
                self.logger.error(f"Failed to write details: {e}")

        if missed_records:
            self.logger.info(f"Writing {len(missed_records)} missed opportunity records...")
            try:
                self.client.table("ml_missed_opportunities").upsert(
                    missed_records, on_conflict="symbol,detection_date"
                ).execute()
            except Exception as e:
                self.logger.error(f"Failed to write missed opportunities: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ML accuracy tracking with model-driven filter learning"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", type=str, help="Date to check (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--filters-only",
        action="store_true",
        help="Skip accuracy analysis — only recompute learned_filters.json",
    )

    args = parser.parse_args()

    config   = load_config(args.config)
    log_level = "DEBUG" if args.verbose else "INFO"
    logger   = setup_logging(log_level)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ML ACCURACY TRACKING WITH MODEL-DRIVEN LEARNING")
    logger.info("=" * 80)

    tracker = ComprehensiveAccuracyTracker(config)

    # ── MODEL-DRIVEN FILTER LEARNING (runs every time) ─────────────────────
    # Even if accuracy analysis is skipped, filters are always refreshed
    # so the screener population stays aligned with the model.
    learned_filters = learn_and_write_filters(tracker.client, logger)

    if args.filters_only:
        logger.info("\n--filters-only flag set. Skipping accuracy analysis.")
        return 0

    # ── ACCURACY ANALYSIS ──────────────────────────────────────────────────
    if args.date:
        check_date = args.date
        logger.info(f"Using manually specified date: {check_date}")
    else:
        check_date = get_most_recent_prediction_date(tracker)
        if not check_date:
            logger.warning("⚠️ No predictions found in database. Nothing to track.")
            return 0
        logger.info(f"✓ Most recent prediction date: {check_date}")

    validation = validate_data_exists(tracker, check_date)
    if not validation["should_proceed"]:
        logger.warning("DATA VALIDATION FAILED — exiting early (no egress wasted).")
        logger.warning(
            f"  predictions_exist={validation['predictions_exist']}, "
            f"winners_exist={validation['winners_exist']}"
        )
        return 0

    predictions_df  = tracker.get_predictions_for_date(check_date)
    winners_df      = tracker.get_actual_winners_for_date(check_date)

    accuracy_records, details_records = tracker.analyze_prediction_accuracy(
        predictions_df, winners_df
    )

    missed_records = tracker.analyze_missed_opportunities(
        predictions_df, winners_df, check_date
    )

    tracker.write_all_records(accuracy_records, details_records, missed_records)

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Accuracy records written : {len(accuracy_records)}")
    logger.info(f"  Detail records written   : {len(details_records)}")
    logger.info(f"  Missed records written   : {len(missed_records)}")
    logger.info(f"\n  Learned filters updated  : {LEARNED_FILTERS_PATH}")
    derived_fields = [k for k in learned_filters if not k.startswith("_")]
    logger.info(f"  Filter fields derived    : {derived_fields}")
    logger.info(
        "\n  These filters will be applied by SmartScreener at next prediction run."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
