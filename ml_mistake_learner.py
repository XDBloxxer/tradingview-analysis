#!/usr/bin/env python3
"""
ML Mistake Learner

Fetches the model's own prediction errors from the accuracy tracking tables
and prepares them as high-signal training data for the weekly retraining step.

WHY THIS MATTERS:
  - Random non-winners teach "average stocks don't explode"
  - False positives teach "stocks that LOOK like winners but aren't"
  
  The second lesson is far more valuable — it targets the model's specific
  blind spots rather than teaching it things it already knows.

MISTAKE TYPES HANDLED:
  1. False Positives  — predicted=1, actual=0  → high-weight negative examples
  2. False Negatives  — predicted=0, actual=1  → high-weight positive examples
     (missed winners we never even screened are handled separately)

OUTPUT:
  Returns a DataFrame of training samples with:
    - All T-1 open/close features (renamed to model long-form names via t1_column_map)
    - label  (0 for false positive, 1 for false negative)
    - sample_weight  (higher than standard training samples)
    - mistake_type   (for logging/analysis — excluded from feature matrix by NON_FEATURE_COLS)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import os

from supabase import create_client, Client

try:
    from t1_column_map import rename_t1_columns
    T1_MAP_AVAILABLE = True
except ImportError:
    T1_MAP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "t1_column_map.py not found — T-1 features in mistake samples will use "
        "raw intraday names and won't match model features."
    )


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT CONSTANTS
# Mistakes are worth more than ordinary training samples because:
#   - They represent the model's specific failure modes
#   - Random non-winners are plentiful; hard negatives are rare
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_FALSE_POSITIVE  = 3.0   # Model was confidently wrong → punish hard
WEIGHT_FALSE_NEGATIVE  = 2.0   # Model missed a winner → reinforce
WEIGHT_STANDARD        = 1.0   # Ordinary winners / non-winners

# Minimum probability threshold to count as a "confident" false positive.
# Low-probability predictions being wrong isn't a mistake worth punishing.
FP_CONFIDENCE_THRESHOLD = 0.60


def _get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH MISTAKE RECORDS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_false_positives(
    client: Client,
    start_date: str,
    end_date: str,
    min_predicted_probability: float = FP_CONFIDENCE_THRESHOLD,
    limit: int = 2000,
) -> pd.DataFrame:
    """
    Pull records where model predicted explosion but stock didn't become a winner.
    Only includes records where the model was *confident* (prob >= threshold),
    because low-confidence wrong predictions are expected and less informative.
    """
    logger.info(f"Fetching false positives ({start_date} → {end_date}, "
                f"min_prob={min_predicted_probability})...")

    try:
        response = (
            client.table("ml_accuracy_details")
            .select("symbol, prediction_date, predicted_probability, "
                    "predicted_signal, outcome_type")
            .eq("outcome_type", "false_positive")
            .gte("prediction_date", start_date)
            .lte("prediction_date", end_date)
            .gte("predicted_probability", min_predicted_probability)
            .limit(limit)
            .execute()
        )
    except Exception:
        # Table may not exist yet — fall back to ml_prediction_accuracy
        logger.warning("ml_accuracy_details not found, trying ml_prediction_accuracy...")
        response = (
            client.table("ml_prediction_accuracy")
            .select("symbol, prediction_date, predicted_probability, predicted_signal")
            .eq("prediction_correct", False)
            .eq("became_winner", False)
            .gte("prediction_date", start_date)
            .lte("prediction_date", end_date)
            .gte("predicted_probability", min_predicted_probability)
            .limit(limit)
            .execute()
        )

    if not response.data:
        logger.info("  No false positives found in date range.")
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    df["mistake_type"] = "false_positive"
    df["label"] = 0
    df["sample_weight"] = WEIGHT_FALSE_POSITIVE
    logger.info(f"  Found {len(df)} confident false positives.")
    return df


def fetch_false_negatives(
    client: Client,
    start_date: str,
    end_date: str,
    limit: int = 2000,
) -> pd.DataFrame:
    """
    Pull records where model predicted no explosion but stock DID become a winner.
    These are missed opportunities the model should have caught.
    """
    logger.info(f"Fetching false negatives ({start_date} → {end_date})...")

    try:
        response = (
            client.table("ml_accuracy_details")
            .select("symbol, prediction_date, predicted_probability, "
                    "predicted_signal, outcome_type")
            .eq("outcome_type", "false_negative")
            .gte("prediction_date", start_date)
            .lte("prediction_date", end_date)
            .limit(limit)
            .execute()
        )
    except Exception:
        logger.warning("ml_accuracy_details not found, trying ml_prediction_accuracy...")
        response = (
            client.table("ml_prediction_accuracy")
            .select("symbol, prediction_date, predicted_probability, predicted_signal")
            .eq("prediction_correct", False)
            .eq("became_winner", True)
            .gte("prediction_date", start_date)
            .lte("prediction_date", end_date)
            .limit(limit)
            .execute()
        )

    if not response.data:
        logger.info("  No false negatives found in date range.")
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    df["mistake_type"] = "false_negative"
    df["label"] = 1
    df["sample_weight"] = WEIGHT_FALSE_NEGATIVE
    logger.info(f"  Found {len(df)} false negatives.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FETCH T-1 INDICATOR SNAPSHOTS FOR MISTAKE SYMBOLS
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_t1_snapshots_for_symbols(
    client: Client,
    table: str,
    symbol_date_pairs: list[tuple[str, str]],
    prefix: str,
) -> pd.DataFrame:
    """
    Fetch T-1 indicator rows for a list of (symbol, detection_date) pairs.

    Returns a DataFrame with columns renamed to model long-form names and
    prefixed by `prefix` (e.g. 't1_close_RSI_14').  rename_t1_columns() is
    applied so the feature names match what the model was trained on.
    """
    if not symbol_date_pairs:
        return pd.DataFrame()

    # Batch queries by date to stay within URL-length limits.
    # Each chunk sends ALL unique symbols for that date slice; since dates
    # don't overlap between chunks there can be no duplicate rows.
    dates   = list({pair[1] for pair in symbol_date_pairs})
    symbols = list({pair[0] for pair in symbol_date_pairs})

    rows = []
    for i in range(0, len(dates), 20):
        date_chunk = dates[i:i + 20]
        try:
            resp = (
                client.table(table)
                .select("*")
                .in_("detection_date", date_chunk)
                .in_("symbol", symbols)
                .execute()
            )
            if resp.data:
                rows.extend(resp.data)
        except Exception as e:
            logger.warning(f"Error fetching from {table}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Keep only the (symbol, detection_date) pairs we actually asked for
    pair_set = set(symbol_date_pairs)
    if "detection_date" in df.columns and "symbol" in df.columns:
        df = df[df.apply(
            lambda r: (r["symbol"], r["detection_date"]) in pair_set, axis=1
        )].copy()

    if df.empty:
        return pd.DataFrame()

    # ── Rename intraday short-form columns to model long-form names ──────────
    # This is the critical step that was previously missing:
    # without it, features come out as "t1_close_rsi" instead of "t1_close_RSI_14"
    # and every feature lookup in the model silently misses.
    if T1_MAP_AVAILABLE:
        df = rename_t1_columns(df, prefix=prefix)
    else:
        # Fallback: manual prefix on non-metadata columns (names won't match model)
        meta_cols = {"id", "created_at", "updated_at", "symbol", "exchange",
                     "detection_date", "snapshot_type", "snapshot_time", "snapshot_date"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df.rename(columns={c: f"{prefix}_{c}" for c in feature_cols})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_mistake_training_samples(
    lookback_days: int = 90,
    use_all_timepoints: bool = True,
    existing_features: list[str] = None,
    min_fp_confidence: float = FP_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Build a DataFrame of training samples derived from the model's own mistakes.

    Each row contains:
      - All T-1 open/close indicator features (same schema as retrain_model.py,
        with model long-form names via t1_column_map)
      - label  (0 = false positive,  1 = false negative)
      - sample_weight  (higher than standard samples — preserved by caller)
      - mistake_type   (string label for logging — excluded from feature matrix
                        by NON_FEATURE_COLS in ml_retrain_model.py)

    The caller (ml_retrain_model.py) should concatenate this with the
    standard combined_df AFTER combine_datasets() so that sample_weight
    values (3.0 / 2.0) are NOT overwritten by T1_WEIGHT.

    Args:
        lookback_days:      How many days back to search for mistakes.
        use_all_timepoints: Include both T-1 close AND T-1 open snapshots.
        existing_features:  Feature list from model_metadata.json (used to
                            ensure consistent column set). If None, raw
                            features are returned without padding.
        min_fp_confidence:  Only include false positives where the model
                            predicted with at least this probability.

    Returns:
        DataFrame of training samples, or empty DataFrame if no mistakes found.
    """
    client = _get_supabase_client()

    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_days)

    # ── 1. Fetch mistake records ──────────────────────────────────────────
    fp_df = fetch_false_positives(
        client, start_date.isoformat(), end_date.isoformat(), min_fp_confidence
    )
    fn_df = fetch_false_negatives(
        client, start_date.isoformat(), end_date.isoformat()
    )

    mistakes_df = pd.concat([fp_df, fn_df], ignore_index=True)

    if mistakes_df.empty:
        logger.info("No mistakes found — skipping mistake-learning step.")
        return pd.DataFrame()

    logger.info(f"Total mistakes to learn from: {len(mistakes_df)} "
                f"({len(fp_df)} FP, {len(fn_df)} FN)")

    # ── 2. Build (symbol, date) lookup ────────────────────────────────────
    # prediction_date from ml_accuracy_details == detection_date in T-1 tables
    # (both refer to the same trading session).
    fp_pairs = (
        [(r.symbol, r.prediction_date) for r in fp_df.itertuples()]
        if not fp_df.empty else []
    )
    fn_pairs = (
        [(r.symbol, r.prediction_date) for r in fn_df.itertuples()]
        if not fn_df.empty else []
    )

    # ── 3. Fetch T-1 close snapshots ──────────────────────────────────────
    logger.info("Fetching T-1 close snapshots for mistakes...")

    # False positives: NOT in winners table → check non_winners first
    fp_close_df = pd.DataFrame()
    if fp_pairs:
        fp_close_df = _fetch_t1_snapshots_for_symbols(
            client, "non_winners_day_prior_close", fp_pairs, "t1_close"
        )
        if fp_close_df.empty:
            logger.debug("  FP: not in non_winners_close, trying winners_close...")
            fp_close_df = _fetch_t1_snapshots_for_symbols(
                client, "winners_day_prior_close", fp_pairs, "t1_close"
            )
        logger.info(f"  FP T-1 close snapshots: {len(fp_close_df)}")

    # False negatives: ARE winners → check winners table first
    fn_close_df = pd.DataFrame()
    if fn_pairs:
        fn_close_df = _fetch_t1_snapshots_for_symbols(
            client, "winners_day_prior_close", fn_pairs, "t1_close"
        )
        if fn_close_df.empty:
            logger.debug("  FN: not in winners_close, trying non_winners_close...")
            fn_close_df = _fetch_t1_snapshots_for_symbols(
                client, "non_winners_day_prior_close", fn_pairs, "t1_close"
            )
        logger.info(f"  FN T-1 close snapshots: {len(fn_close_df)}")

    # ── 4. Fetch T-1 open snapshots (optional) ────────────────────────────
    fp_open_df = pd.DataFrame()
    fn_open_df = pd.DataFrame()
    if use_all_timepoints:
        logger.info("Fetching T-1 open snapshots for mistakes...")
        if fp_pairs:
            fp_open_df = _fetch_t1_snapshots_for_symbols(
                client, "non_winners_day_prior_open", fp_pairs, "t1_open"
            )
            logger.info(f"  FP T-1 open snapshots: {len(fp_open_df)}")
        if fn_pairs:
            fn_open_df = _fetch_t1_snapshots_for_symbols(
                client, "winners_day_prior_open", fn_pairs, "t1_open"
            )
            logger.info(f"  FN T-1 open snapshots: {len(fn_open_df)}")

    # ── 5. Merge snapshots with mistake metadata ───────────────────────────
    def _merge_mistake_with_snapshots(
        mistake_rows: pd.DataFrame,
        close_snapshots: pd.DataFrame,
        open_snapshots: pd.DataFrame,
        label: int,
        weight: float,
        mistake_type: str,
    ) -> list[dict]:
        """
        For each mistake row, combine its close + open snapshot into one
        training sample dict.  Skips rows where close data is missing.
        """
        if mistake_rows.empty or close_snapshots.empty:
            return []

        samples = []
        # Columns that carry no signal and should not become features
        skip_cols = {"id", "created_at", "updated_at", "snapshot_type",
                     "snapshot_time", "snapshot_date"}

        for _, mistake in mistake_rows.iterrows():
            symbol = mistake.get("symbol")
            date   = mistake.get("prediction_date")

            close_match = close_snapshots[
                (close_snapshots["symbol"] == symbol) &
                (close_snapshots["detection_date"] == date)
            ]
            if close_match.empty:
                continue

            sample: dict = {}

            # T-1 close features (already renamed to model long-form names)
            for col in close_match.columns:
                if col not in {"symbol", "detection_date"} and col not in skip_cols:
                    sample[col] = close_match.iloc[0][col]

            # T-1 open features (optional)
            if not open_snapshots.empty:
                open_match = open_snapshots[
                    (open_snapshots["symbol"] == symbol) &
                    (open_snapshots["detection_date"] == date)
                ]
                if not open_match.empty:
                    for col in open_match.columns:
                        if col not in {"symbol", "detection_date"} and col not in skip_cols:
                            sample[col] = open_match.iloc[0][col]

            sample["symbol"]         = symbol
            sample["detection_date"] = date
            sample["label"]          = label
            sample["sample_weight"]  = weight
            sample["mistake_type"]   = mistake_type
            samples.append(sample)

        return samples

    all_samples: list[dict] = []

    fp_samples = _merge_mistake_with_snapshots(
        fp_df, fp_close_df, fp_open_df,
        label=0, weight=WEIGHT_FALSE_POSITIVE, mistake_type="false_positive",
    )
    all_samples.extend(fp_samples)
    logger.info(f"Built {len(fp_samples)} false-positive training samples.")

    fn_samples = _merge_mistake_with_snapshots(
        fn_df, fn_close_df, fn_open_df,
        label=1, weight=WEIGHT_FALSE_NEGATIVE, mistake_type="false_negative",
    )
    all_samples.extend(fn_samples)
    logger.info(f"Built {len(fn_samples)} false-negative training samples.")

    if not all_samples:
        logger.warning("Could not build any mistake samples — T-1 snapshots missing.")
        return pd.DataFrame()

    result = pd.DataFrame(all_samples)

    # ── 6. Pad to full feature set ────────────────────────────────────────
    # Only pads columns that are genuinely missing; never overwrites real data.
    if existing_features:
        for feat in existing_features:
            if feat not in result.columns:
                feat_lower = feat.lower()
                if any(x in feat_lower for x in ["rsi", "stoch", "willr", "cci"]):
                    result[feat] = 50.0
                elif "volume" in feat_lower or "obv" in feat_lower:
                    result[feat] = 100_000.0
                elif any(x in feat_lower for x in ["price", "close", "open", "high", "low"]):
                    result[feat] = 50.0
                else:
                    result[feat] = 0.0

    logger.info(f"✓ Mistake training set: {len(result)} samples "
                f"({len(fp_samples)} FP, {len(fn_samples)} FN), "
                f"{len(result.columns)} columns")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def log_mistake_summary(mistake_df: pd.DataFrame) -> None:
    """Log a readable summary of what the model is learning from."""
    if mistake_df.empty:
        logger.info("No mistake data to summarise.")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("MISTAKE LEARNING SUMMARY")
    logger.info("=" * 60)

    for mtype in ["false_positive", "false_negative"]:
        subset = mistake_df[mistake_df["mistake_type"] == mtype]
        if subset.empty:
            continue
        label = (
            "False Positives (model said BUY, stock didn't explode)"
            if mtype == "false_positive"
            else "False Negatives (model said SKIP, stock exploded)"
        )
        logger.info(f"\n  {label}")
        logger.info(f"  Count : {len(subset)}")
        logger.info(f"  Weight: {subset['sample_weight'].iloc[0]}x standard samples")
        if "symbol" in subset.columns:
            symbols = subset["symbol"].unique().tolist()
            logger.info(
                f"  Symbols: {', '.join(symbols[:10])}"
                + (" ..." if len(symbols) > 10 else "")
            )

    logger.info("")
    logger.info(f"  Total mistake samples: {len(mistake_df)}")
    logger.info(f"  Effective weight vs standard: "
                f"{mistake_df['sample_weight'].mean():.1f}x")
    logger.info("=" * 60)
    logger.info("")
