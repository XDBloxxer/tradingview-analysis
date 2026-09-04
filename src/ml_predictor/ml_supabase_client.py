"""
ML Supabase Client - COMPLETE VERSION
Handles ML prediction storage, accuracy tracking, and screening logs
"""

import logging
import os
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from supabase import create_client, Client


class MLPredictionSupabaseClient:
    """
    Client for storing and retrieving ML predictions, accuracy tracking,
    and screening logs.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase URL and KEY must be provided via SUPABASE_URL / SUPABASE_KEY "
                "environment variables."
            )

        self.client: Client = create_client(supabase_url, supabase_key)

        self.predictions_table   = "ml_explosion_predictions"
        self.accuracy_table      = "ml_prediction_accuracy"
        self.screening_log_table = "ml_screening_logs"

        self.logger.info("ML Supabase client initialized")

    # Columns that exist in the ml_explosion_predictions Supabase table.
    # Any key NOT in this set is stripped before upsert to prevent
    # PGRST204 "column not found in schema cache" errors when the code adds
    # new fields (e.g. gain_source) ahead of a table migration being applied.
    # Update this set whenever a new column is added to the table.
    _PREDICTIONS_COLUMNS: frozenset = frozenset({
        "symbol", "exchange", "prediction_date",
        "explosion_probability", "prediction", "signal",
        "target_gain_pct", "target_gain_low", "target_gain_high",
        "current_price", "target_price", "target_price_low", "target_price_high",
        "rsi", "macd", "adx", "volume_ratio", "hv_20", "bb_width",
        "model_version", "screening_universe",
    })

    def write_predictions_upsert(self, predictions: list) -> int:
        """
        Upsert predictions — overwrites existing rows for the same
        (symbol, prediction_date) instead of skipping them.
        Fixes re-runs appearing to store 0 records.

        Keys not present in _PREDICTIONS_COLUMNS are stripped before the
        upsert so that code-side fields added ahead of a schema migration
        (e.g. gain_source) never cause a PGRST204 error.
        """
        if not predictions:
            return 0
        try:
            sanitized = [
                {k: v for k, v in self._sanitize_dict(p).items()
                 if k in self._PREDICTIONS_COLUMNS}
                for p in predictions
            ]

            # Warn once if any keys were dropped so it's easy to spot in logs.
            all_keys = {k for p in predictions for k in p}
            extra    = all_keys - self._PREDICTIONS_COLUMNS
            if extra:
                self.logger.warning(
                    f"write_predictions_upsert: dropping {len(extra)} key(s) not in "
                    f"ml_explosion_predictions schema: {sorted(extra)}. "
                    "Add them to _PREDICTIONS_COLUMNS once the Supabase migration is applied."
                )

            response = (
                self.client.table(self.predictions_table)
                .upsert(sanitized, on_conflict="symbol,prediction_date")
                .execute()
            )
            count = len(response.data) if response.data else 0
            self.logger.info(f"✓ Upserted {count} predictions")
            return count
        except Exception as e:
            self.logger.error(f"Failed to upsert predictions: {e}", exc_info=True)
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Sanitisation helpers
    # ─────────────────────────────────────────────────────────────────────

    def _sanitize_value(self, value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            if np.isinf(value) or np.isnan(value):
                return None
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, float):
            if np.isinf(value) or np.isnan(value):
                return None
        return value

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._sanitize_value(v) for k, v in data.items()}

    # ─────────────────────────────────────────────────────────────────────
    # Predictions
    # ─────────────────────────────────────────────────────────────────────

    def write_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """
        Write ML predictions to database, skipping duplicates.

        Returns:
            Number of new records written.
        """
        if not predictions:
            self.logger.warning("No predictions to write")
            return 0

        try:
            prediction_date = predictions[0].get("prediction_date")
            symbols         = [p["symbol"] for p in predictions]

            existing = (
                self.client.table(self.predictions_table)
                .select("symbol")
                .eq("prediction_date", prediction_date)
                .in_("symbol", symbols)
                .execute()
            )
            existing_symbols = {r["symbol"] for r in existing.data} if existing.data else set()

            new_predictions = [p for p in predictions if p["symbol"] not in existing_symbols]

            if existing_symbols:
                self.logger.info(
                    f"Skipping {len(existing_symbols)} predictions that already exist"
                )
            if not new_predictions:
                self.logger.info("No new predictions to write")
                return 0

            sanitized = [self._sanitize_dict(pred) for pred in new_predictions]
            response  = self.client.table(self.predictions_table).insert(sanitized).execute()
            count     = len(response.data) if response.data else 0
            self.logger.info(f"✓ Wrote {count} predictions to database")
            return count

        except Exception as e:
            self.logger.error(f"Failed to write predictions: {e}", exc_info=True)
            raise

    def read_predictions(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
        symbol:     Optional[str] = None,
        min_probability: Optional[float] = None,
    ) -> pd.DataFrame:
        try:
            query = self.client.table(self.predictions_table).select("*")
            if start_date:
                query = query.gte("prediction_date", start_date)
            if end_date:
                query = query.lte("prediction_date", end_date)
            if symbol:
                query = query.eq("symbol", symbol)
            if min_probability is not None:
                query = query.gte("explosion_probability", min_probability)
            response = query.execute()
            if not response.data:
                return pd.DataFrame()
            df = pd.DataFrame(response.data)
            self.logger.info(f"Retrieved {len(df)} predictions")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read predictions: {e}", exc_info=True)
            return pd.DataFrame()

    def get_predictions_for_date(self, prediction_date: str) -> pd.DataFrame:
        return self.read_predictions(
            start_date=prediction_date, end_date=prediction_date
        )

    # ─────────────────────────────────────────────────────────────────────
    # Accuracy tracking
    # ─────────────────────────────────────────────────────────────────────

    def write_accuracy_records(self, accuracy_records: List[Dict[str, Any]]) -> int:
        if not accuracy_records:
            return 0
        try:
            sanitized = [self._sanitize_dict(rec) for rec in accuracy_records]
            response  = (
                self.client.table(self.accuracy_table)
                .upsert(sanitized, on_conflict="symbol,prediction_date")
                .execute()
            )
            count = len(response.data) if response.data else 0
            self.logger.info(f"✓ Wrote {count} accuracy records")
            return count
        except Exception as e:
            self.logger.error(f"Failed to write accuracy records: {e}", exc_info=True)
            raise

    def write_records_upsert(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        on_conflict: str,
    ) -> int:
        """
        Generic sanitized upsert for any table.

        BUGFIX (2026-09): ml_track_comprehensive_accuracy.py used to call
        self.client.table(...).upsert(...) directly for the
        ml_accuracy_details and ml_missed_opportunities tables, bypassing
        _sanitize_dict(). A single NaN/Infinity float anywhere in the batch
        (e.g. predicted_probability for a row yfinance couldn't resolve)
        makes postgrest-py's JSON encoder raise "Out of range float values
        are not JSON compliant" for the *entire* upsert — so all rows in
        the batch silently fail to write, not just the offending one. This
        is why downstream tables could show nulls/gaps almost everywhere
        even though only one symbol's data was actually bad.

        Every non-prediction-table write should go through this method
        (or otherwise call _sanitize_dict itself) instead of hitting
        self.client.table(...) directly.
        """
        if not records:
            return 0
        try:
            sanitized = [self._sanitize_dict(rec) for rec in records]
            response = (
                self.client.table(table_name)
                .upsert(sanitized, on_conflict=on_conflict)
                .execute()
            )
            count = len(response.data) if response.data else 0
            self.logger.info(f"✓ Wrote {count} record(s) to {table_name}")
            return count
        except Exception as e:
            self.logger.error(f"Failed to write records to {table_name}: {e}", exc_info=True)
            raise

    def get_historical_prediction_accuracy(
        self,
        days_back: int = 30,
        min_probability: float = 0.5,
    ) -> pd.DataFrame:
        """
        Fetch historical predictions that have been resolved (became_winner is not null).
        Used for calibrating gain estimates in explosion_predictor.
        """
        try:
            start_date = (
                datetime.now().date() - timedelta(days=days_back)
            ).isoformat()

            # Correct Supabase Python syntax for "column IS NOT NULL":
            #   .not_.is_("column", "null")
            response = (
                self.client.table(self.accuracy_table)
                .select("*")
                .gte("prediction_date", start_date)
                .gte("predicted_probability", min_probability)
                .not_.is_("became_winner", "null")
                .execute()
            )

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            if "predicted_probability" in df.columns:
                df["probability"] = df["predicted_probability"]
            if "actual_gain_pct" in df.columns:
                df["actual_gain_pct"] = df["actual_gain_pct"].fillna(0)
            return df

        except Exception as e:
            self.logger.error(f"Failed to get historical accuracy: {e}")
            return pd.DataFrame()

    def get_prediction_accuracy_stats(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            query = self.client.table(self.accuracy_table).select("*")
            if start_date:
                query = query.gte("prediction_date", start_date)
            if end_date:
                query = query.lte("prediction_date", end_date)
            response = query.execute()
            if not response.data:
                return {"error": "No accuracy data found"}

            df      = pd.DataFrame(response.data)
            total   = len(df)
            correct = df["prediction_correct"].sum()
            winners = df[df["became_winner"] == True]
            predicted_winners = df[df["predicted_probability"] >= 0.5]
            tp  = len(predicted_winners[predicted_winners["became_winner"] == True])
            fp  = len(predicted_winners[predicted_winners["became_winner"] == False])
            fn  = len(winners[winners["predicted_probability"] < 0.5])
            precision = tp / len(predicted_winners) if len(predicted_winners) > 0 else 0
            recall    = tp / len(winners)            if len(winners)            > 0 else 0
            f1        = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            return {
                "total_predictions": total,
                "correct_predictions": int(correct),
                "accuracy": correct / total if total > 0 else 0,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "total_actual_winners": len(winners),
                "predicted_winners": len(predicted_winners),
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy stats: {e}")
            return {"error": str(e)}

    # ─────────────────────────────────────────────────────────────────────
    # Screening log
    # ─────────────────────────────────────────────────────────────────────

    def write_screening_log(self, log_data: Dict[str, Any]) -> bool:
        try:
            sanitized = self._sanitize_dict(log_data)
            self.client.table(self.screening_log_table)\
                .upsert(sanitized, on_conflict="screening_date")\
                .execute()
            self.logger.info("✓ Wrote screening log")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write screening log: {e}")
            return False

    def get_recent_screening_logs(self, limit: int = 10) -> pd.DataFrame:
        try:
            response = (
                self.client.table(self.screening_log_table)
                .select("*")
                .order("screening_date", desc=True)
                .limit(limit)
                .execute()
            )
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to get screening logs: {e}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────
    # T-1 snapshot helpers (used by model_trainer / accuracy tracker)
    # ─────────────────────────────────────────────────────────────────────

    def _fetch_snapshot_table(
        self,
        table: str,
        start_date: Optional[str],
        end_date:   Optional[str],
        limit: int,
    ) -> pd.DataFrame:
        try:
            query = self.client.table(table).select("*")
            if start_date:
                query = query.gte("detection_date", start_date)
            if end_date:
                query = query.lte("detection_date", end_date)
            query    = query.limit(limit)
            response = query.execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error reading {table}: {e}")
            return pd.DataFrame()

    def get_winners_day_prior_close(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        return self._fetch_snapshot_table(
            "winners_day_prior_close", start_date, end_date, limit
        )

    def get_winners_day_prior_open(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        return self._fetch_snapshot_table(
            "winners_day_prior_open", start_date, end_date, limit
        )

    def get_non_winners_day_prior_close(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        return self._fetch_snapshot_table(
            "non_winners_day_prior_close", start_date, end_date, limit
        )

    def get_non_winners_day_prior_open(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        return self._fetch_snapshot_table(
            "non_winners_day_prior_open", start_date, end_date, limit
        )

    def get_daily_winners(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ) -> pd.DataFrame:
        try:
            query = self.client.table("daily_winners").select(
                "symbol,detection_date,change_pct,price,volume,high,low,open,close"
            )
            if start_date:
                query = query.gte("detection_date", start_date)
            if end_date:
                query = query.lte("detection_date", end_date)
            response = query.execute()
            return pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error reading daily winners: {e}")
            return pd.DataFrame()
