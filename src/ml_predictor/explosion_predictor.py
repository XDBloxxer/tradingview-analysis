"""
Explosion Predictor - HYBRID MODEL VERSION
Works with models that know BOTH:
- T-3, T-5, T-10 (flat features from CSV: Close, RSI_14, MACD)
- T-1 open/close (prefixed features from database: t1_open_RSI_14, t1_close_MACD_12_26_9)
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import joblib

# Columns that must never be treated as model features
_META_COLS = {"symbol", "exchange"}


class ExplosionPredictor:
    """
    Hybrid explosion predictor.
    Works with models that expect BOTH old CSV features AND new T-1 split features.
    """

    def __init__(self, model_dir: str = "ml_models"):
        self.logger     = logging.getLogger(__name__)
        self.model_dir  = Path(model_dir)
        self.model      = None
        self.regressor  = None
        self.scaler     = None
        self.feature_names: List[str] = []
        self.metadata:  dict = {}

        self._load_model()

    # ─────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load trained classifier (and optional regressor)."""
        model_path     = self.model_dir / "best_model.pkl"
        regressor_path = self.model_dir / "gain_regressor.pkl"
        scaler_path    = self.model_dir / "scaler.pkl"
        metadata_path  = self.model_dir / "model_metadata.json"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model files not found in {self.model_dir}")

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if regressor_path.exists():
            self.regressor = joblib.load(regressor_path)
            self.logger.info("✓ Loaded classifier + regressor (dual-output mode)")
        else:
            self.regressor = None
            self.logger.warning("⚠ Regressor not found — will use rule-based gain estimates")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

            self.feature_names = (
                self.metadata.get("features")
                or self.metadata.get("feature_names_sample")
                or []
            )

            if not self.feature_names:
                if hasattr(self.scaler, "feature_names_in_"):
                    self.feature_names = list(self.scaler.feature_names_in_)
                else:
                    n = self.scaler.n_features_in_
                    self.feature_names = [f"feature_{i}" for i in range(n)]
                self.logger.warning(
                    f"'features' key missing from metadata — "
                    f"inferred {len(self.feature_names)} feature names from scaler"
                )
        else:
            if hasattr(self.scaler, "feature_names_in_"):
                self.feature_names = list(self.scaler.feature_names_in_)
            else:
                n = self.scaler.n_features_in_
                self.feature_names = [f"feature_{i}" for i in range(n)]
            self.logger.warning(
                f"No metadata file — inferred {len(self.feature_names)} features from scaler"
            )

        # ── GUARD: strip metadata columns from feature list ───────────────
        # If "symbol" or "exchange" somehow ended up in the saved feature list
        # (e.g. from a bug in a previous training run), remove them now so they
        # never reach scaler.transform().
        cleaned = [f for f in self.feature_names if f not in _META_COLS]
        if len(cleaned) != len(self.feature_names):
            removed = set(self.feature_names) - set(cleaned)
            self.logger.warning(
                f"Removed non-numeric columns from feature list: {removed}"
            )
            self.feature_names = cleaned

        self.logger.info(f"✓ Model expects {len(self.feature_names)} features")

        has_t1   = any("t1_open" in f or "t1_close" in f for f in self.feature_names)
        has_flat = any(f in {"Close", "RSI_14", "MACD_12_26_9"} for f in self.feature_names)
        if has_flat and has_t1:
            self.logger.info("✓ Model type: HYBRID (T-3/T-5/T-10 + T-1 open/close)")
        elif has_flat:
            self.logger.info("✓ Model type: CSV-ONLY (T-3/T-5/T-10)")
        elif has_t1:
            self.logger.info("✓ Model type: DATABASE-ONLY (T-1 open/close)")

    # ─────────────────────────────────────────────────────────────────────
    # Feature preparation
    # ─────────────────────────────────────────────────────────────────────

    def prepare_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align input data_df to the model's expected feature schema.

        Missing features receive intelligent defaults (not 0.0 for everything)
        to avoid systematic bias.  NaN is NOT filled here — XGBoost's native
        missing-value routing handles it correctly and should be preserved.

        NOTE: 'symbol' and 'exchange' are carried alongside the feature matrix
        as separate columns but are NEVER included in self.feature_names, so
        they never reach scaler.transform().
        """
        self.logger.info(f"Preparing features for {len(data_df)} stocks")

        feature_df = pd.DataFrame(index=data_df.index)

        # Preserve metadata columns alongside (not as features)
        for col in ("symbol", "exchange"):
            if col in data_df.columns:
                feature_df[col] = data_df[col]

        matched = 0
        missing = 0
        for feature in self.feature_names:
            # Paranoia check: skip any metadata column that slipped into feature_names
            if feature in _META_COLS:
                continue
            if feature in data_df.columns:
                feature_df[feature] = data_df[feature]
                matched += 1
            else:
                feature_df[feature] = self._get_default_value(feature, data_df)
                missing += 1

        coverage = (matched / len(self.feature_names) * 100) if self.feature_names else 0
        self.logger.info(
            f"Feature coverage: {coverage:.1f}% ({matched}/{len(self.feature_names)})"
        )
        if missing:
            self.logger.debug(f"Missing {missing} features — using intelligent defaults")
        if coverage < 50:
            self.logger.warning(
                f"⚠️  LOW feature coverage ({coverage:.1f}%) — predictions may be unreliable"
            )

        return feature_df

    def _get_default_value(self, feature: str, data: pd.DataFrame):
        """Return an intelligent scalar default for a feature not present in data."""
        f = feature.lower()
        for pfx in ("t1_open_", "t1_close_", "t3_", "t5_", "t10_"):
            if f.startswith(pfx):
                f = f[len(pfx):]
                break

        if any(x in f for x in ("rsi", "stoch", "willr", "cci")):
            return 50.0
        if any(x in f for x in ("change", "pct", "ratio", "slope", "diff", "roc", "mom", "macd", "ao")):
            return 0.0
        if any(x in f for x in ("above", "below", "cross", "flag", "signal")):
            return 0.0
        if "volume" in f or "obv" in f:
            for col in data.columns:
                if "volume" in col.lower() and col not in _META_COLS:
                    try:
                        return float(data[col].median())
                    except Exception:
                        pass
            return 100_000.0
        if any(x in f for x in ("price", "close", "open", "high", "low", "ema", "sma", "wma", "vwap")):
            for col in data.columns:
                if "close" in col.lower() and col not in _META_COLS:
                    try:
                        return float(data[col].median())
                    except Exception:
                        pass
            return 50.0
        if any(x in f for x in ("atr", "volatility", "hv", "bb")):
            return 1.0
        return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # NaN-safe scaling
    # ─────────────────────────────────────────────────────────────────────

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the stored scaler while preserving NaN positions.

        Defensive: drop any non-numeric / metadata columns that may have
        leaked into X before calling scaler.transform().
        """
        # ── Drop any metadata / non-numeric columns that must not be scaled ──
        cols_to_drop = [c for c in X.columns if c in _META_COLS]
        if cols_to_drop:
            self.logger.warning(
                f"_scale_features: dropping non-feature columns before scaling: "
                f"{cols_to_drop}"
            )
            X = X.drop(columns=cols_to_drop)

        # Ensure all remaining columns are numeric; coerce if needed
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                self.logger.warning(
                    f"_scale_features: column '{col}' is non-numeric "
                    f"(dtype={X[col].dtype}), coercing to NaN"
                )
                X = X.copy()
                X[col] = pd.to_numeric(X[col], errors="coerce")

        nan_mask = X.isna()

        col_means = pd.Series(self.scaler.mean_, index=X.columns)
        X_filled  = X.fillna(col_means)

        X_scaled_vals = self.scaler.transform(X_filled)
        X_scaled      = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
        X_scaled[nan_mask] = np.nan   # restore NaN for XGBoost missing-value routing

        return X_scaled

    # ─────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make explosion predictions on data_df.

        Returns:
            DataFrame sorted by explosion_probability descending.
        """
        features_df = self.prepare_features(data_df)

        # Select ONLY the model feature columns (never symbol/exchange)
        X        = features_df[self.feature_names].copy()
        X_scaled = self._scale_features(X)

        predictions   = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        result_df = pd.DataFrame({
            "explosion_probability": probabilities,
            "prediction":            predictions,
            "signal":                pd.Series(probabilities).apply(self._classify_signal),
        })

        for col in ("symbol", "exchange"):
            if col in features_df.columns:
                result_df.insert(0, col, features_df[col].values)

        return result_df.sort_values(
            "explosion_probability", ascending=False
        ).reset_index(drop=True)

    def predict_with_targets(
        self,
        data_df: pd.DataFrame,
        historical_gains_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Make predictions and attach target gain / price estimates.
        """
        predictions = self.predict(data_df)
        features_df = self.prepare_features(data_df)
        X           = features_df[self.feature_names].copy()
        X_scaled    = self._scale_features(X)

        # ── Gain prediction ───────────────────────────────────────────────
        if self.regressor is not None:
            self.logger.info("Using LEARNED gain predictions from regressor")
            predicted_gains = self.regressor.predict(X_scaled)
            predictions["target_gain_pct"]  = predicted_gains
            predictions["target_gain_low"]   = predicted_gains * 0.8
            predictions["target_gain_high"]  = predicted_gains * 1.2

        elif historical_gains_df is not None and not historical_gains_df.empty:
            self.logger.info("Using historical calibration for gain predictions")
            gain_buckets = historical_gains_df.copy()
            gain_buckets["prob_bucket"] = pd.cut(
                gain_buckets["predicted_probability"],
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=["Low", "Medium", "High", "Very High"],
            )
            avg_gains = gain_buckets.groupby("prob_bucket")["actual_gain_pct"].agg(
                ["mean", "median", "std"]
            )
            predictions["prob_bucket"] = pd.cut(
                predictions["explosion_probability"],
                bins=[0, 0.5, 0.7, 0.9, 1.0],
                labels=["Low", "Medium", "High", "Very High"],
            )
            predictions = predictions.merge(
                avg_gains, left_on="prob_bucket", right_index=True, how="left"
            )
            predictions["target_gain_pct"]  = predictions["median"]
            predictions["target_gain_low"]   = predictions["median"] - predictions["std"]
            predictions["target_gain_high"]  = predictions["median"] + predictions["std"]
            predictions = predictions.drop(["prob_bucket", "mean", "median", "std"], axis=1)

        else:
            self.logger.warning("Using rule-based gain estimates (no regressor or history)")
            predictions["target_gain_pct"] = predictions["explosion_probability"].apply(
                self._estimate_target_gain
            )
            predictions["target_gain_low"]  = predictions["target_gain_pct"] * 0.5
            predictions["target_gain_high"] = predictions["target_gain_pct"] * 1.5

        # Fill any remaining NaN gains
        nan_gain = predictions["target_gain_pct"].isna()
        if nan_gain.any():
            predictions.loc[nan_gain, "target_gain_pct"] = (
                predictions.loc[nan_gain, "explosion_probability"]
                .apply(self._estimate_target_gain)
            )
            predictions.loc[nan_gain, "target_gain_low"]  = (
                predictions.loc[nan_gain, "target_gain_pct"] * 0.5
            )
            predictions.loc[nan_gain, "target_gain_high"] = (
                predictions.loc[nan_gain, "target_gain_pct"] * 1.5
            )

        # ── Attach current price and target prices ────────────────────────
        if "symbol" in predictions.columns:
            close_col = next(
                (c for c in features_df.columns
                 if "close" in c.lower() and c not in _META_COLS),
                None,
            )
            if close_col:
                price_df = features_df[["symbol", close_col]].copy()
                price_df.columns = ["symbol", "close"]
                predictions = predictions.merge(price_df, on="symbol", how="left")
                predictions["current_price"]     = predictions["close"]
                predictions["target_price"]      = predictions["close"] * (1 + predictions["target_gain_pct"]  / 100)
                predictions["target_price_low"]  = predictions["close"] * (1 + predictions["target_gain_low"]  / 100)
                predictions["target_price_high"] = predictions["close"] * (1 + predictions["target_gain_high"] / 100)
                predictions = predictions.drop("close", axis=1)

        return predictions

    # ─────────────────────────────────────────────────────────────────────
    # Signal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _classify_signal(self, probability: float) -> str:
        if probability >= 0.90:
            return "STRONG BUY"
        elif probability >= 0.70:
            return "BUY"
        elif probability >= 0.50:
            return "HOLD"
        return "AVOID"

    def _estimate_target_gain(self, probability: float) -> float:
        if probability >= 0.95:  return 30.0
        if probability >= 0.90:  return 25.0
        if probability >= 0.80:  return 20.0
        if probability >= 0.70:  return 15.0
        if probability >= 0.60:  return 10.0
        if probability >= 0.50:  return 7.0
        return 3.0
