"""
Explosion Predictor - HYBRID MODEL VERSION

KEY FIXES:
1. Case-normalises feature lookup: model trained with lowercase (t3_close, t3_sma_5)
   but fetcher produces mixed-case (t3_Close, t3_SMA_5).  A case-insensitive lookup
   map is built once at load time so all 413 features resolve correctly.
2. Scaler column-count fix: if the scaler has one extra column that is a known
   meta/string column (e.g. 'exchange'), it is kept in the matrix as 0.0 so the
   count always matches — it is never stripped.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import joblib

# Columns that are metadata, not numeric features.
# If they appear in the scaler they will be filled with 0.0.
_META_COLS = {"symbol", "exchange"}


class ExplosionPredictor:

    def __init__(self, model_dir: str = "ml_models"):
        self.logger        = logging.getLogger(__name__)
        self.model_dir     = Path(model_dir)
        self.model         = None
        self.regressor     = None
        self.scaler        = None
        self.feature_names: List[str] = []   # exact names the scaler expects
        self.metadata:      dict = {}

        # Built in _build_lookup(): lowercase(feature) -> exact feature name
        self._lower_to_feature: Dict[str, str] = {}

        self._load_model()

    # ─────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_model(self):
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
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        # ── Feature list: scaler is ground truth ─────────────────────────
        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.scaler.n_features_in_)]
            self.logger.warning(
                "Scaler has no feature_names_in_ — using positional names. "
                "Column order must match training exactly."
            )

        self._build_lookup()

        self.logger.info(
            f"✓ Model expects {len(self.feature_names)} features "
            f"(scaler n_features_in_={self.scaler.n_features_in_})"
        )

        has_t1   = any("t1_open" in f or "t1_close" in f for f in self.feature_names)
        has_flat = any(f.lower() in {"close", "rsi_14", "macd_12_26_9"} for f in self.feature_names)
        if has_flat and has_t1:
            self.logger.info("✓ Model type: HYBRID (T-3/T-5/T-10 + T-1 open/close)")
        elif has_flat:
            self.logger.info("✓ Model type: CSV-ONLY (T-3/T-5/T-10)")
        elif has_t1:
            self.logger.info("✓ Model type: DATABASE-ONLY (T-1 open/close)")
        else:
            self.logger.warning(
                "⚠ Could not identify model type. First 10 features: %s",
                self.feature_names[:10],
            )

    def _build_lookup(self):
        """
        Build a case-insensitive lookup: lowercase(model_feature) -> model_feature.

        The model was trained on lowercase column names (t3_close, t3_sma_5, …)
        but fetch_stock_data_for_prediction produces mixed-case names
        (t3_Close, t3_SMA_5, …).  This map lets us resolve both.
        """
        self._lower_to_feature = {f.lower(): f for f in self.feature_names}

    # ─────────────────────────────────────────────────────────────────────
    # Feature preparation
    # ─────────────────────────────────────────────────────────────────────

    def prepare_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align input data_df to the model's expected feature schema.

        Uses a case-insensitive lookup so t3_Close matches t3_close, etc.
        Builds all columns at once via pd.DataFrame(dict) to avoid the
        fragmentation PerformanceWarning.
        """
        self.logger.info(f"Preparing features for {len(data_df)} stocks")

        # Build a lowercase → actual-column-name map for the INPUT data
        input_lower: Dict[str, str] = {c.lower(): c for c in data_df.columns}

        feature_data: dict = {}
        matched      = 0
        missing_names: List[str] = []

        for feature in self.feature_names:
            feature_lower = feature.lower()

            # 1. Exact match
            if feature in data_df.columns:
                feature_data[feature] = data_df[feature].values
                matched += 1

            # 2. Case-insensitive match
            elif feature_lower in input_lower:
                src_col = input_lower[feature_lower]
                feature_data[feature] = data_df[src_col].values
                matched += 1

            # 3. Known meta column — fill with 0.0 (keeps scaler count correct)
            elif feature_lower in _META_COLS:
                feature_data[feature] = 0.0
                # Don't count as missing — it's intentionally numeric-zeroed

            # 4. Truly missing — use intelligent default
            else:
                feature_data[feature] = self._get_default_value(feature, data_df)
                missing_names.append(feature)

        # Build all columns at once (avoids fragmentation warning)
        feature_df = pd.DataFrame(feature_data, index=data_df.index)

        # Re-attach metadata as extra columns (never fed to the scaler)
        for col in ("symbol", "exchange"):
            if col in data_df.columns:
                feature_df[col] = data_df[col].values

        coverage = (matched / len(self.feature_names) * 100) if self.feature_names else 0
        self.logger.info(
            f"Feature coverage: {coverage:.1f}% ({matched}/{len(self.feature_names)})"
        )

        if missing_names:
            self.logger.debug(f"Missing {len(missing_names)} features — using intelligent defaults")

        if coverage < 50:
            self.logger.warning(
                f"⚠️  LOW feature coverage ({coverage:.1f}%) — predictions may be unreliable"
            )
            sample_missing   = missing_names[:20]
            sample_available = [c for c in data_df.columns if c not in _META_COLS][:20]
            self.logger.warning(f"   Sample MISSING  features : {sample_missing}")
            self.logger.warning(f"   Sample AVAILABLE columns : {sample_available}")

        return feature_df

    def _get_default_value(self, feature: str, data: pd.DataFrame):
        """Return an intelligent scalar default for a missing feature."""
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
        X must contain exactly self.feature_names columns (no extra meta cols).
        """
        expected_n = self.scaler.n_features_in_
        if X.shape[1] != expected_n:
            raise ValueError(
                f"_scale_features: X has {X.shape[1]} columns but scaler expects "
                f"{expected_n}. len(feature_names)={len(self.feature_names)}."
            )

        # Coerce any non-numeric columns to NaN
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            self.logger.warning(f"Coercing non-numeric columns to NaN: {non_numeric}")
            X = X.copy()
            for c in non_numeric:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        nan_mask = X.isna()

        # Build mean Series aligned to X.columns
        if hasattr(self.scaler, "feature_names_in_"):
            mean_series = (
                pd.Series(self.scaler.mean_, index=list(self.scaler.feature_names_in_))
                .reindex(X.columns)
            )
        else:
            mean_series = pd.Series(self.scaler.mean_, index=X.columns)

        X_filled      = X.fillna(mean_series)
        X_scaled_vals = self.scaler.transform(X_filled)
        X_scaled      = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
        X_scaled[nan_mask] = np.nan   # restore NaN for XGBoost missing-value routing

        return X_scaled

    # ─────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, data_df: pd.DataFrame) -> pd.DataFrame:
        features_df = self.prepare_features(data_df)

        # Feed ONLY the model feature columns to the scaler
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
