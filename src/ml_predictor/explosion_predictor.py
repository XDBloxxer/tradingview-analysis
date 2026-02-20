"""
Explosion Predictor

FIXES:
  1. Dot-normalization in feature lookup: the model was trained with column names
     like t3_bbl_20_2_0_2_0 (dots replaced with underscores, lowercase) but the
     indicator calculator produces t3_BBL_20_2.0_2.0 (dots kept, uppercase).
     The previous case-insensitive lookup caught the case difference but NOT the
     dot→underscore difference, so all BB columns silently got neutral defaults.
     Fix: _build_lookup and prepare_features now normalize with .replace('.','_')
     in addition to .lower() so t3_BBL_20_2.0_2.0 correctly maps to t3_bbl_20_2_0_2_0.

  2. Bimodal collapse detection: predict() checks if the probability distribution
     has a gap in the 0.15–0.85 range and logs a clear warning.

  3. Adaptive signal thresholds: when bimodal collapse is detected, the classifier
     falls back to RELATIVE thresholds (percentile-based).

  4. Case-normalises feature lookup (retained from prior version).

  5. Scaler column-count fix (retained from prior version).
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import joblib

_META_COLS = {"symbol", "exchange"}

# Absolute thresholds used when the model is well-calibrated
SIGNAL_THRESHOLDS = {
    "STRONG BUY": 0.90,
    "BUY":        0.70,
    "HOLD":       0.50,
}

BIMODAL_MIDRANGE = (0.15, 0.85)
BIMODAL_MIN_MIDRANGE_COUNT = 5


def _norm(s: str) -> str:
    """Normalize a feature name for matching: lowercase + dots→underscores.

    This is the single source of truth for normalization so that column names
    produced by the indicator calculator (e.g. t3_BBL_20_2.0_2.0) correctly
    match what the model was trained on (e.g. t3_bbl_20_2_0_2_0).
    """
    return s.lower().replace(".", "_")


class ExplosionPredictor:

    def __init__(self, model_dir: str = "ml_models"):
        self.logger        = logging.getLogger(__name__)
        self.model_dir     = Path(model_dir)
        self.model         = None
        self.regressor     = None
        self.scaler        = None
        self.feature_names: List[str] = []
        self.metadata:      dict = {}
        self._norm_to_feature: Dict[str, str] = {}   # normalized → original model feature name
        self._regressor_n_features: Optional[int] = None
        self._is_bimodal   = False

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
            try:
                self._regressor_n_features = self.regressor.n_features_in_
            except AttributeError:
                try:
                    self._regressor_n_features = self.regressor.get_booster().num_features()
                except Exception:
                    self._regressor_n_features = None
            self.logger.info(
                f"✓ Loaded regressor (expects {self._regressor_n_features} features)"
            )
        else:
            self.regressor = None
            self.logger.warning("⚠ Regressor not found — will use rule-based gain estimates")

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.scaler.n_features_in_)]
            self.logger.warning("Scaler has no feature_names_in_ — using positional names.")

        self._build_lookup()

        classifier_n = self.scaler.n_features_in_
        self.logger.info(
            f"✓ Classifier/scaler expects {classifier_n} features; "
            f"regressor expects {self._regressor_n_features} features"
        )

        if (self._regressor_n_features is not None
                and self._regressor_n_features != classifier_n):
            self.logger.warning(
                f"⚠ Regressor feature count ({self._regressor_n_features}) differs from "
                f"classifier/scaler ({classifier_n}). "
                f"Regressor will be SKIPPED — rule-based gain estimates will be used. "
                f"Re-train the regressor with the same feature set to fix this."
            )
            self.regressor = None

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
        Build normalized(model_feature) → model_feature lookup for matching.

        FIX: Previously used only .lower() so t3_BBL_20_2.0_2.0 would NOT match
        t3_bbl_20_2_0_2_0 (dots vs underscores). Now uses _norm() which does
        both .lower() AND .replace('.','_'), fixing all Bollinger Band columns
        and any other indicator names with dots in them.
        """
        self._norm_to_feature = {_norm(f): f for f in self.feature_names}

    # ─────────────────────────────────────────────────────────────────────
    # Feature preparation
    # ─────────────────────────────────────────────────────────────────────

    def prepare_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align input data_df to the model's expected feature schema.

        Lookup order:
          1. Direct exact match (fastest, handles already-correct names)
          2. Normalized match via _norm() — handles case + dot differences
             e.g. t3_BBL_20_2.0_2.0 → t3_bbl_20_2_0_2_0
          3. Metadata columns (symbol/exchange) → 0.0
          4. Intelligent default based on indicator type
        """
        self.logger.info(f"Preparing features for {len(data_df)} stocks")

        # Build normalized lookup for input columns
        input_norm_to_col: Dict[str, str] = {_norm(c): c for c in data_df.columns}

        feature_data: dict = {}
        matched       = 0
        missing_names: List[str] = []

        for feature in self.feature_names:
            feature_norm = _norm(feature)

            if feature in data_df.columns:
                # 1. Exact match
                feature_data[feature] = data_df[feature].values
                matched += 1
            elif feature_norm in input_norm_to_col:
                # 2. Normalized match (handles case + dot→underscore differences)
                feature_data[feature] = data_df[input_norm_to_col[feature_norm]].values
                matched += 1
            elif feature_norm in _META_COLS:
                feature_data[feature] = 0.0
            else:
                feature_data[feature] = self._get_default_value(feature, data_df)
                missing_names.append(feature)

        feature_df = pd.DataFrame(feature_data, index=data_df.index)

        for col in ("symbol", "exchange"):
            if col in data_df.columns:
                feature_df[col] = data_df[col].values

        coverage = (matched / len(self.feature_names) * 100) if self.feature_names else 0
        self.logger.info(
            f"Feature coverage: {coverage:.1f}% ({matched}/{len(self.feature_names)})"
        )
        if missing_names:
            self.logger.info(f"First 30 MISSING features: {missing_names[:30]}")
            sample_avail = [c for c in data_df.columns if c not in _META_COLS][:30]
            self.logger.info(f"First 30 AVAILABLE columns: {sample_avail}")
        if coverage < 50:
            self.logger.warning(
                f"⚠️  LOW feature coverage ({coverage:.1f}%) — predictions may be unreliable"
            )

        return feature_df

    def _get_default_value(self, feature: str, data: pd.DataFrame):
        f = _norm(feature)
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
        """Apply the stored scaler while preserving NaN positions."""
        expected_n = self.scaler.n_features_in_
        if X.shape[1] != expected_n:
            raise ValueError(
                f"_scale_features: X has {X.shape[1]} columns but scaler expects "
                f"{expected_n}. len(feature_names)={len(self.feature_names)}."
            )

        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            self.logger.warning(f"Coercing non-numeric columns to NaN: {non_numeric}")
            X = X.copy()
            for c in non_numeric:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        nan_mask = X.isna()

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
        X_scaled[nan_mask] = np.nan

        return X_scaled

    # ─────────────────────────────────────────────────────────────────────
    # Bimodal detection & adaptive signal classification
    # ─────────────────────────────────────────────────────────────────────

    def _detect_bimodal(self, probabilities: np.ndarray) -> bool:
        mid_lo, mid_hi = BIMODAL_MIDRANGE
        mid_count = int(((probabilities > mid_lo) & (probabilities < mid_hi)).sum())
        if mid_count < BIMODAL_MIN_MIDRANGE_COUNT:
            self.logger.warning(
                f"⚠️  BIMODAL COLLAPSE DETECTED: only {mid_count} predictions in "
                f"{mid_lo*100:.0f}%–{mid_hi*100:.0f}% range "
                f"(out of {len(probabilities)} total). "
                f"Switching to percentile-based signal thresholds."
            )
            return True
        return False

    def _classify_signal_absolute(self, probability: float) -> str:
        if probability >= SIGNAL_THRESHOLDS["STRONG BUY"]:
            return "STRONG BUY"
        elif probability >= SIGNAL_THRESHOLDS["BUY"]:
            return "BUY"
        elif probability >= SIGNAL_THRESHOLDS["HOLD"]:
            return "HOLD"
        return "AVOID"

    def _classify_signals_relative(self, probabilities: pd.Series) -> pd.Series:
        n = len(probabilities)
        if n == 0:
            return pd.Series([], dtype=str)

        lo, hi = BIMODAL_MIDRANGE
        high_scores = probabilities[probabilities >= hi]

        signals = pd.Series("AVOID", index=probabilities.index)

        if len(high_scores) > 0:
            p90 = high_scores.quantile(0.90)
            p70 = high_scores.quantile(0.70)
            p50 = high_scores.quantile(0.50)

            signals.loc[high_scores.index] = high_scores.apply(
                lambda p: (
                    "STRONG BUY" if p >= p90 else
                    "BUY"        if p >= p70 else
                    "HOLD"
                )
            )

        self.logger.info(
            "Relative signal distribution: "
            + ", ".join(
                f"{s}={int((signals==s).sum())}"
                for s in ["STRONG BUY", "BUY", "HOLD", "AVOID"]
            )
        )
        return signals

    # ─────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, data_df: pd.DataFrame) -> pd.DataFrame:
        features_df = self.prepare_features(data_df)
        X           = features_df[self.feature_names].copy()
        X_scaled    = self._scale_features(X)

        predictions   = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        self._is_bimodal = self._detect_bimodal(probabilities)

        prob_series = pd.Series(probabilities, index=data_df.index)

        if self._is_bimodal:
            signals = self._classify_signals_relative(prob_series)
        else:
            signals = prob_series.apply(self._classify_signal_absolute)

        result_df = pd.DataFrame({
            "explosion_probability": probabilities,
            "prediction":            predictions,
            "signal":                signals.values,
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
            try:
                predicted_gains = self.regressor.predict(X_scaled)
                self.logger.info("Using LEARNED gain predictions from regressor")
                predictions["target_gain_pct"]  = predicted_gains
                predictions["target_gain_low"]   = predicted_gains * 0.8
                predictions["target_gain_high"]  = predicted_gains * 1.2
            except Exception as e:
                self.logger.warning(
                    f"Regressor predict failed ({e}) — falling back to rule-based estimates"
                )
                self.regressor = None

        if self.regressor is None:
            if historical_gains_df is not None and not historical_gains_df.empty:
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
        return self._classify_signal_absolute(probability)

    def _estimate_target_gain(self, probability: float) -> float:
        if probability >= 0.95:  return 30.0
        if probability >= 0.90:  return 25.0
        if probability >= 0.80:  return 20.0
        if probability >= 0.70:  return 15.0
        if probability >= 0.60:  return 10.0
        if probability >= 0.50:  return 7.0
        return 3.0
