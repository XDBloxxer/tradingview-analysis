"""
Explosion Predictor — with diagnostic logging to find equal-probability root cause.

DIAGNOSTIC MODE: On first run, prepare_features() logs:
  - Per-group feature variance (t3_, t5_, t10_, t1_close_, t1_open_)
  - Sample raw values for 3 stocks on key features
  - Exact match path (direct / norm / default) for every feature
This tells us definitively whether features are zero-variance or mismatched.

FIXES INCLUDED:
  1. Dot-normalization: _norm() does .lower() + .replace('.','_') so
     t3_BBL_20_2.0_2.0 matches model feature t3_bbl_20_2_0_2_0.
  2. Bimodal collapse detection + relative signal thresholds.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import joblib

_META_COLS = {"symbol", "exchange"}

SIGNAL_THRESHOLDS = {
    "STRONG BUY": 0.90,
    "BUY":        0.70,
    "HOLD":       0.50,
}

BIMODAL_MIDRANGE = (0.15, 0.85)
BIMODAL_MIN_MIDRANGE_COUNT = 5


def _norm(s: str) -> str:
    """Normalize for matching: lowercase + dots to underscores."""
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
        self._norm_to_feature: Dict[str, str] = {}
        self._regressor_n_features: Optional[int] = None
        self._is_bimodal   = False
        self._diag_done    = False   # log diagnostics once per session

        self._load_model()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

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
            self.logger.info(f"Loaded regressor (expects {self._regressor_n_features} features)")
        else:
            self.regressor = None
            self.logger.warning("Regressor not found - will use rule-based gain estimates")

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.scaler.n_features_in_)]
            self.logger.warning("Scaler has no feature_names_in_ - using positional names.")

        self._build_lookup()

        classifier_n = self.scaler.n_features_in_
        self.logger.info(
            f"Classifier/scaler expects {classifier_n} features; "
            f"regressor expects {self._regressor_n_features} features"
        )

        if (self._regressor_n_features is not None
                and self._regressor_n_features != classifier_n):
            self.logger.warning(
                f"Regressor feature count ({self._regressor_n_features}) != "
                f"classifier ({classifier_n}). Regressor SKIPPED."
            )
            self.regressor = None

        has_t1   = any("t1_open" in f or "t1_close" in f for f in self.feature_names)
        has_flat = any(
            f.startswith("t3_") or f.startswith("t5_") or f.startswith("t10_")
            for f in self.feature_names
        )
        if has_flat and has_t1:
            self.logger.info("Model type: HYBRID (T-3/T-5/T-10 + T-1 open/close)")
        elif has_flat:
            self.logger.info("Model type: CSV-ONLY (T-3/T-5/T-10)")
        elif has_t1:
            self.logger.info("Model type: DATABASE-ONLY (T-1 open/close)")
        else:
            self.logger.warning("Could not identify model type. First 10 features: %s",
                                self.feature_names[:10])

    def _build_lookup(self):
        """normalized(model_feature) -> model_feature. Handles case + dot differences."""
        self._norm_to_feature = {_norm(f): f for f in self.feature_names}

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def _log_feature_diagnostics(self, feature_df: pd.DataFrame, match_log: dict):
        """
        Log per-group variance and sample values to find equal-probability root cause.
        Called once per session after first prepare_features().
        """
        self.logger.info("")
        self.logger.info("=" * 72)
        self.logger.info("FEATURE DIAGNOSTIC REPORT")
        self.logger.info("=" * 72)

        direct  = sum(1 for v in match_log.values() if v == "direct")
        norm_m  = sum(1 for v in match_log.values() if v == "norm")
        default = sum(1 for v in match_log.values() if v == "default")
        self.logger.info(
            f"Match breakdown: direct={direct}  norm={norm_m}  "
            f"default(constant)={default}  meta={len(match_log)-direct-norm_m-default}"
        )
        self.logger.info("")

        groups = [
            ("t1_close_", [c for c in feature_df.columns if c.startswith("t1_close_")]),
            ("t1_open_",  [c for c in feature_df.columns if c.startswith("t1_open_")]),
            ("t3_",       [c for c in feature_df.columns if c.startswith("t3_")]),
            ("t5_",       [c for c in feature_df.columns if c.startswith("t5_")]),
            ("t10_",      [c for c in feature_df.columns if c.startswith("t10_")]),
        ]

        for group_name, cols in groups:
            num_cols = [c for c in cols
                        if c in feature_df.columns
                        and pd.api.types.is_numeric_dtype(feature_df[c])]
            if not num_cols:
                self.logger.info(f"  {group_name:<14}  0 features")
                continue

            sub        = feature_df[num_cols].astype(float)
            col_stds   = sub.std()
            zero_var   = int((col_stds < 1e-9).sum())
            live_var   = int((col_stds >= 1e-9).sum())
            mean_std   = float(col_stds[col_stds >= 1e-9].mean()) if live_var > 0 else 0.0
            dflt_count = sum(1 for c in num_cols if match_log.get(c) == "default")

            self.logger.info(
                f"  {group_name:<14}  {len(num_cols):>3} cols | "
                f"zero-var: {zero_var:>3} | live-var: {live_var:>3} | "
                f"mean_std: {mean_std:.4f} | from_default: {dflt_count}"
            )

            # Sample live values for up to 5 varying features, 3 stocks
            live_cols = [c for c in num_cols if col_stds.get(c, 0) >= 1e-9][:5]
            if live_cols and len(feature_df) >= 3:
                for col in live_cols:
                    vals = feature_df[col].astype(float).iloc[:3].tolist()
                    self.logger.info(
                        f"    sample {col}: "
                        + "  ".join(f"{v:>10.4f}" for v in vals)
                    )

        self.logger.info("")

        # Cross-reference with feature_importance.csv
        fi_path = self.model_dir / "feature_importance.csv"
        if fi_path.exists():
            try:
                fi = pd.read_csv(fi_path).head(20)
                self.logger.info("  Top 20 model features by importance vs live data:")
                self.logger.info(f"  {'Feature':<45} {'Imp':>6}  {'Std':>8}  {'Src'}")
                self.logger.info("  " + "-" * 72)
                for _, row in fi.iterrows():
                    feat      = row["feature"]
                    imp       = row["importance"]
                    feat_norm = _norm(feat)
                    # Find this feature in the dataframe
                    feat_col  = None
                    if feat in feature_df.columns:
                        feat_col = feat
                    else:
                        # Try norm match
                        for col in feature_df.columns:
                            if _norm(col) == feat_norm:
                                feat_col = col
                                break
                    if feat_col is not None and feat_col in feature_df.columns:
                        std_val = float(feature_df[feat_col].astype(float).std())
                        src     = match_log.get(feat_col, match_log.get(feat, "?"))
                    else:
                        std_val = float("nan")
                        src     = "MISSING"
                    self.logger.info(
                        f"  {feat:<45} {imp:>6.4f}  {std_val:>8.4f}  {src}"
                    )
            except Exception as e:
                self.logger.warning(f"Could not read feature_importance.csv: {e}")

        self.logger.info("=" * 72)
        self.logger.info("")

    # -------------------------------------------------------------------------
    # Feature preparation
    # -------------------------------------------------------------------------

    def prepare_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align input data_df to the model's expected feature schema.
        Lookup: direct match -> norm match (_norm handles case+dots) -> default.
        """
        self.logger.info(f"Preparing features for {len(data_df)} stocks")

        input_norm_to_col: Dict[str, str] = {_norm(c): c for c in data_df.columns}

        feature_data: dict = {}
        match_log:    dict = {}
        matched       = 0
        missing_names: List[str] = []

        for feature in self.feature_names:
            feature_norm = _norm(feature)

            if feature in data_df.columns:
                feature_data[feature] = data_df[feature].values
                match_log[feature]    = "direct"
                matched += 1
            elif feature_norm in input_norm_to_col:
                feature_data[feature] = data_df[input_norm_to_col[feature_norm]].values
                match_log[feature]    = "norm"
                matched += 1
            elif feature_norm in _META_COLS:
                feature_data[feature] = 0.0
                match_log[feature]    = "meta"
            else:
                feature_data[feature] = self._get_default_value(feature, data_df)
                match_log[feature]    = "default"
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
            self.logger.warning(f"LOW feature coverage ({coverage:.1f}%)")

        # Diagnostics on first call only
        if not self._diag_done:
            self._log_feature_diagnostics(feature_df, match_log)
            self._diag_done = True

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

    # -------------------------------------------------------------------------
    # NaN-safe scaling
    # -------------------------------------------------------------------------

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        expected_n = self.scaler.n_features_in_
        if X.shape[1] != expected_n:
            raise ValueError(
                f"_scale_features: X has {X.shape[1]} cols but scaler expects {expected_n}.")

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

    # -------------------------------------------------------------------------
    # Bimodal detection & adaptive signals
    # -------------------------------------------------------------------------

    def _detect_bimodal(self, probabilities: np.ndarray) -> bool:
        mid_lo, mid_hi = BIMODAL_MIDRANGE
        mid_count = int(((probabilities > mid_lo) & (probabilities < mid_hi)).sum())
        if mid_count < BIMODAL_MIN_MIDRANGE_COUNT:
            self.logger.warning(
                f"BIMODAL COLLAPSE: only {mid_count} predictions in mid-range. "
                f"Switching to percentile-based signals."
            )
            return True
        return False

    def _classify_signal_absolute(self, probability: float) -> str:
        if probability >= SIGNAL_THRESHOLDS["STRONG BUY"]: return "STRONG BUY"
        elif probability >= SIGNAL_THRESHOLDS["BUY"]:       return "BUY"
        elif probability >= SIGNAL_THRESHOLDS["HOLD"]:      return "HOLD"
        return "AVOID"

    def _classify_signals_relative(self, probabilities: pd.Series) -> pd.Series:
        lo, hi      = BIMODAL_MIDRANGE
        high_scores = probabilities[probabilities >= hi]
        signals     = pd.Series("AVOID", index=probabilities.index)
        if len(high_scores) > 0:
            p90 = high_scores.quantile(0.90)
            p70 = high_scores.quantile(0.70)
            signals.loc[high_scores.index] = high_scores.apply(
                lambda p: "STRONG BUY" if p >= p90 else "BUY" if p >= p70 else "HOLD"
            )
        return signals

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(self, data_df: pd.DataFrame) -> pd.DataFrame:
        features_df = self.prepare_features(data_df)
        X           = features_df[self.feature_names].copy()
        X_scaled    = self._scale_features(X)

        # Log scaled variance to confirm differentiation before model call
        scaled_std    = X_scaled.std()
        zero_var_cols = int((scaled_std < 1e-9).sum())
        self.logger.info(
            f"Scaled matrix: {X_scaled.shape} | "
            f"zero-var cols: {zero_var_cols} | "
            f"mean_std: {scaled_std.mean():.4f}"
        )
        if zero_var_cols > len(self.feature_names) * 0.5:
            self.logger.warning(
                f"MORE THAN HALF of features are zero-variance after scaling. "
                f"This will cause equal probabilities. "
                f"Check the diagnostic report above for which groups are affected."
            )

        predictions   = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        self.logger.info(
            f"Probability spread: min={probabilities.min():.4f}  "
            f"max={probabilities.max():.4f}  "
            f"std={probabilities.std():.6f}"
        )

        self._is_bimodal = self._detect_bimodal(probabilities)

        prob_series = pd.Series(probabilities, index=data_df.index)
        signals     = (self._classify_signals_relative(prob_series) if self._is_bimodal
                       else prob_series.apply(self._classify_signal_absolute))

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

        if self.regressor is not None:
            try:
                predicted_gains = self.regressor.predict(X_scaled)
                self.logger.info("Using LEARNED gain predictions from regressor")
                predictions["target_gain_pct"]  = predicted_gains
                predictions["target_gain_low"]   = predicted_gains * 0.8
                predictions["target_gain_high"]  = predicted_gains * 1.2
            except Exception as e:
                self.logger.warning(f"Regressor predict failed ({e}) - falling back to rule-based")
                self.regressor = None

        if self.regressor is None:
            if historical_gains_df is not None and not historical_gains_df.empty:
                gain_buckets = historical_gains_df.copy()
                gain_buckets["prob_bucket"] = pd.cut(
                    gain_buckets["predicted_probability"],
                    bins=[0, 0.5, 0.7, 0.9, 1.0],
                    labels=["Low", "Medium", "High", "Very High"],
                )
                avg_gains = gain_buckets.groupby("prob_bucket")["actual_gain_pct"].agg(
                    ["mean", "median", "std"])
                predictions["prob_bucket"] = pd.cut(
                    predictions["explosion_probability"],
                    bins=[0, 0.5, 0.7, 0.9, 1.0],
                    labels=["Low", "Medium", "High", "Very High"],
                )
                predictions = predictions.merge(
                    avg_gains, left_on="prob_bucket", right_index=True, how="left")
                predictions["target_gain_pct"]  = predictions["median"]
                predictions["target_gain_low"]   = predictions["median"] - predictions["std"]
                predictions["target_gain_high"]  = predictions["median"] + predictions["std"]
                predictions = predictions.drop(["prob_bucket", "mean", "median", "std"], axis=1)
            else:
                self.logger.warning("Using rule-based gain estimates")
                predictions["target_gain_pct"]  = predictions["explosion_probability"].apply(
                    self._estimate_target_gain)
                predictions["target_gain_low"]   = predictions["target_gain_pct"] * 0.5
                predictions["target_gain_high"]  = predictions["target_gain_pct"] * 1.5

        nan_gain = predictions["target_gain_pct"].isna()
        if nan_gain.any():
            predictions.loc[nan_gain, "target_gain_pct"] = (
                predictions.loc[nan_gain, "explosion_probability"]
                .apply(self._estimate_target_gain))
            predictions.loc[nan_gain, "target_gain_low"]  = (
                predictions.loc[nan_gain, "target_gain_pct"] * 0.5)
            predictions.loc[nan_gain, "target_gain_high"] = (
                predictions.loc[nan_gain, "target_gain_pct"] * 1.5)

        if "symbol" in predictions.columns:
            close_col = next(
                (c for c in features_df.columns
                 if "close" in c.lower() and c not in _META_COLS), None)
            if close_col:
                price_df = features_df[["symbol", close_col]].copy()
                price_df.columns = ["symbol", "close"]
                predictions = predictions.merge(price_df, on="symbol", how="left")
                predictions["current_price"]     = predictions["close"]
                predictions["target_price"]      = predictions["close"] * (1 + predictions["target_gain_pct"] / 100)
                predictions["target_price_low"]  = predictions["close"] * (1 + predictions["target_gain_low"] / 100)
                predictions["target_price_high"] = predictions["close"] * (1 + predictions["target_gain_high"] / 100)
                predictions = predictions.drop("close", axis=1)

        return predictions

    def _classify_signal(self, probability: float) -> str:
        return self._classify_signal_absolute(probability)

    def _estimate_target_gain(self, probability: float) -> float:
        if probability >= 0.95: return 30.0
        if probability >= 0.90: return 25.0
        if probability >= 0.80: return 20.0
        if probability >= 0.70: return 15.0
        if probability >= 0.60: return 10.0
        if probability >= 0.50: return 7.0
        return 3.0

