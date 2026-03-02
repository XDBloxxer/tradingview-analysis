"""
Explosion Predictor

FIXES IN THIS VERSION (2026-03-02 v4):

FIX 1 — Auto-detect which feature prefix the loaded model uses:
  The previous version hard-coded t1_close_ in the screen script, but the
  model stored in ml_models/ may have been trained on t3_/t5_/t10_ features
  (base CSV). When the prefixes don't match, prepare_features() fills
  everything with defaults → identical probabilities for every stock.

  Fix: ExplosionPredictor now exposes `model_feature_prefix` property that
  returns the dominant prefix ("t1_close", "t3", etc.) detected from the
  model's own feature_names list. ml_screen_and_predict reads this and
  uses the correct prefix when building features from TV data.

FIX 2 — Bimodal/compression detection now also handles the case where
  all probabilities are in a NARROW band above 0.50 (e.g. 0.70–0.75):
  The previous compression check only looked at probs < 0.50. After the
  prefix fix, if the model IS getting real features but they're all similar,
  probabilities cluster near 0.73. This triggered no fallback → all BUY.
  
  Fix: Also detect when prob std < 0.02 (very narrow band regardless of
  absolute level). This ensures relative ranking activates whenever the
  distribution is too flat to be meaningful.

FIX 3 — _classify_signals_relative operates on full distribution (carried):
  Top 2% → STRONG BUY, next 8% → BUY, next 15% → HOLD, rest → AVOID.

FIX 4 — Gain regressor receives SCALED features (carried and corrected):
  The regressor is now saved/loaded expecting the same scaled input as the
  classifier. predict_with_targets passes X_scaled (already StandardScaler
  transformed) to both classifier and regressor.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib

_META_COLS = {"symbol", "exchange"}

SIGNAL_THRESHOLDS = {
    "STRONG BUY": 0.90,
    "BUY":        0.70,
    "HOLD":       0.50,
}

# Relative-ranking percentile thresholds (used when _is_bimodal = True)
RELATIVE_STRONG_BUY_PCT = 0.98   # top 2%
RELATIVE_BUY_PCT        = 0.90   # top 2-10%
RELATIVE_HOLD_PCT       = 0.75   # top 10-25%

# Compression detection thresholds
COMPRESSION_THRESHOLD      = 0.85   # >85% below 0.50 → use relative ranking
NARROW_BAND_STD_THRESHOLD  = 0.02   # std < 0.02 regardless of level → use relative ranking

# Original mid-range sparsity check (bimodal toward extremes)
BIMODAL_MIDRANGE             = (0.15, 0.85)
BIMODAL_MIN_MIDRANGE_COUNT   = 5


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
        self._diag_done    = False
        # FIX 1: detected prefix for external callers
        self._model_feature_prefix: str = "t1_close"

        self._load_model()

    # -------------------------------------------------------------------------
    # FIX 1: Expose which prefix the model uses so screen script can match it
    # -------------------------------------------------------------------------

    @property
    def model_feature_prefix(self) -> str:
        """
        Returns the dominant feature prefix in this model's feature list.
        Possible values: "t1_close", "t1_open", "t3", "t5", "t10", "mixed".

        ml_screen_and_predict uses this to build features with the correct prefix
        so they actually match the model's expectations.
        """
        return self._model_feature_prefix

    def _detect_model_prefix(self) -> str:
        """
        Examine self.feature_names and return the dominant prefix group.
        Called once after model load.
        """
        if not self.feature_names:
            return "t1_close"

        counts = {
            "t1_close": sum(1 for f in self.feature_names if f.startswith("t1_close_")),
            "t1_open":  sum(1 for f in self.feature_names if f.startswith("t1_open_")),
            "t3":       sum(1 for f in self.feature_names if f.startswith("t3_")),
            "t5":       sum(1 for f in self.feature_names if f.startswith("t5_")),
            "t10":      sum(1 for f in self.feature_names if f.startswith("t10_")),
        }

        total = sum(counts.values())
        if total == 0:
            return "t1_close"

        dominant = max(counts, key=counts.get)

        # Log the breakdown so it's visible in logs
        self.logger.info("Model feature prefix breakdown:")
        for pfx, cnt in counts.items():
            pct = cnt / total * 100 if total else 0
            self.logger.info(f"  {pfx:12s}: {cnt:4d}  ({pct:.1f}%)")
        self.logger.info(f"  → dominant prefix: '{dominant}' — screen script will use this")

        # If the model has BOTH t1_close and t3 features (hybrid), prefer t1_close
        if counts["t1_close"] > 0 and counts["t3"] > 0:
            return "t1_close"

        return dominant

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
            self.logger.info(f"Loaded gain regressor (expects {self._regressor_n_features} features)")
        else:
            self.regressor = None
            self.logger.warning(
                "gain_regressor.pkl not found — will use rule-based gain estimates. "
                "Run ml_retrain_model.py once ≥30 winner rows have gain data."
            )

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.scaler.n_features_in_)]
            self.logger.warning("Scaler has no feature_names_in_ - using positional names.")

        self._build_lookup()

        # FIX 1: Detect the dominant prefix in the loaded model
        self._model_feature_prefix = self._detect_model_prefix()

        classifier_n = self.scaler.n_features_in_
        self.logger.info(
            f"Classifier/scaler expects {classifier_n} features; "
            f"regressor expects {self._regressor_n_features} features"
        )

        if (self._regressor_n_features is not None
                and self._regressor_n_features != classifier_n):
            self.logger.warning(
                f"Regressor feature count ({self._regressor_n_features}) != "
                f"classifier ({classifier_n}). Regressor DISABLED — retrain both together."
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
            self.logger.info("Model type: CSV-ONLY (T-3/T-5/T-10) — features must use t3_ prefix")
        elif has_t1:
            self.logger.info("Model type: DATABASE-ONLY (T-1 open/close) — features must use t1_close_ prefix")
        else:
            self.logger.warning("Could not identify model type. First 10 features: %s",
                                self.feature_names[:10])

    def _build_lookup(self):
        self._norm_to_feature = {_norm(f): f for f in self.feature_names}

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def _log_feature_diagnostics(self, feature_df: pd.DataFrame, match_log: dict):
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

            live_cols = [c for c in num_cols if col_stds.get(c, 0) >= 1e-9][:5]
            if live_cols and len(feature_df) >= 3:
                for col in live_cols:
                    vals = feature_df[col].astype(float).iloc[:3].tolist()
                    self.logger.info(
                        f"    sample {col}: "
                        + "  ".join(f"{v:>10.4f}" for v in vals)
                    )

        self.logger.info("")

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
                    feat_col  = None
                    if feat in feature_df.columns:
                        feat_col = feat
                    else:
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
            self.logger.warning(
                f"LOW feature coverage ({coverage:.1f}%) — model may be using t3_ prefix "
                f"but features were built with '{self._model_feature_prefix}' prefix. "
                f"Check that build_features_from_tv_data uses the correct prefix."
            )

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

        return X_scaled

    # -------------------------------------------------------------------------
    # FIX 2: Enhanced bimodal/compression/narrow-band detection
    # -------------------------------------------------------------------------

    def _detect_bimodal(self, probabilities: np.ndarray) -> bool:
        """
        Detect whether the model's output distribution is degenerate and
        requires relative (percentile-based) signal classification.

        Three failure modes:
          A) Compression below 0.50: >85% of predictions below 0.50
             (T3-only model with weak features)
          B) Narrow band regardless of level: std < 0.02
             (all stocks get ~0.73 because features are too uniform)
          C) Bimodal collapse toward extremes: sparse mid-range
             (original failure mode)
        """
        n = len(probabilities)

        # Mode A: Low-probability compression
        below_half = int((probabilities < 0.50).sum())
        compression_rate = below_half / n if n > 0 else 0
        if compression_rate > COMPRESSION_THRESHOLD:
            self.logger.warning(
                f"LOW-PROB COMPRESSION: {compression_rate:.1%} of predictions below 0.50. "
                f"Switching to percentile-based signals."
            )
            return True

        # Mode B: FIX — narrow probability band (new check)
        prob_std = float(np.std(probabilities))
        if prob_std < NARROW_BAND_STD_THRESHOLD:
            self.logger.warning(
                f"NARROW PROBABILITY BAND: std={prob_std:.4f} < {NARROW_BAND_STD_THRESHOLD}. "
                f"All stocks are getting near-identical scores. "
                f"Switching to percentile-based signals to give differentiated BUY/STRONG BUY/HOLD."
            )
            return True

        # Mode C: Bimodal toward extremes
        mid_lo, mid_hi = BIMODAL_MIDRANGE
        mid_count = int(((probabilities > mid_lo) & (probabilities < mid_hi)).sum())
        if mid_count < BIMODAL_MIN_MIDRANGE_COUNT:
            self.logger.warning(
                f"BIMODAL COLLAPSE: only {mid_count} predictions in mid-range "
                f"({mid_lo:.0%}–{mid_hi:.0%}). Switching to percentile-based signals."
            )
            return True

        return False

    def _classify_signal_absolute(self, probability: float) -> str:
        if probability >= SIGNAL_THRESHOLDS["STRONG BUY"]: return "STRONG BUY"
        elif probability >= SIGNAL_THRESHOLDS["BUY"]:       return "BUY"
        elif probability >= SIGNAL_THRESHOLDS["HOLD"]:      return "HOLD"
        return "AVOID"

    def _classify_signals_relative(self, probabilities: pd.Series) -> pd.Series:
        """
        FIX 3: Rank-based signal classification on the FULL distribution.

        Top 2%   → STRONG BUY
        2–10%    → BUY
        10–25%   → HOLD
        Bottom 75% → AVOID
        """
        n = len(probabilities)
        signals = pd.Series("AVOID", index=probabilities.index)

        if n == 0:
            return signals

        ranks = probabilities.rank(pct=True)

        signals[ranks >= RELATIVE_STRONG_BUY_PCT] = "STRONG BUY"
        signals[(ranks >= RELATIVE_BUY_PCT) & (ranks < RELATIVE_STRONG_BUY_PCT)] = "BUY"
        signals[(ranks >= RELATIVE_HOLD_PCT) & (ranks < RELATIVE_BUY_PCT)] = "HOLD"

        strong_buy_n = (signals == "STRONG BUY").sum()
        buy_n        = (signals == "BUY").sum()
        hold_n       = (signals == "HOLD").sum()

        self.logger.info(
            f"Relative signals (percentile-based): STRONG BUY={strong_buy_n}, BUY={buy_n}, "
            f"HOLD={hold_n}, AVOID={n - strong_buy_n - buy_n - hold_n}"
        )
        self.logger.info(
            f"  Prob range: {probabilities.min():.4f} – {probabilities.max():.4f}, "
            f"std={probabilities.std():.4f}"
        )

        return signals

    # -------------------------------------------------------------------------
    # Prediction — internal, returns (result_df, features_df, X_scaled)
    # -------------------------------------------------------------------------

    def _predict_internal(
        self, data_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        features_df = self.prepare_features(data_df)
        X           = features_df[self.feature_names].copy()
        X_scaled    = self._scale_features(X)

        scaled_std    = X_scaled.std()
        zero_var_cols = int((scaled_std < 1e-9).sum())
        self.logger.info(
            f"Scaled matrix: {X_scaled.shape} | "
            f"zero-var cols: {zero_var_cols} | "
            f"mean_std: {scaled_std.mean():.4f}"
        )
        if zero_var_cols > len(self.feature_names) * 0.5:
            self.logger.warning(
                "MORE THAN HALF of features are zero-variance after scaling. "
                "This strongly suggests the feature prefix in the screen script "
                f"does not match what the model was trained on. "
                f"Model dominant prefix: '{self._model_feature_prefix}'. "
                "Check TV_TO_MODEL_BASE builds features with this prefix."
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
        }, index=data_df.index)

        for col in ("symbol", "exchange"):
            if col in features_df.columns:
                result_df.insert(0, col, features_df[col].values)

        result_df = result_df.sort_values(
            "explosion_probability", ascending=False
        ).reset_index(drop=True)

        result_df["_orig_idx"] = range(len(result_df))
        X_scaled_sorted = X_scaled.reset_index(drop=True).iloc[
            result_df["_orig_idx"].values
        ].reset_index(drop=True)
        result_df = result_df.drop(columns=["_orig_idx"])

        return result_df, features_df, X_scaled_sorted

    def predict(self, data_df: pd.DataFrame) -> pd.DataFrame:
        result_df, _, _ = self._predict_internal(data_df)
        return result_df

    # -------------------------------------------------------------------------
    # Prediction with gain targets
    # -------------------------------------------------------------------------

    def predict_with_targets(
        self,
        data_df: pd.DataFrame,
        historical_gains_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        predictions, features_df, X_scaled = self._predict_internal(data_df)

        # FIX 4: Regressor receives X_scaled (same as classifier)
        if self.regressor is not None:
            try:
                predicted_gains = self.regressor.predict(X_scaled)
                self.logger.info(
                    f"Gain regressor: predicted {len(predicted_gains)} gains  "
                    f"range=[{predicted_gains.min():.1f}%, {predicted_gains.max():.1f}%]  "
                    f"mean={predicted_gains.mean():.1f}%  std={predicted_gains.std():.2f}%"
                )

                # Sanity check: if all gains are nearly identical the regressor isn't helping
                if predicted_gains.std() < 1.0:
                    self.logger.warning(
                        f"Gain regressor std={predicted_gains.std():.4f}% — very flat. "
                        "Disabling and falling back to historical/rule-based estimates."
                    )
                    self.regressor = None
                else:
                    predictions["target_gain_pct"]  = predicted_gains
                    predictions["target_gain_low"]   = predicted_gains * 0.8
                    predictions["target_gain_high"]  = predicted_gains * 1.2
            except Exception as e:
                self.logger.warning(f"Regressor predict failed ({e}) — falling back")
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
                predictions["target_gain_low"]   = predictions["median"] - predictions["std"].fillna(0)
                predictions["target_gain_high"]  = predictions["median"] + predictions["std"].fillna(0)
                predictions = predictions.drop(["prob_bucket", "mean", "median", "std"], axis=1)
            else:
                self.logger.warning("No regressor and no historical data — using rule-based gain estimates")
                predictions["target_gain_pct"]  = predictions["explosion_probability"].apply(
                    self._estimate_target_gain)
                predictions["target_gain_low"]   = predictions["target_gain_pct"] * 0.5
                predictions["target_gain_high"]  = predictions["target_gain_pct"] * 1.5

        # Fill any remaining NaN gains
        nan_gain = predictions["target_gain_pct"].isna()
        if nan_gain.any():
            predictions.loc[nan_gain, "target_gain_pct"] = (
                predictions.loc[nan_gain, "explosion_probability"]
                .apply(self._estimate_target_gain))
            predictions.loc[nan_gain, "target_gain_low"]  = (
                predictions.loc[nan_gain, "target_gain_pct"] * 0.5)
            predictions.loc[nan_gain, "target_gain_high"] = (
                predictions.loc[nan_gain, "target_gain_pct"] * 1.5)

        # Current price lookup
        if "symbol" in predictions.columns:
            current_price = self._extract_current_price(data_df, predictions)
            if current_price is not None:
                predictions["current_price"]     = current_price
                predictions["target_price"]      = current_price * (1 + predictions["target_gain_pct"] / 100)
                predictions["target_price_low"]  = current_price * (1 + predictions["target_gain_low"] / 100)
                predictions["target_price_high"] = current_price * (1 + predictions["target_gain_high"] / 100)
            else:
                self.logger.warning(
                    "Could not determine current_price — target_price columns will be missing."
                )

        return predictions

    def _extract_current_price(
        self, data_df: pd.DataFrame, predictions: pd.DataFrame
    ) -> Optional[pd.Series]:
        if "symbol" not in data_df.columns:
            return None

        # Build candidate list using both t1_close_ and t3_ prefixes so this works
        # regardless of which model is loaded
        price_candidates = ["current_price", "t3_Close", "t1_close_Close", "Close"]
        for col in data_df.columns:
            for pfx in ("t3_", "t1_close_", "t1_open_"):
                if col.startswith(pfx) and col.lower().endswith("_close"):
                    if col not in price_candidates:
                        price_candidates.append(col)

        price_col = next(
            (c for c in price_candidates if c in data_df.columns), None
        )

        if price_col is None:
            for col in data_df.columns:
                for pfx in ("t3_", "t1_close_", "t1_open_"):
                    if col.startswith(pfx) and any(x in col.lower() for x in ("close", "price")):
                        price_col = col
                        break
                if price_col:
                    break

        if price_col is None:
            return None

        if price_col != "current_price":
            self.logger.info(f"Using '{price_col}' as current price source")

        price_map = data_df.set_index("symbol")[price_col]
        aligned   = predictions["symbol"].map(price_map)

        if aligned.isna().all():
            self.logger.warning(f"Price column '{price_col}' produced all NaN after symbol join")
            return None

        return aligned.values

    # -------------------------------------------------------------------------
    # Compatibility shims
    # -------------------------------------------------------------------------

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
