#!/usr/bin/env python3
"""
ml_retrain_model.py  —  Weekly Full Retrain From Scratch

Replaces the previous fine-tuning approach with a complete retrain every week.

DATA SOURCES (combined into one training dataset):
  1. ml_training_base    — original CSV data pivoted to wide format, both classes
                           Feature prefixes: t3_, t5_, t10_ only
  2. winners_day_prior_close / winners_day_prior_open
                         — accumulating T-1 winner samples from daily runs (label=1)
  3. non_winners_day_prior_close / non_winners_day_prior_open
                         — accumulating T-1 non-winner samples from daily runs (label=0)
  4. ml_mistake_learner  — high-weight samples from the model's own past errors
                           (false positives: weight 3x, false negatives: weight 2x)

NOTE ON CLASS BALANCE:
  ml_training_base contains both winners (label=1) and non-winners (label=0) from
  the original CSV, all with t3_/t5_/t10_ features from daily bars.
  
  The T-1 snapshot tables (winners_day_prior_*, non_winners_day_prior_*) contain
  raw intraday indicator data WITHOUT pre-assigned labels — the label is assigned
  in load_t1_data() based on which table the row came from (winners → 1, non_winners → 0).
  
  daily_winners and daily_non_winners are ground-truth outcome tables used by the
  accuracy tracker, not directly by the trainer.

WHY FULL RETRAIN (not fine-tuning):
  - Only ~3,600 base rows — trivially fast to retrain (seconds, not minutes)
  - Fine-tuning with dummy-default T-3/T-7/T-14 values was corrupting new trees
  - NaN for genuinely missing columns is correct; XGBoost handles it natively
  - feature_importance.csv is regenerated each run — always accurate and current

OUTPUTS (same paths as before, drop-in compatible with ml_weekly_retrain.yml):
  ml_models/best_model.pkl
  ml_models/scaler.pkl
  ml_models/model_metadata.json
  ml_models/feature_importance.csv
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client
from xgboost import XGBClassifier

# T-1 column name translator (intraday short names → model long names)
try:
    from t1_column_map import rename_t1_columns
    T1_MAP_AVAILABLE = True
except ImportError:
    T1_MAP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "t1_column_map.py not found — T-1 features will not be renamed. "
        "Place t1_column_map.py alongside ml_retrain_model.py."
    )

# Mistake learner — high-signal training samples from past prediction errors
try:
    from ml_mistake_learner import build_mistake_training_samples, log_mistake_summary
    MISTAKE_LEARNER_AVAILABLE = True
except ImportError:
    MISTAKE_LEARNER_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ml_mistake_learner.py not found — mistake-learning step will be skipped."
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TABLE_BASE             = "ml_training_base"
TABLE_WINNERS_CLOSE    = "winners_day_prior_close"
TABLE_WINNERS_OPEN     = "winners_day_prior_open"
TABLE_NON_WINNERS_CLOSE = "non_winners_day_prior_close"
TABLE_NON_WINNERS_OPEN  = "non_winners_day_prior_open"

MODEL_DIR               = Path("ml_models")
MODEL_PATH              = MODEL_DIR / "best_model.pkl"
SCALER_PATH             = MODEL_DIR / "scaler.pkl"
GAIN_REGRESSOR_PATH     = MODEL_DIR / "gain_regressor.pkl"
METADATA_PATH           = MODEL_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

BASE_CSV_WEIGHT        = 1.5
T1_WEIGHT              = 1.0
MIN_T1_ROWS_FOR_EQUAL_WEIGHT = 1800

XGBOOST_PARAMS = {
    "n_estimators":       300,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   3,
    "gamma":              0.1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "scale_pos_weight":   1,   # overridden at train time
    "objective":          "binary:logistic",
    "eval_metric":        "logloss",
    "use_label_encoder":  False,
    "random_state":       42,
    "n_jobs":             -1,
    "early_stopping_rounds": 30,
}

# Columns excluded from the feature matrix X.
# "mistake_type" must be here — it's a string column added by ml_mistake_learner
# that would otherwise be coerced to NaN silently by pd.to_numeric.
NON_FEATURE_COLS = {
    "id", "created_at", "updated_at", "date", "symbol", "ticker",
    "label", "source", "sample_weight", "detection_date", "explosion_date",
    "change_pct", "rank", "notes", "mistake_type", "actual_gain_pct",
}

T1_MARKER_PREFIXES = ("t1_", "open_", "close_")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set.")
        sys.exit(1)
    return create_client(url, key)


def fetch_table_paginated(client: Client, table: str, page_size: int = 1000) -> pd.DataFrame:
    """Fetch all rows from a Supabase table using pagination."""
    rows   = []
    offset = 0
    while True:
        resp = (
            client.table(table)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        logger.info(f"  {table}: fetched {len(rows)} rows so far...")
        if len(batch) < page_size:
            break
        offset += page_size
    df = pd.DataFrame(rows)
    logger.info(f"  {table}: total {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_base_training_data(client: Client) -> pd.DataFrame:
    """Load original CSV data from ml_training_base."""
    logger.info(f"Loading base training data from '{TABLE_BASE}'...")
    df = fetch_table_paginated(client, TABLE_BASE)
    if df.empty:
        logger.error(
            f"Table '{TABLE_BASE}' is empty! "
            "Run upload_base_training_data.py first."
        )
        sys.exit(1)

    if "label" not in df.columns:
        logger.error(f"'{TABLE_BASE}' has no 'label' column.")
        sys.exit(1)

    if "sample_weight" not in df.columns:
        df["sample_weight"] = BASE_CSV_WEIGHT
    df["source"] = df.get("source", "base_csv")

    logger.info(f"Base data: {len(df)} rows, "
                f"pos={int((df['label']==1).sum())}, "
                f"neg={int((df['label']==0).sum())}")
    return df


def load_t1_data(client: Client) -> pd.DataFrame:
    """
    Load accumulated T-1 winner and non-winner samples.

    Applies t1_column_map to rename intraday short-form column names
    (rsi, stoch.k, ema20, …) to the model's expected long-form names
    (RSI_14, STOCHk_14_3_3, EMA_20, …) with the correct prefix.

    close tables → prefix "t1_close"
    open  tables → prefix "t1_open"
    """
    logger.info("Loading accumulated T-1 training data...")

    TABLE_CONFIG = [
        (TABLE_WINNERS_CLOSE,      1, "t1_close"),
        (TABLE_WINNERS_OPEN,       1, "t1_open"),
        (TABLE_NON_WINNERS_CLOSE,  0, "t1_close"),
        (TABLE_NON_WINNERS_OPEN,   0, "t1_open"),
    ]

    frames = []

    for table, label, prefix in TABLE_CONFIG:
        try:
            df = fetch_table_paginated(client, table)
            if df.empty:
                continue

            df["label"]  = label
            df["source"] = table

            if T1_MAP_AVAILABLE:
                before = len(df.columns)
                df     = rename_t1_columns(df, prefix=prefix)
                after  = len([c for c in df.columns if c.startswith(prefix)])
                logger.info(
                    f"  {table}: renamed {after} feature columns "
                    f"(had {before}, kept metadata + {after} features)"
                )

                # Belt-and-suspenders: drop any surviving duplicate column names
                dupes = df.columns[df.columns.duplicated()].tolist()
                if dupes:
                    logger.warning(
                        f"  {table}: dropping {len(dupes)} duplicate column(s) "
                        f"after rename: {dupes[:10]}"
                    )
                    df = df.loc[:, ~df.columns.duplicated(keep="first")]
            else:
                logger.warning(
                    f"  {table}: column map unavailable — "
                    "T-1 features will be NaN in model (not ideal but won't crash)"
                )

            frames.append(df)

        except Exception as e:
            logger.warning(f"Could not load '{table}': {e}")

    if not frames:
        logger.warning("No T-1 data found. Training on base data only.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["sample_weight"] = T1_WEIGHT

    t1_feature_cols = [c for c in combined.columns
                       if c.startswith("t1_close_") or c.startswith("t1_open_")]
    non_null_t1 = combined[t1_feature_cols].notna().any().sum() if t1_feature_cols else 0

    logger.info(f"T-1 data: {len(combined)} rows, "
                f"pos={int((combined['label']==1).sum())}, "
                f"neg={int((combined['label']==0).sum())}")
    logger.info(f"T-1 feature columns populated: {non_null_t1}/{len(t1_feature_cols)}")

    return combined


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def combine_datasets(base_df: pd.DataFrame, t1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate base + T-1 data.

    Columns present only in base → NaN in T-1 rows (XGBoost handles natively)
    Columns present only in T-1  → NaN in base rows (XGBoost handles natively)

    NOTE: mistake samples should be added AFTER this function returns,
    so their custom sample_weights (3.0 / 2.0) are not overwritten here.
    """
    if t1_df.empty:
        logger.info("Combining: base data only (no T-1 data yet)")
        return base_df.copy()

    t1_count = len(t1_df)
    if t1_count >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT:
        logger.info(
            f"T-1 data ({t1_count} rows) >= threshold ({MIN_T1_ROWS_FOR_EQUAL_WEIGHT}). "
            "Using equal sample weights (1.0 / 1.0)."
        )
        base_df = base_df.copy()
        base_df["sample_weight"] = 1.0
    else:
        logger.info(
            f"T-1 data ({t1_count} rows) < threshold ({MIN_T1_ROWS_FOR_EQUAL_WEIGHT}). "
            f"Base rows weighted {BASE_CSV_WEIGHT}x, T-1 rows weighted {T1_WEIGHT}x."
        )

    combined = pd.concat([base_df, t1_df], ignore_index=True, sort=False)

    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())
    logger.info(
        f"Combined dataset: {len(combined)} rows, "
        f"{len(combined.columns)} columns, "
        f"pos={n_pos}, neg={n_neg}"
    )

    if n_neg == 0:
        logger.error(
            "CRITICAL: No negative (non-winner) samples found. "
            "The model cannot train without both classes. "
            "Ensure non_winners_day_prior_close/open tables have data in Supabase."
        )
        sys.exit(1)

    if n_pos > 0 and (n_neg / n_pos) < 0.2:
        logger.warning(
            f"Class imbalance WARNING: {n_pos} positives vs {n_neg} negatives "
            f"(ratio {n_neg/n_pos:.2f}). scale_pos_weight will compensate, "
            "but consider accumulating more non-winner data before relying on this model."
        )

    return combined


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract feature matrix X, labels y, and sample weights w.

    Returns:
        X: DataFrame of features (NaN allowed — XGBoost handles natively)
        y: Series of labels (0/1)
        w: Series of sample weights
    """
    y = df["label"].astype(int)
    w = (
        df["sample_weight"].astype(float)
        if "sample_weight" in df.columns
        else pd.Series(1.0, index=df.index)
    )

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and not c.startswith("Unnamed")
    ]

    X = df[feature_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    nan_pct = X.isna().mean().mean() * 100
    logger.info(f"Overall NaN rate: {nan_pct:.1f}% (expected for cross-lag rows)")

    return X, y, w


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def build_scaler(X: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """
    Fit scaler on non-NaN values per column. Returns scaler + scaled DataFrame.

    NaN positions are PRESERVED after scaling so XGBoost can use its native
    missing-value routing.  We fill NaN with column mean only for the purpose
    of fitting and applying the scaler, then immediately restore NaN.
    """
    scaler    = StandardScaler()
    col_means = X.mean()
    X_filled  = X.fillna(col_means)
    scaler.fit(X_filled)

    nan_mask       = X.isna()
    X_scaled_vals  = scaler.transform(X_filled)
    X_scaled       = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
    X_scaled[nan_mask] = np.nan   # restore NaN so XGBoost routes correctly

    return scaler, X_scaled


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    """Train XGBClassifier from scratch with early stopping."""
    params = XGBOOST_PARAMS.copy()

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0 and n_neg > 0:
        params["scale_pos_weight"] = round(n_neg / n_pos, 3)
        logger.info(f"  scale_pos_weight set to {params['scale_pos_weight']:.3f} "
                    f"(neg={n_neg} / pos={n_pos})")

    early_stopping = params.pop("early_stopping_rounds", 30)

    model = XGBClassifier(**params, early_stopping_rounds=early_stopping)

    logger.info("Training XGBoost model from scratch...")
    logger.info(f"  Train: {len(X_train)} rows")
    logger.info(f"  Val:   {len(X_val)} rows")

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train.values,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info(f"  Best iteration: {model.best_iteration}")
    logger.info(f"  Best val logloss: {model.best_score:.4f}")

    return model


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    val_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Stratified split preserving class balance."""
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w,
        test_size=val_fraction,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        f"Train/val split: {len(X_train)} train "
        f"(pos={int((y_train==1).sum())}, neg={int((y_train==0).sum())}), "
        f"{len(X_val)} val "
        f"(pos={int((y_val==1).sum())}, neg={int((y_val==0).sum())})"
    )
    return X_train, X_val, y_train, y_val, w_train, w_val


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Generate feature_importance.csv using gain importance."""
    booster = model.get_booster()
    scores  = booster.get_score(importance_type="gain")

    importance_list = []
    for feat, score in scores.items():
        if feat.startswith("f") and feat[1:].isdigit():
            idx  = int(feat[1:])
            name = feature_names[idx] if idx < len(feature_names) else feat
        else:
            name = feat
        importance_list.append({"feature": name, "importance": round(score, 6)})

    fi_df  = pd.DataFrame(importance_list)
    fi_df  = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    total  = fi_df["importance"].sum()
    if total > 0:
        fi_df["importance"] = (fi_df["importance"] / total).round(6)

    logger.info(f"Feature importance computed: {len(fi_df)} features")
    logger.info("Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return fi_df


# ---------------------------------------------------------------------------
# Gain regressor
# ---------------------------------------------------------------------------

def train_gain_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    combined_df: pd.DataFrame,
    feature_names: list[str],
) -> "Optional[XGBClassifier]":
    """
    Train a regression model to predict actual % gain for stocks the
    classifier labels as winners.

    WHY A SEPARATE REGRESSOR:
      The classifier only outputs a probability (0–1). The gain regressor
      takes the same features and predicts the actual % gain, so the predictor
      can show realistic price targets instead of rigid rule-based estimates.

    TRAINING DATA:
      Only rows where label=1 (actual winners) are used — we can only measure
      gain for stocks that actually exploded. Non-winners have gain=0 or
      undefined, which would teach the regressor the wrong thing.

    TARGET:
      actual_gain_pct (% change from prior close to intraday high on explosion day).
      Falls back to change_pct if actual_gain_pct is not in combined_df.

    Returns:
        Trained XGBRegressor, or None if not enough winner rows to train on.
    """
    from xgboost import XGBRegressor

    # Find gain column — prefer actual_gain_pct, fall back to change_pct
    gain_col = None
    for candidate in ("actual_gain_pct", "change_pct"):
        if candidate in combined_df.columns:
            gain_col = candidate
            break

    if gain_col is None:
        logger.warning("No gain column found (actual_gain_pct / change_pct) — "
                       "skipping gain regressor training.")
        return None

    # Restrict to winner rows that have a real gain value
    winner_mask  = (combined_df["label"] == 1) & combined_df[gain_col].notna()
    n_winners    = int(winner_mask.sum())

    if n_winners < 30:
        logger.warning(f"Only {n_winners} winner rows with gain data — "
                       "need ≥30 to train gain regressor. Skipping.")
        return None

    logger.info(f"\n── Training gain regressor on {n_winners} winner rows ──")

    # Build aligned feature / target arrays using the same index as combined_df
    X_reg = pd.DataFrame(index=combined_df.index, columns=feature_names)
    for col in feature_names:
        if col in combined_df.columns:
            X_reg[col] = combined_df[col]
    X_reg = X_reg.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    y_reg = pd.to_numeric(combined_df[gain_col], errors="coerce")
    w_reg = combined_df["sample_weight"].astype(float) \
            if "sample_weight" in combined_df.columns \
            else pd.Series(1.0, index=combined_df.index)

    # Filter to winner rows
    X_reg  = X_reg[winner_mask]
    y_reg  = y_reg[winner_mask]
    w_reg  = w_reg[winner_mask]

    # Scale using the same column-mean fill approach as the classifier
    col_means  = X_reg.mean()
    X_reg_fill = X_reg.fillna(col_means)

    from sklearn.model_selection import train_test_split
    if len(X_reg) >= 10:
        X_tr, X_va, y_tr, y_va, w_tr, _ = train_test_split(
            X_reg_fill, y_reg, w_reg,
            test_size=0.2, random_state=42,
        )
    else:
        X_tr, X_va, y_tr, y_va, w_tr = X_reg_fill, X_reg_fill, y_reg, y_reg, w_reg

    regressor = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    regressor.fit(
        X_tr, y_tr,
        sample_weight=w_tr.values,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    val_pred = regressor.predict(X_va)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_va, val_pred)
    r2  = r2_score(y_va, val_pred) if len(y_va) > 1 else float("nan")
    logger.info(f"  Gain regressor — val MAE: {mae:.2f}%  R²: {r2:.3f}")
    logger.info(f"  Predicted gains range: {val_pred.min():.1f}% – {val_pred.max():.1f}%")

    return regressor


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    model: XGBClassifier,
    scaler: StandardScaler,
    fi_df: pd.DataFrame,
    feature_names: list[str],
    training_stats: dict,
    gain_regressor=None,
) -> None:
    """Save model, scaler, gain regressor, feature importance, and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  MODEL_PATH,  protocol=4)
    logger.info(f"Saved model  → {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH, protocol=4)
    logger.info(f"Saved scaler → {SCALER_PATH}")

    if gain_regressor is not None:
        joblib.dump(gain_regressor, GAIN_REGRESSOR_PATH, protocol=4)
        logger.info(f"Saved gain regressor → {GAIN_REGRESSOR_PATH}")
    else:
        logger.info("Gain regressor not trained this run — predictor will use rule-based gains")

    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Saved feature importance → {FEATURE_IMPORTANCE_PATH}")

    metadata = {
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "source":                "ml_retrain_model.py",
        "training_approach":     "full_retrain_from_scratch",
        "n_features":            len(feature_names),
        "features":              feature_names,
        "feature_names_sample":  feature_names[:20],
        "best_iteration":        int(model.best_iteration),
        "best_val_logloss":      float(model.best_score),
        "gain_regressor_trained": gain_regressor is not None,
        **training_stats,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata → {METADATA_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("=" * 60)
    logger.info("ML RETRAIN — FULL RETRAIN FROM SCRATCH")
    logger.info("=" * 60)

    # ── Connect ──────────────────────────────────────────────────────────────
    client = get_supabase_client()

    # ── Load standard training data ───────────────────────────────────────────
    base_df     = load_base_training_data(client)
    t1_df       = load_t1_data(client)
    combined_df = combine_datasets(base_df, t1_df)

    # ── Enrich combined_df with intraday peak gain from daily_winners ─────────
# ── Enrich combined_df with intraday peak gain from daily_winners ─────────
    logger.info("Fetching intraday peak gain data from daily_winners for gain regressor...")
    try:
        winners_response = fetch_table_paginated(client, "daily_winners")
        if not winners_response.empty:
            required = {"symbol", "detection_date", "high", "price"}
            if required.issubset(winners_response.columns):
                winners_gain = winners_response[["symbol", "detection_date", "high", "price"]].copy()
                winners_gain["actual_gain_pct"] = (
                    (winners_gain["high"] / winners_gain["price"] - 1) * 100
                ).clip(lower=0)
    
                # Find symbol column — base CSV uses "ticker", T-1 data uses "symbol"
                symbol_col = next(
                    (c for c in ["symbol", "ticker"] if c in combined_df.columns),
                    None
                )
    
                # Find date column — prioritize detection_date variants over event_date
                date_col = next(
                    (c for c in ["detection_date_x", "detection_date", "event_date"]
                     if c in combined_df.columns),
                    None
                )
    
                if symbol_col and date_col:
                    logger.info(f"Joining gain data on {symbol_col} + {date_col}")
                    combined_df = combined_df.merge(
                        winners_gain[["symbol", "detection_date", "actual_gain_pct"]],
                        left_on=[symbol_col, date_col],
                        right_on=["symbol", "detection_date"],
                        how="left",
                    ).drop(columns=["detection_date"], errors="ignore")
                    n_with_gain = combined_df["actual_gain_pct"].notna().sum()
                    logger.info(f"Enriched {n_with_gain} rows with intraday peak gain data")
                else:
                    logger.warning(f"Could not find symbol col ({symbol_col}) or date col ({date_col}) — gain regressor will be skipped")
            else:
                missing = required - set(winners_response.columns)
                logger.warning(f"daily_winners missing columns: {missing} — gain regressor will be skipped")
        else:
            logger.warning("daily_winners table is empty — gain regressor will be skipped")
    except Exception as e:
        logger.warning(f"Could not fetch gain data: {e} — gain regressor will be skipped")
    logger.info(f"combined_df columns with 'date': {[c for c in combined_df.columns if 'date' in c.lower()]}")
    logger.info(f"combined_df shape before merge: {combined_df.shape}")
    logger.info(f"winners_gain sample:\n{winners_gain[['symbol','detection_date','actual_gain_pct']].head()}")
    logger.info(f"combined_df symbol+date sample:\n{combined_df[['symbol', date_col]].head() if date_col else 'NO DATE COL FOUND'}")

    # ── Load mistake samples and append AFTER combine_datasets ───────────────
    # Crucial: appending here preserves the high sample_weights (3.0 / 2.0)
    # assigned by ml_mistake_learner.  If mistake samples were passed into
    # combine_datasets(), their weights would be overwritten with T1_WEIGHT.
    if MISTAKE_LEARNER_AVAILABLE:
        logger.info("\n" + "=" * 60)
        logger.info("MISTAKE LEARNING STEP")
        logger.info("=" * 60)

        # Derive the feature list from what we've assembled so far so the
        # mistake learner can pad missing columns with sensible defaults.
        proto_features = [
            c for c in combined_df.columns
            if c not in NON_FEATURE_COLS and not c.startswith("Unnamed")
        ]

        mistake_df = build_mistake_training_samples(
            lookback_days=90,
            use_all_timepoints=True,
            existing_features=proto_features,
        )

        if not mistake_df.empty:
            log_mistake_summary(mistake_df)
            combined_df = pd.concat([combined_df, mistake_df],
                                    ignore_index=True, sort=False)
            logger.info(
                f"Dataset after adding mistakes: {len(combined_df)} rows "
                f"(+{len(mistake_df)} mistake samples)"
            )
        else:
            logger.info("No mistake samples to add this run.")
    else:
        logger.warning("ml_mistake_learner not available — skipping mistake-learning step.")

    # ── Prepare features ──────────────────────────────────────────────────────
    X, y, w = prepare_features(combined_df)
    feature_names = list(X.columns)

    # ── Scale ─────────────────────────────────────────────────────────────────
    logger.info("Fitting scaler...")
    scaler, X_scaled = build_scaler(X)

    # ── Train/val split ───────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val, w_train, w_val = train_val_split(X_scaled, y, w)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train, w_train, X_val, y_val)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df = compute_feature_importance(model, feature_names)

    # ── Train gain regressor ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("GAIN REGRESSOR TRAINING")
    logger.info("=" * 60)
    gain_regressor = train_gain_regressor(
        X_train, y_train, w_train, combined_df, feature_names
    )

    # ── Evaluate classifier ───────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, classification_report

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)
    auc       = roc_auc_score(y_val, val_proba)
    logger.info(f"Validation AUC-ROC: {auc:.4f}")
    logger.info("Classification report (val):")
    for line in classification_report(y_val, val_pred).split("\n"):
        if line.strip():
            logger.info(f"  {line}")

    # ── Training stats for metadata ───────────────────────────────────────────
    n_mistakes = len(mistake_df) if (MISTAKE_LEARNER_AVAILABLE and "mistake_df" in dir()) else 0
    training_stats = {
        "n_total_samples":      len(combined_df),
        "n_base_samples":       len(base_df),
        "n_t1_samples":         len(t1_df) if not t1_df.empty else 0,
        "n_mistake_samples":    n_mistakes,
        "n_positive":           int((y == 1).sum()),
        "n_negative":           int((y == 0).sum()),
        "positive_rate":        float((y == 1).mean()),
        "val_auc_roc":          float(auc),
        "base_sample_weight":   BASE_CSV_WEIGHT,
        "t1_sample_weight":     T1_WEIGHT,
        "equal_weight_applied": (
            len(t1_df) >= MIN_T1_ROWS_FOR_EQUAL_WEIGHT
            if not t1_df.empty else False
        ),
        "gain_regressor_trained": gain_regressor is not None,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(model, scaler, fi_df, feature_names, training_stats, gain_regressor)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RETRAIN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total samples    : {training_stats['n_total_samples']}")
    logger.info(f"  Base CSV samples : {training_stats['n_base_samples']}")
    logger.info(f"  T-1 samples      : {training_stats['n_t1_samples']}")
    logger.info(f"  Mistake samples  : {training_stats['n_mistake_samples']}")
    logger.info(f"  Positive rate    : {training_stats['positive_rate']:.1%}")
    logger.info(f"  Validation AUC   : {auc:.4f}")
    logger.info(f"  Best iteration   : {model.best_iteration}")
    logger.info(f"  Features         : {len(feature_names)}")
    logger.info(f"  Gain regressor   : {'✓ trained' if gain_regressor else '— skipped (not enough winner gain data)'}")
    logger.info("")
    logger.info("Files written:")
    logger.info(f"  {MODEL_PATH}")
    logger.info(f"  {SCALER_PATH}")
    if gain_regressor is not None:
        logger.info(f"  {GAIN_REGRESSOR_PATH}")
    logger.info(f"  {METADATA_PATH}")
    logger.info(f"  {FEATURE_IMPORTANCE_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
