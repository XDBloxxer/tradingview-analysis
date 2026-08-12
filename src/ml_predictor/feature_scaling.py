"""
feature_scaling.py — Shared feature-scaling logic for training and prediction.

This module holds the ONLY implementation of winsorization + StandardScaler
fit/transform used by the pipeline. It exists so that ml_retrain_model.py
(training) and explosion_predictor.py (live prediction) call the exact same
functions — not two independently maintained implementations that can drift
out of sync with each other.

Previously explosion_predictor.py had its own hand-rolled scaling logic in
_scale_features() that mirrored build_scaler()/scale_with_fitted_scaler() by
manual re-implementation (rebuilding mean_series, fillna, sparse-column NaN
restoration by hand) and had no winsorization step at all — so a live batch
could contain an outlier value the training-time winsorization would have
clipped, and that value would flow straight into self.scaler.transform()
unclipped, putting the feature on a scale the model never saw during
training. Moving the real implementation here and having both callers use it
removes that drift risk entirely: whatever training does to a column is
exactly what prediction does to that column, for every feature, every time.

Nothing in this module talks to Supabase, argparse, or does any I/O beyond
what's passed in — it's safe to import from the lightweight prediction path
without pulling in the training script's heavier dependencies.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

_WINSOR_LOWER_PCT = 0.5   # per-column winsorization bounds, fit on train only
_WINSOR_UPPER_PCT = 99.5


def compute_winsor_bounds(
    X_train: pd.DataFrame,
    lower_pct: float = _WINSOR_LOWER_PCT,
    upper_pct: float = _WINSOR_UPPER_PCT,
) -> dict:
    """
    Compute per-column (lower, upper) winsorization bounds from X_train only.

    GENERAL OUTLIER GUARD: the per-feature clips added to normalise_t1_features()
    catch the close-anchored % conversions (price lines / MACD-MOM-AO / ATR /
    slopes), but any OTHER column — multiday features, ratios, anything not
    routed through that function — can still carry a handful of extreme rows
    into StandardScaler.fit(). StandardScaler has no built-in outlier
    resistance: a single blown-out row shifts mean_ and inflates std_ for that
    column, which compresses every other (normal) row's scaled value toward
    zero and quietly destroys signal. This computes a generous per-column
    [0.5th, 99.5th] percentile band from the TRAINING split only (never val —
    that would leak val-set information into the bound), to be applied before
    the scaler ever sees the data.

    Columns that are entirely NaN, or have too few non-NaN values to form a
    meaningful percentile (< 20 observations), are skipped (no bound stored —
    left unclipped, since a bound estimated from a handful of points is
    itself unreliable).
    """
    bounds: dict = {}
    for col in X_train.columns:
        s = X_train[col].dropna()
        if len(s) < 20:
            continue
        lo = float(np.percentile(s, lower_pct))
        hi = float(np.percentile(s, upper_pct))
        if hi <= lo:
            # Degenerate (near-constant) column — nothing meaningful to clip.
            continue
        bounds[col] = (lo, hi)
    return bounds


def apply_winsor_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Clip each column in X to the (lower, upper) bound in `bounds`, if present.
    NaN is preserved (clip leaves NaN untouched). Columns not in `bounds`
    (e.g. skipped as degenerate/too-sparse during fitting) pass through
    unmodified. Safe to call on train (bounds computed from itself, so this
    only trims the very rows that produced the tails) or val/any other split
    (bounds always come from train — no leakage).
    """
    X = X.copy()
    for col, (lo, hi) in bounds.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)
    return X


def build_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame, list, dict]:
    """
    Fit scaler on train-split rows only. Returns scaler, scaled X_train, the
    list of sparse column names determined from training-set coverage, and
    the per-column winsorization bounds used before fitting.

    WINSORIZATION FIX: X_train is winsorized (clipped to per-column [0.5th,
    99.5th] percentile bounds fit on X_train itself) BEFORE the scaler sees
    it, so a handful of extreme-outlier rows in any column can no longer
    drag StandardScaler's mean_/std_ and compress every other row's signal.
    The bounds are returned so the caller can apply the SAME (train-derived)
    bounds to X_val / any other split via apply_winsor_bounds() — never
    re-fit bounds on non-train data, or val information leaks into training.

    LEAKAGE FIX: The scaler is now fit exclusively on X_train so that
    validation-set rows never influence the scaler's mean_ / std_ parameters.
    Call scale_with_fitted_scaler(scaler, X_val, sparse_cols) to transform the
    val set (or any other split) using the same, already-fitted scaler and the
    sparse-column list computed here from the training set.

    NaN RESTORATION FIX (t1_ features): Sparse columns (coverage < SPARSE_THRESHOLD)
    have NaN restored AFTER scaling so XGBoost can use its native missing-value
    branch logic.  Previously fillna(col_mean) → scale → fillna(0.0) made these
    columns appear as the constant 0.0 for 85% of rows, hiding them from gain-based
    feature importance entirely.  StandardScaler still receives NaN-free input
    (required), but XGBoost receives NaN for genuinely absent values, matching the
    inference path in _scale_features() in explosion_predictor.py.

    SPARSE THRESHOLD FIX: sparse_cols is derived exclusively from X_train coverage
    so that the same set of columns is treated as sparse during both training and
    validation transforms.  Previously scale_with_fitted_scaler() re-computed
    coverage from whatever X was passed in, meaning a column that is 60% populated
    in train but 40% in val could be classified differently between splits.
    """
    SPARSE_THRESHOLD = 0.5   # columns with < 50% coverage get NaN restored post-scale

    winsor_bounds = compute_winsor_bounds(X_train)
    n_winsor_cols = len(winsor_bounds)
    X_train_w = apply_winsor_bounds(X_train, winsor_bounds)
    if n_winsor_cols:
        n_cells_clipped = int(
            sum(
                ((X_train[col] < lo) | (X_train[col] > hi)).sum()
                for col, (lo, hi) in winsor_bounds.items()
            )
        )
        logger.info(
            f"Winsorized {n_winsor_cols} columns to [{_WINSOR_LOWER_PCT}, "
            f"{_WINSOR_UPPER_PCT}] percentile bounds (train-fit only); "
            f"{n_cells_clipped} cell(s) clipped before scaler fit."
        )

    scaler        = StandardScaler()
    col_means     = X_train_w.mean()           # computed on winsorized train rows only
    X_filled      = X_train_w.fillna(col_means)
    scaler.fit(X_filled)                     # fit on train rows only — no val leakage

    X_scaled_vals = scaler.transform(X_filled)
    X_scaled      = pd.DataFrame(X_scaled_vals, columns=X_train.columns, index=X_train.index)
    # Fill any remaining NaN (e.g. columns with all-NaN that have no mean) with 0.
    X_scaled      = X_scaled.fillna(0.0)

    # ── Restore NaN for sparse columns so XGBoost uses missing-value branches ──
    # Identify columns with low coverage in the training set. This used to be
    # almost always t1_ intraday columns (NaN for every base-CSV row, back when
    # base rows were a large fraction of the dataset and t1 coverage was
    # inconsistent). With t1_/t3_/t5_/t10_ coverage now near 100%, this set is
    # typically small or empty — coverage is measured fresh from X_train every
    # run, so whatever is genuinely sparse (e.g. a short-history stock missing
    # a long-window indicator) is still caught correctly; nothing here assumes
    # which columns should be sparse.
    # Restoring NaN lets XGBoost route these rows through its learned "missing"
    # branch rather than treating them as "value = column mean", which was
    # previously causing t1_ features to appear near-constant and be ignored.
    coverage = X_train.notna().mean()
    sparse_cols = coverage[coverage < SPARSE_THRESHOLD].index.tolist()
    if sparse_cols:
        nan_mask = X_train[sparse_cols].isna()
        X_scaled.loc[:, sparse_cols] = X_scaled[sparse_cols].where(~nan_mask, other=np.nan)
        logger.info(
            f"NaN restored for {len(sparse_cols)} sparse columns "
            f"(coverage < {SPARSE_THRESHOLD:.0%}) so XGBoost uses native missing-value branches. "
            f"Examples: {sparse_cols[:5]}"
        )

    return scaler, X_scaled, sparse_cols, winsor_bounds


def scale_with_fitted_scaler(
    scaler: StandardScaler,
    X: pd.DataFrame,
    sparse_threshold_cols: list | None = None,
    sparse_threshold: float = 0.5,
    winsor_bounds: dict | None = None,
) -> pd.DataFrame:
    """
    Transform X using an already-fitted scaler (e.g. to scale the val set or
    to reassemble a full scaled DataFrame for the gain regressor).

    WINSORIZATION FIX: pass the SAME winsor_bounds dict returned by
    build_scaler() (fit on X_train only) so X is clipped the same way before
    being filled/transformed — keeps val (and any other split) on consistent
    footing with what the scaler was actually fit on, with no re-fitting of
    bounds on non-train data.

    NaN RESTORATION FIX: mirrors build_scaler — sparse columns have NaN restored
    after scaling so XGBoost receives genuinely missing values rather than 0.0.

    SPARSE THRESHOLD FIX: Pass ``sparse_threshold_cols`` (the third return value
    of ``build_scaler``) so that sparse-column membership is determined from the
    training-set coverage rather than re-computed from the coverage of whatever X
    is passed in here.  A column that is 60% populated in train but 40% in val
    would otherwise be classified differently between splits, causing NaN
    restoration to differ between training and inference paths.

    Fallback: if ``sparse_threshold_cols`` is None (e.g. calling legacy code that
    hasn't been updated yet), coverage is re-computed from X as before and a
    DeprecationWarning is logged so callers know to pass the list.
    """
    X_w = apply_winsor_bounds(X, winsor_bounds) if winsor_bounds else X

    col_means = pd.Series(scaler.mean_, index=X.columns)
    X_filled  = X_w.fillna(col_means)

    X_scaled_vals = scaler.transform(X_filled)
    X_scaled      = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)
    X_scaled      = X_scaled.fillna(0.0)

    # ── Restore NaN for sparse columns (mirrors build_scaler logic) ────────────
    if sparse_threshold_cols is not None:
        # Preferred path: use the column list determined from train coverage.
        sparse_cols = [c for c in sparse_threshold_cols if c in X.columns]
    else:
        # Legacy fallback: re-compute from the passed-in X.
        # This is kept for backward compatibility but produces inconsistent results
        # when val coverage differs from train coverage.
        logger.warning(
            "scale_with_fitted_scaler called without sparse_threshold_cols — "
            "sparse columns will be inferred from the coverage of the input X, "
            "which may differ from train coverage. Pass the sparse_cols list "
            "returned by build_scaler() to ensure consistent NaN restoration."
        )
        coverage = X.notna().mean()
        sparse_cols = coverage[coverage < sparse_threshold].index.tolist()

    if sparse_cols:
        nan_mask = X[sparse_cols].isna()
        X_scaled.loc[:, sparse_cols] = X_scaled[sparse_cols].where(~nan_mask, other=np.nan)

    return X_scaled

