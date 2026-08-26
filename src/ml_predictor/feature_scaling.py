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

# Heavy-tailed / volume-based columns get a signed log1p transform BEFORE
# winsorization (see select_log_transform_cols() / apply_log_transform()
# below). Matched by substring, case-insensitive, against the column name —
# this is deliberately name-pattern-based rather than a fitted/statistical
# decision so that column membership is 100% deterministic from the column
# name alone and can never disagree between train and predict (unlike the
# percentile-based winsor bounds, there's nothing here that needs to be
# fit on X_train and persisted to be reproduced later).
_LOG_TRANSFORM_NAME_TOKENS = ("obv", "vwap", "volume", "hv_")


def select_log_transform_cols(columns) -> list:
    """
    Return the subset of `columns` that should get a signed log1p transform:
    OBV / volume-based features (t1_open_OBV, t5_obv, Volume_MA*, ...), VWAP
    (t1_close_VWAP, ...), and historical-volatility ratios (t3_hv_10,
    t5_hv_20, ...). These are exactly the columns the CV-AUC review flagged
    as heavy-tailed and only winsorized, not transformed: winsorizing alone
    clips the worst outliers but leaves the right-skew intact (e.g. VWAP
    winsor bounds spanning -56 to 1871, OBV spanning -186 to 3824), which
    still lets a handful of extreme rows dominate split points for trees and
    makes any linear/GLM consumer (e.g. the gain regressor, if ever swapped
    to a linear model) unnecessarily sensitive to scale.

    Matching is by case-insensitive substring against the column name, so it
    applies uniformly across the t1_/t3_/t5_/t10_ prefixed variants and the
    base (un-prefixed) multiday columns alike.
    """
    return [
        col for col in columns
        if any(token in col.lower() for token in _LOG_TRANSFORM_NAME_TOKENS)
    ]


def apply_log_transform(X: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Apply a signed log1p transform to the given columns of X:
        signed_log1p(x) = sign(x) * log1p(|x|)

    Signed rather than plain log1p because these columns (OBV ratios, VWAP
    %-distance-from-close, etc.) can legitimately be negative — plain log1p
    would produce NaN for any negative value. signed_log1p compresses the
    magnitude of both tails symmetrically while preserving sign and leaving
    values near zero close to unchanged (it's the identity to first order
    for |x| << 1), so downstream winsorization percentiles and the trained
    model's notion of "this feature's scale" stay well-behaved instead of
    being stretched across a wildly asymmetric raw range.

    NaN is preserved. Columns not present in X, or not in `cols`, pass
    through unmodified. Applying this twice is NOT idempotent (unlike
    winsorization) — always call it exactly once, before winsorizing/scaling,
    on both the train-fit path and the transform-only path, using the same
    `cols` list (persisted via model_metadata.json — see build_scaler()).
    """
    if not cols:
        return X
    X = X.copy()
    for col in cols:
        if col not in X.columns:
            continue
        num = pd.to_numeric(X[col], errors="coerce")
        X[col] = np.sign(num) * np.log1p(num.abs())
    return X


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


def build_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame, list, dict, list]:
    """
    Fit scaler on train-split rows only. Returns scaler, scaled X_train, the
    list of sparse column names determined from training-set coverage, the
    per-column winsorization bounds used before fitting, and the list of
    columns that got the signed-log1p heavy-tail transform.

    LOG-TRANSFORM FIX: heavy-tailed / volume-based columns (OBV, VWAP,
    historical-volatility ratios — see select_log_transform_cols()) are
    signed-log1p transformed BEFORE winsor bounds are computed or applied.
    Winsorizing alone clips the worst outliers but leaves the underlying
    right-skew intact (e.g. raw OBV/VWAP winsor bounds spanning hundreds to
    thousands on one side and tens on the other), which still lets a
    handful of extreme rows dominate tree split points and makes any
    linear/GLM consumer more scale-sensitive than necessary. Doing this
    ahead of winsorization means the percentile bounds themselves are
    computed on the compressed, more symmetric distribution. The column
    list is returned so the caller can apply the SAME transform (via
    apply_log_transform()) to X_val / any other split — column membership
    is name-pattern-based so it's already deterministic, but the list is
    still threaded through and persisted (model_metadata.json) so training
    and prediction can never disagree about which columns were transformed.

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

    log_transform_cols = select_log_transform_cols(X_train.columns)
    X_train_log = apply_log_transform(X_train, log_transform_cols)
    if log_transform_cols:
        logger.info(
            f"Signed-log1p transformed {len(log_transform_cols)} heavy-tailed "
            f"column(s) before winsorizing/scaling: {log_transform_cols}"
        )

    winsor_bounds = compute_winsor_bounds(X_train_log)
    n_winsor_cols = len(winsor_bounds)
    X_train_w = apply_winsor_bounds(X_train_log, winsor_bounds)
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

    return scaler, X_scaled, sparse_cols, winsor_bounds, log_transform_cols


def scale_with_fitted_scaler(
    scaler: StandardScaler,
    X: pd.DataFrame,
    sparse_threshold_cols: list | None = None,
    sparse_threshold: float = 0.5,
    winsor_bounds: dict | None = None,
    log_transform_cols: list | None = None,
) -> pd.DataFrame:
    """
    Transform X using an already-fitted scaler (e.g. to scale the val set or
    to reassemble a full scaled DataFrame for the gain regressor).

    LOG-TRANSFORM FIX: pass the SAME log_transform_cols list returned by
    build_scaler() so heavy-tailed columns (OBV/VWAP/historical-volatility —
    see select_log_transform_cols()) get the identical signed-log1p
    transform applied here, BEFORE winsorizing/scaling, that they got on the
    training split. Column membership is name-pattern-based (deterministic
    from the column name alone) so this list will already agree with
    build_scaler()'s even if not explicitly passed through, but callers
    should always thread it through explicitly (persisted in
    model_metadata.json) rather than re-deriving it, for the same
    train/predict-must-never-drift reason winsor_bounds is threaded through.

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
    X_log = apply_log_transform(X, log_transform_cols) if log_transform_cols else X
    X_w = apply_winsor_bounds(X_log, winsor_bounds) if winsor_bounds else X_log

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


# ---------------------------------------------------------------------------
# Unit normalisation (dollar-scale / raw-count -> scale-free t1_ features)
# ---------------------------------------------------------------------------
# Moved here from ml_retrain_model.py so explosion_predictor.py (prediction)
# can call the EXACT same detection/conversion logic training uses, instead
# of relying purely on the assumption that intraday_data_collector.py always
# normalises at collection time. Training self-heals legacy raw-dollar rows
# via this function; prediction previously had no equivalent runtime guard,
# so a collector regression could silently feed a raw-dollar value into the
# scaler at inference with no detection. Both callers now share one
# implementation, same pattern as build_scaler()/scale_with_fitted_scaler().

def normalise_t1_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Ensure all T-1 feature columns that carry dollar-scale or cumulative-volume
    values are expressed in the same scale-free units that multiday_feature_collector
    writes to the DB (and that the model was trained to expect).

    WHY THIS IS NEEDED
    ------------------
    The intraday_data_collector was originally storing raw dollar values for
    MAs, Bollinger Bands, MACD, etc.  A fix to the collector now normalises
    those values at collection time, but rows already in the DB are raw.
    This function detects — per column, per batch — whether each feature is
    already normalised and skips it if so, making the operation idempotent:
    running it on already-normalised data is a no-op.

    NORMALISATION CATEGORIES (mirror of multiday_feature_collector.py)
    -------------------------------------------------------------------
    A. Price lines  → (value / close - 1) * 100   [% distance from close]
       SMA_5/10/20/50, EMA_5/10/12/20/26/50, BB lower/middle/upper,
       Keltner lower/middle/upper, Donchian lower/middle/upper, VWAP.

    B. Dollar diffs → value / close * 100          [% of close]
       MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9, MOM_10, AO.

    C. ATR          → value / close * 100          [% of close]
       ATR_14.  multiday stores ATRr (already %) but intraday's `ta` library
       returns dollar ATR, so the same normalisation is required.

    D. Volume       → value / Volume_MA20          [ratio]
       Volume_MA5, Volume_MA10, OBV.
       Volume_MA20 / Volume_MA20 = 1.0 always (sentinel value; set to 1.0).

    DERIVED RE-COMPUTATION
    ----------------------
    Four derived columns depend on the raw MA values and must be re-derived
    after normalisation so they are consistent with their inputs:
       EMA_20_Slope   = diff(EMA_20)            after normalisation
       SMA_20_Slope   = diff(SMA_20)            after normalisation
       EMA_12_26_Diff = EMA_12 − EMA_26         after normalisation
       SMA_20_50_Diff = SMA_20 − SMA_50         after normalisation
       Price_vs_SMA20 = −SMA_20                 (= 0 − normalised_SMA_20)
       Price_vs_SMA50 = −SMA_50
       Price_vs_EMA20 = −EMA_20
       ATR_14_Slope   = diff(ATR_14)            after normalisation

    DETECTION LOGIC
    ---------------
    Detection for groups A/B/C/D is PER ROW, not per-column-median. A batch
    fetched from the DB can contain rows from multiple ingestion runs — some
    already normalised (post-collector-fix), some still raw (pre-fix /
    older rows) — mixed together in the same column. A column-median check
    decides raw-vs-normalised for the WHOLE column at once, so whichever
    state is in the majority wins and the minority rows are silently left
    on the wrong scale (this was already identified and fixed for group D;
    it applied equally to A/B/C and is now fixed the same way there). Each
    row's own value/close ratio (or, for group D, its own Volume_MA20)
    decides that row's raw/normalised state, so a column that is 90% clean
    and 10% raw gets exactly the 10% fixed. The close price column (e.g.
    t1_close_Close) is used as the per-row anchor for price-relative checks.

    Each group has its own detection rule (now applied per row):

      Price lines: a normalised MA sits within ±50% of close (i.e. in the
        range −50 to +50).  A raw MA equals the close price itself — so its
        median absolute value ≈ median(close).  Threshold: if the median
        absolute value of the column exceeds 50 (which is impossible for a
        correctly normalised % distance), the column is raw.

      Dollar diffs (MACD/MOM/AO): normalised values are small percentages,
        typically ±5%.  Raw values are dollar differences whose magnitude
        scales with close price.  Threshold: if median(|col|) > 20 the column
        is raw.  (A normalised MACD of 20% of close would be extreme.)

      ATR: normalised ATR is 1–20% for volatile small caps.  Dollar ATR for
        a $5 stock may be $0.30 (= 6% normalised) — difficult to distinguish
        just from magnitude.  Use close as anchor: if median(ATR) > 0.5 *
        median(close) the column must be raw (a 50%+ daily ATR is impossible).
        Fallback: if median(ATR) > 50 it is certainly raw.

      Volume MAs / OBV: after normalisation Volume_MA20 is exactly 1.0 and
        Volume_MA5/MA10/OBV are ratios near 1.0. Unlike every other group
        above, this detection is done PER ROW rather than on the column
        median: a row is raw iff its own Volume_MA20 value is > 100 (raw
        share counts are always > 100; normalised Volume_MA20 is always
        exactly 1.0). This matters because a single DataFrame can contain
        rows from multiple ingestion batches — some already normalised,
        some not — and a column-median threshold would normalise-or-skip
        the *entire* column based on whichever state is in the majority,
        silently leaving the minority batch's rows on a raw share-count
        scale in the same column as rows pinned at the 1.0 sentinel.

    Args:
        df:     DataFrame as returned by rename_t1_columns — column names
                already carry the full prefix (e.g. 't1_close_SMA_20').
        prefix: 't1_close' or 't1_open' — used to locate the close-price
                anchor column and to build the full column names.

    Returns:
        df with unnormalised columns normalised in-place (copy returned).
    """
    if df.empty:
        return df

    df = df.copy()

    close_col = f"{prefix}_Close"
    vol_ma20_col = f"{prefix}_Volume_MA20"

    # --- close anchor (Series of per-row close prices) ----------------------
    # Used for price-relative normalisation; may be absent for old DB rows.
    close_s = (
        pd.to_numeric(df[close_col], errors="coerce")
        if close_col in df.columns
        else None
    )
    safe_close = close_s.replace(0, np.nan) if close_s is not None else None
    median_close = float(close_s.median()) if close_s is not None and close_s.notna().any() else None

    # --- volume anchor -------------------------------------------------------
    vol_ma20_s = (
        pd.to_numeric(df[vol_ma20_col], errors="coerce")
        if vol_ma20_col in df.columns
        else None
    )

    normalised_count = 0
    skipped_count    = 0

    def _median_abs(col_name: str) -> float:
        """Median of the absolute values of a column; NaN if column absent."""
        if col_name not in df.columns:
            return float("nan")
        s = pd.to_numeric(df[col_name], errors="coerce").dropna()
        return float(s.abs().median()) if len(s) else float("nan")

    # ------------------------------------------------------------------
    # PER-ROW raw/normalised masks (groups A/B/C)
    # ------------------------------------------------------------------
    # FIX (root cause #1): the previous _is_raw_* helpers decided "raw vs
    # normalised" once per column, from the column's median. That's fine
    # when a batch is homogeneous, but this table accumulates rows from
    # multiple ingestion runs — some collected before the intraday
    # collector's normalisation fix, some after. If most rows in a given
    # retrain's batch are already normalised, the column median sits in
    # the normalised range and the whole column is skipped, silently
    # leaving the raw-dollar minority rows untouched in the same column
    # (this is exactly the failure mode already documented and fixed for
    # Volume group D below — vol_needs_norm going False and stranding the
    # raw share-count rows). That mixed-scale minority is what produced
    # the blown-out tails in model_metadata.json (median ≈ correct,
    # min/max in the tens of thousands).
    #
    # The fix mirrors group D's approach: every row decides its OWN raw/
    # normalised state, using the same close-anchored ratio test the old
    # code applied to the column as a whole. Rows are normalised (or left
    # alone) individually, so a column that's 90% clean and 10% raw gets
    # exactly the 10% fixed instead of the whole column being judged by
    # majority vote.
    def _raw_price_line_mask(col_name: str) -> "pd.Series | None":
        """Per-row boolean mask: True where the row's own value looks like a
        raw dollar price line rather than a % distance from close."""
        if col_name not in df.columns or safe_close is None:
            return None
        num   = pd.to_numeric(df[col_name], errors="coerce")
        ratio = num / safe_close
        # Applying the old column-median window (0.3x-3.0x of close) per row
        # instead of per-column-median turns out to be ambiguous on a
        # per-row basis: a legitimately normalised, wide "% distance from
        # close" value (e.g. -20, on a $10 stock) has |value/close| = 2.0,
        # which lands inside the same 0.3-3.0x window a genuinely raw price
        # line would. The column median averaged this collision away; a
        # per-row check needs an extra discriminator to avoid re-corrupting
        # already-clean rows:
        #   1) a raw dollar price line is always POSITIVE (MAs/bands are
        #      never negative); a normalised "% distance from close" value
        #      is frequently negative. Requiring num > 0 rules out roughly
        #      half of the collision zone for free.
        #   2) an MA/band rarely strays more than ~2x above/below its own
        #      close in practice (that would imply a >100% multi-day move),
        #      so tightening the window to 0.5x-2.0x (from 0.3x-3.0x)
        #      shrinks the overlap with genuinely wide normalised values
        #      (which cluster in the -50..50 range and increasingly rarely
        #      land at exactly 0.5x-2.0x of that row's own close) without
        #      giving up real raw-price detection, which sits at ~1.0x by
        #      construction.
        return (num > 0) & (ratio.abs().between(0.5, 2.0))

    def _raw_dollar_diff_mask(col_name: str) -> "pd.Series | None":
        """Per-row boolean mask: True where MACD/MOM/AO looks like a raw
        dollar difference rather than a % of close."""
        if col_name not in df.columns:
            return None
        num = pd.to_numeric(df[col_name], errors="coerce")
        return num.abs() > 20.0

    def _raw_atr_mask(col_name: str) -> "pd.Series | None":
        """Per-row boolean mask: True where ATR looks like raw dollar ATR
        rather than % of close."""
        if col_name not in df.columns:
            return None
        num = pd.to_numeric(df[col_name], errors="coerce")
        mask = num.abs() > 50.0
        if safe_close is not None:
            mask = mask | (num.abs() > safe_close * 0.5)
        return mask

    def _raw_slope_mask(col_name: str) -> "pd.Series | None":
        """Per-row boolean mask for the Slope columns (SMA_20_Slope,
        EMA_20_Slope, ATR_14_Slope): True where the row's value looks like a
        raw dollar-scale slope (from a pre-fix collector row) rather than a
        %-point change already computed on normalised units.

        These three columns arrive in `df` ALREADY POPULATED by
        rename_t1_columns() — the intraday collector computes them as a
        genuine diff(1) over its own continuous per-symbol bar series (see
        `sma20_slope`/`ema20_slope`/`atr_pct` in
        intraday_data_collector.py) and that value is correct as collected.
        The only thing that can be wrong with it is units: pre-fix collector
        rows computed the diff BEFORE normalising the underlying MA/ATR, so
        their slope is a raw dollar-per-bar change, not a %-point change.
        Same magnitude heuristic as the other dollar-diff columns (group B).
        """
        if col_name not in df.columns:
            return None
        num = pd.to_numeric(df[col_name], errors="coerce")
        return num.abs() > 20.0

    # ------------------------------------------------------------------
    # Column-level helpers, now used ONLY to decide whether the *derived*
    # re-computation columns further down are safe to rebuild (they need
    # to know "is this base column normalised at all, anywhere in this
    # batch" — a coarser question than the per-row masks above). Kept as
    # medians deliberately: a single stray raw row shouldn't block
    # re-deriving the whole Slope/Diff column, since that column is about
    # to be rebuilt per-row from the (now per-row-normalised) base column
    # anyway.
    def _is_raw_price_line(col_name: str) -> bool:
        if col_name not in df.columns or safe_close is None:
            return False
        num   = pd.to_numeric(df[col_name], errors="coerce")
        ratio = (num / safe_close).dropna()
        if ratio.empty:
            return False
        med_ratio = float(ratio.abs().median())
        return 0.3 <= med_ratio <= 3.0

    def _is_raw_dollar_diff(col_name: str) -> bool:
        """True if MACD/MOM/AO column looks like a raw dollar difference."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        return med > 20.0

    def _is_raw_atr(col_name: str) -> bool:
        """True if ATR column looks like raw dollar ATR rather than % of close."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        if med > 50.0:
            return True
        if median_close is not None and median_close > 0:
            return med > median_close * 0.5
        return False

    def _is_raw_volume_ma(col_name: str) -> bool:
        """True if a Volume_MA column is still in raw share counts."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        # After normalisation Volume_MA5/10 are ratios near 1.0; Volume_MA20 = 1.0.
        # Raw share counts are always > 100 (even the most thinly traded stocks).
        return med > 100.0

    def _is_raw_obv(col_name: str) -> bool:
        """True if OBV is in raw cumulative share counts rather than ÷ vol_ma20."""
        med = _median_abs(col_name)
        if np.isnan(med):
            return False
        # Normalised OBV is a ratio to daily avg volume — typically ±200.
        # Raw OBV for even a thinly traded stock easily hits tens of thousands.
        return med > 10_000.0

    # ── Outlier guard for close-anchored % conversions ──────────────────────
    # BUG FIX: safe_close only guards against close == 0, not close ≈ 0. For
    # sub-penny stocks (close e.g. $0.0002) an ordinary dollar-scale move
    # divided by that close explodes into a percentage in the tens of
    # thousands, even though the underlying value was never "raw" in a
    # meaningful sense. That contaminates X_train_raw fed to build_scaler()
    # (StandardScaler's mean_/std_ get dragged by a handful of these rows,
    # compressing genuine signal for every normal-priced stock) and
    # contaminates the top10_training_stats percentile snapshot saved to
    # model_metadata.json — which explosion_predictor.py then uses as the
    # live clip bounds (percentiles[0], percentiles[-1]), so the guard rail
    # itself becomes [-hundreds_of_thousands, +thousands] and stops guarding
    # anything. Clip every close-anchored % conversion to a generous but
    # finite band immediately after computing it, before it can reach the
    # scaler fit or the metadata snapshot.
    _PRICE_LINE_CLIP_PCT = 100.0   # % distance from close
    _DOLLAR_DIFF_CLIP_PCT = 200.0  # % of close (MACD/MOM/AO/slopes)
    _ATR_CLIP_PCT = 100.0          # % of close

    def _clip_inplace(mask: "pd.Series", col_name: str, bound: float) -> None:
        vals = pd.to_numeric(df.loc[mask, col_name], errors="coerce")
        clipped = vals.clip(-bound, bound)
        n_clipped = int((clipped != vals).sum())
        df.loc[mask, col_name] = clipped
        if n_clipped:
            logger.debug(
                f"  normalise_t1: {col_name} → clipped {n_clipped} row(s) to "
                f"±{bound} after % conversion (sub-penny-close outlier guard)"
            )

    # ── A. Price lines → (value / close − 1) × 100 ──────────────────────────
    PRICE_LINE_COLS = [
        f"{prefix}_SMA_5",   f"{prefix}_SMA_10",  f"{prefix}_SMA_20",  f"{prefix}_SMA_50",
        f"{prefix}_EMA_5",   f"{prefix}_EMA_10",  f"{prefix}_EMA_12",
        f"{prefix}_EMA_20",  f"{prefix}_EMA_26",  f"{prefix}_EMA_50",
        f"{prefix}_BBL_20_2.0_2.0", f"{prefix}_BBM_20_2.0_2.0", f"{prefix}_BBU_20_2.0_2.0",
        f"{prefix}_KCLe_20_2",      f"{prefix}_KCBe_20_2",      f"{prefix}_KCUe_20_2",
        f"{prefix}_DCL_20_20",      f"{prefix}_DCM_20_20",      f"{prefix}_DCU_20_20",
        f"{prefix}_VWAP",
    ]

    for col in PRICE_LINE_COLS:
        if col not in df.columns:
            continue
        raw_mask = _raw_price_line_mask(col)
        if raw_mask is None or not raw_mask.any():
            skipped_count += 1
            continue
        if safe_close is not None:
            n_raw = int(raw_mask.sum())
            num = pd.to_numeric(df[col], errors="coerce")
            df.loc[raw_mask, col] = (
                num.loc[raw_mask] / safe_close.loc[raw_mask] - 1
            ) * 100
            _clip_inplace(raw_mask, col, _PRICE_LINE_CLIP_PCT)
            normalised_count += 1
            logger.debug(
                f"  normalise_t1: {col} → % dist from close ({n_raw} raw row(s))"
            )
        else:
            logger.warning(
                f"  normalise_t1: {col} appears raw but {close_col} is absent "
                "— cannot normalise.  Rows without a close price will have NaN."
            )

    # ── B. Dollar diffs → value / close × 100 ────────────────────────────────
    DOLLAR_DIFF_COLS = [
        f"{prefix}_MACD_12_26_9",
        f"{prefix}_MACDh_12_26_9",
        f"{prefix}_MACDs_12_26_9",
        f"{prefix}_MOM_10",
        f"{prefix}_AO",
    ]

    for col in DOLLAR_DIFF_COLS:
        if col not in df.columns:
            continue
        raw_mask = _raw_dollar_diff_mask(col)
        if raw_mask is None or not raw_mask.any():
            skipped_count += 1
            continue
        if safe_close is not None:
            n_raw = int(raw_mask.sum())
            num = pd.to_numeric(df[col], errors="coerce")
            df.loc[raw_mask, col] = num.loc[raw_mask] / safe_close.loc[raw_mask] * 100
            _clip_inplace(raw_mask, col, _DOLLAR_DIFF_CLIP_PCT)
            normalised_count += 1
            logger.debug(
                f"  normalise_t1: {col} → % of close ({n_raw} raw row(s))"
            )
        else:
            logger.warning(
                f"  normalise_t1: {col} appears raw but {close_col} is absent."
            )

    # ── C. ATR → value / close × 100 ─────────────────────────────────────────
    atr_col = f"{prefix}_ATR_14"
    if atr_col in df.columns:
        raw_mask = _raw_atr_mask(atr_col)
        if raw_mask is not None and raw_mask.any():
            if safe_close is not None:
                n_raw = int(raw_mask.sum())
                num = pd.to_numeric(df[atr_col], errors="coerce")
                df.loc[raw_mask, atr_col] = num.loc[raw_mask] / safe_close.loc[raw_mask] * 100
                _clip_inplace(raw_mask, atr_col, _ATR_CLIP_PCT)
                normalised_count += 1
                logger.debug(
                    f"  normalise_t1: {atr_col} → % of close ({n_raw} raw row(s))"
                )
            else:
                logger.warning(
                    f"  normalise_t1: {atr_col} appears raw but {close_col} is absent."
                )
        else:
            skipped_count += 1

    # ── D. Volume → value / Volume_MA20 ──────────────────────────────────────
    #
    # IMPORTANT: detection here MUST be done PER-ROW, not per-column-median.
    # A DataFrame passed to this function can contain a mix of rows from
    # different ingestion batches — some already normalised (post-fix
    # collector), some still raw (pre-fix collector / old DB rows). A
    # column-level median check decides "raw vs normalised" for the WHOLE
    # column at once, so whichever state is in the majority wins and the
    # minority rows are left untouched. In particular, if most rows are
    # already normalised, `vol_needs_norm` comes back False and the ~10%
    # still-raw rows keep their raw share-count values (hundreds of
    # thousands to millions) sitting in the same column as rows pinned at
    # the 1.0 sentinel — a mixed-scale column that leaks ingestion-batch
    # identity into the model instead of carrying real signal. Detecting
    # and normalising per row (using each row's own Volume_MA20 value as
    # the raw/normalised signal, since raw Volume_MA20 is always > 100 and
    # normalised Volume_MA20 is always exactly 1.0) fixes this regardless
    # of how the batches are mixed.
    vol_ma5_col  = f"{prefix}_Volume_MA5"
    vol_ma10_col = f"{prefix}_Volume_MA10"
    obv_col      = f"{prefix}_OBV"

    if vol_ma20_s is not None:
        # Per-row raw mask: a row is "raw" iff its OWN Volume_MA20 value is
        # still a raw share count (> 100). Post-normalisation Volume_MA20
        # is always exactly 1.0, so this check is self-consistent and
        # idempotent — already-normalised rows are correctly left alone.
        raw_row_mask = vol_ma20_s > 100.0
        n_raw = int(raw_row_mask.sum())

        if n_raw > 0:
            # Use each raw row's own (pre-overwrite) Volume_MA20 as the
            # denominator for that row.
            denom = vol_ma20_s.where(raw_row_mask).replace(0, np.nan)

            for col in [vol_ma5_col, vol_ma10_col, obv_col]:
                if col in df.columns:
                    num = pd.to_numeric(df[col], errors="coerce")
                    df.loc[raw_row_mask, col] = (
                        num.loc[raw_row_mask] / denom.loc[raw_row_mask]
                    )
                    normalised_count += 1
                    logger.debug(
                        f"  normalise_t1: {col} → ratio to vol_ma20 "
                        f"({n_raw} raw rows)"
                    )

            # Volume_MA20 / itself = 1.0 always (matches multiday behaviour).
            # Only overwrite the rows that were actually raw — already
            # normalised rows are already 1.0 and are left untouched.
            df.loc[raw_row_mask, vol_ma20_col] = 1.0
            normalised_count += 1
            logger.debug(
                f"  normalise_t1: {vol_ma20_col} → 1.0 sentinel "
                f"({n_raw} raw rows)"
            )
        else:
            skipped_count += 1   # every row already normalised
    else:
        logger.warning(
            f"  normalise_t1: Volume MAs/OBV appear raw but {vol_ma20_col} is "
            "absent — cannot normalise volume features."
        )

    # ── Derived re-computation ────────────────────────────────────────────────
    # These were computed before normalisation in the old collector and are
    # therefore in raw-dollar units.  Re-derive them from the now-normalised
    # base columns so they are consistent with multiday behaviour.
    #
    # We only re-derive when the base columns exist AND are now normalised
    # (i.e. median absolute value ≤ 50 after the steps above).

    ema20 = f"{prefix}_EMA_20"
    sma20 = f"{prefix}_SMA_20"
    ema12 = f"{prefix}_EMA_12"
    ema26 = f"{prefix}_EMA_26"
    sma50 = f"{prefix}_SMA_50"
    atr14 = f"{prefix}_ATR_14"

    # ------------------------------------------------------------------
    # FIX (root cause #2, revised): the original fix here computed Slope
    # via `.diff(1)` grouped by symbol over THIS table — the winners/
    # non-winners label-event table, which is one row per (symbol,
    # detection_date), not a continuous daily series. That was strictly
    # better than the original ungrouped diff(1) (which mixed unrelated
    # symbols), but it introduced a NEW problem: a symbol only has a
    # valid "previous row" to diff against if it happens to appear twice
    # in THIS table. Winners occasionally repeat (~57% still got a real
    # value); non-winners are overwhelmingly one-off "checked and
    # rejected" events, so ~100% of non-winner rows got NaN. That's
    # label-correlated missingness — the model can trivially learn
    # "Slope is NaN => non-winner", which is leakage of a different
    # flavor than the one we were trying to fix (confirmed empirically:
    # diagnose_feature_leakage.py's NaN-rate-gap check went from clean to
    # a ~43pp gap on these exact columns after that change).
    #
    # The actual fix: don't re-derive Slope from THIS table at all. The
    # intraday collector already computes a genuine slope — diff(1) over
    # its own continuous per-symbol bar series (see
    # intraday_data_collector.py: sma20_slope/ema20_slope/atr_pct) — and
    # that value arrives in `df` pre-populated via rename_t1_columns(),
    # BEFORE this function even runs. The only thing that can be wrong
    # with it is units: pre-fix collector rows computed that diff before
    # normalising the underlying MA, so it's occasionally a raw
    # dollar-per-bar change instead of a %-point change. So treat it like
    # every other dollar-diff column (group B: MACD/MOM/AO) — rescale the
    # raw-looking rows in place, per row, and leave everything else
    # (including every already-normalised row) untouched. No diff
    # recomputation, no dependency on a "previous row" existing anywhere,
    # so no way for this to introduce missingness that didn't already
    # exist in the source column.
    def _rescale_slope_col(col_name: str) -> None:
        if col_name not in df.columns:
            return
        raw_mask = _raw_slope_mask(col_name)
        if raw_mask is None or not raw_mask.any():
            return
        if safe_close is None:
            logger.warning(
                f"  normalise_t1: {col_name} appears raw but {close_col} is "
                "absent — cannot rescale."
            )
            return
        n_raw = int(raw_mask.sum())
        num = pd.to_numeric(df[col_name], errors="coerce")
        df.loc[raw_mask, col_name] = num.loc[raw_mask] / safe_close.loc[raw_mask] * 100
        _clip_inplace(raw_mask, col_name, _DOLLAR_DIFF_CLIP_PCT)
        logger.debug(
            f"  normalise_t1: {col_name} → % of close, in place "
            f"({n_raw} raw row(s), no re-derivation)"
        )

    ema20_slope_col = f"{prefix}_EMA_20_Slope"
    _rescale_slope_col(ema20_slope_col)

    sma20_slope_col = f"{prefix}_SMA_20_Slope"
    _rescale_slope_col(sma20_slope_col)

    # EMA_12_26_Diff and SMA_20_50_Diff: spread of normalised MAs
    ema_diff_col = f"{prefix}_EMA_12_26_Diff"
    if ema12 in df.columns and ema26 in df.columns:
        if not _is_raw_price_line(ema12) and not _is_raw_price_line(ema26):
            df[ema_diff_col] = (
                pd.to_numeric(df[ema12], errors="coerce") -
                pd.to_numeric(df[ema26], errors="coerce")
            )
            logger.debug(f"  normalise_t1: re-derived {ema_diff_col}")

    sma_diff_col = f"{prefix}_SMA_20_50_Diff"
    if sma20 in df.columns and sma50 in df.columns:
        if not _is_raw_price_line(sma20) and not _is_raw_price_line(sma50):
            df[sma_diff_col] = (
                pd.to_numeric(df[sma20], errors="coerce") -
                pd.to_numeric(df[sma50], errors="coerce")
            )
            logger.debug(f"  normalise_t1: re-derived {sma_diff_col}")

    # Price_vs_MA: base (multiday_feature_collector.py) defines this as
    # (close - MA) / MA * 100 = (close/MA - 1) * 100 — i.e. it divides by
    # the MA, not by close. The stored MA columns here are normalised to
    # "% distance from close": normalised_MA = (MA/close - 1)*100, so
    # MA/close = 1 + normalised_MA/100. Negating (−normalised_MA) instead
    # gives (close-MA)/close*100, which divides by CLOSE — only matching
    # base when close ~= MA and diverging on large moves. Derive
    # algebraically instead so both paths divide by MA:
    #   (close/MA - 1)*100 = (1 / (1 + normalised_MA/100) - 1) * 100
    for ma_col, vs_col in [
        (sma20, f"{prefix}_Price_vs_SMA20"),
        (sma50, f"{prefix}_Price_vs_SMA50"),
        (ema20, f"{prefix}_Price_vs_EMA20"),
    ]:
        if ma_col in df.columns and not _is_raw_price_line(ma_col):
            _norm_ma = pd.to_numeric(df[ma_col], errors="coerce")
            df[vs_col] = (1.0 / (1.0 + _norm_ma / 100.0) - 1.0) * 100.0
            logger.debug(f"  normalise_t1: re-derived {vs_col}")

    # ATR_14_Slope: same treatment as EMA/SMA Slope above — the collector's
    # own value is already correct, rescale it per row instead of
    # re-deriving it from this event-level table.
    atr_slope_col = f"{prefix}_ATR_14_Slope"
    _rescale_slope_col(atr_slope_col)

    if normalised_count > 0 or skipped_count > 0:
        logger.info(
            f"  normalise_t1 [{prefix}]: "
            f"{normalised_count} column(s) normalised, "
            f"{skipped_count} already normalised (skipped)."
        )

    return df
