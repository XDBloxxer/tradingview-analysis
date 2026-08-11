"""
symbol_demeaning.py — causal per-symbol demeaning for fingerprint-prone features
==================================================================================

Added 2026-08-11 in response to diagnose_symbol_fingerprint_leak.py:
HV_10/20/30 carry a large between-symbol ("which stock is this") component
(AUC 0.79-0.84) alongside a smaller but real within-symbol day-to-day signal
(AUC 0.55-0.60). Feeding the raw indicator to every model lets it substitute
"which stock" for "is this stock's vol elevated today right now".

This module does NOT touch data collection and does NOT block the feature.
It's a shared post-processing step, called from exactly two places so every
model (training AND live scoring) sees the same transformed value:

  - ml_retrain_model.py:prepare_features() calls demean_training_features()
    on the historical training frame, then main() persists each symbol's
    latest trailing mean via compute_symbol_baselines()/save_symbol_baselines().
  - src/ml_predictor/explosion_predictor.py:ExplosionPredictor.prepare_features()
    calls demean_live_features() on the live row(s), using
    load_symbol_baselines() to pull the baseline saved at the last retrain.

Net effect: the model is trained on, and scored on, "this symbol's HV minus
its own typical HV" instead of raw HV — collapsing the fingerprint component
while keeping the real within-symbol signal intact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_predictor.feature_selection import _base_feature_name

logger = logging.getLogger(__name__)

# Bases to demean by default — see diagnose_symbol_fingerprint_leak.py output.
# Extend this tuple (not individual call sites) if other bases turn out to be
# similarly symbol-fingerprinted (e.g. BBB_20_2.0_2.0 was also flagged there
# but is left out for now — HV is the confirmed, requested case).
DEFAULT_DEMEAN_BASES = ("HV_10", "HV_20", "HV_30")

DEFAULT_BASELINE_PATH = "ml_models/feature_selection/symbol_hv_baselines.json"


def _matching_columns(columns, bases) -> list:
    targets = {b.lower() for b in bases}
    return [c for c in columns if _base_feature_name(c) in targets]


def demean_training_features(
    X: pd.DataFrame,
    symbols: pd.Series,
    dates: pd.Series,
    bases: tuple = DEFAULT_DEMEAN_BASES,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Causally demean every lag/side variant of `bases` in X.

    Each row's demeaned value = raw_value − (that symbol's expanding mean of
    the same column, computed from STRICTLY EARLIER rows only — via
    .shift(1) on the per-symbol expanding mean, so a row is never demeaned
    against itself and never sees same-day-or-later values).

    Rows are ordered by (symbol, date, original row order) before the
    expanding mean is computed, so this is safe even if `X`/`symbols`/`dates`
    arrive in an arbitrary (e.g. Supabase fetch) order.

    Cold-start rows (a symbol's first appearance in the training window, no
    prior history) have no trailing mean to subtract. They're left at their
    RAW value rather than NaN'd out — so a symbol's earliest rows aren't
    silently dropped from training. This is a known, small imperfection
    (first-appearance rows keep their fingerprint component) that shrinks as
    more history accumulates per symbol across retrains.
    """
    cols = _matching_columns(X.columns, bases)
    if not cols:
        logger.info(f"[symbol-demean] none of {bases} found in X columns — nothing to do")
        return X

    if len(symbols) != len(X) or len(dates) != len(X):
        logger.warning(
            "[symbol-demean] symbols/dates length mismatch with X — skipping demeaning "
            "for this run (leaving raw values in place)"
        )
        return X

    X = X.copy()
    order = pd.DataFrame(
        {
            "symbol": pd.Series(symbols).values,
            "date": pd.to_datetime(pd.Series(dates).values, errors="coerce"),
        },
        index=X.index,
    )
    order["_orig_order"] = np.arange(len(order))
    sort_idx = order.sort_values(["symbol", "date", "_orig_order"]).index

    n_cold_start = 0
    for col in cols:
        s = X.loc[sort_idx, col]
        grp = s.groupby(order.loc[sort_idx, "symbol"])
        trailing_mean = grp.apply(lambda g: g.expanding(min_periods=min_periods).mean().shift(1))
        if isinstance(trailing_mean.index, pd.MultiIndex):
            trailing_mean.index = trailing_mean.index.droplevel(0)
        trailing_mean = trailing_mean.reindex(sort_idx)

        cold_start_mask = trailing_mean.isna()
        n_cold_start += int(cold_start_mask.sum())

        demeaned = s - trailing_mean
        demeaned[cold_start_mask] = s[cold_start_mask]  # keep raw where no prior history

        X.loc[sort_idx, col] = demeaned.values

    logger.info(
        f"[symbol-demean] causally demeaned {len(cols)} column(s) for bases {bases} "
        f"({n_cold_start} cold-start cell(s) across those columns kept raw — no prior "
        f"history for that symbol yet)"
    )
    return X


def compute_symbol_baselines(
    df: pd.DataFrame,
    symbols: pd.Series,
    bases: tuple = DEFAULT_DEMEAN_BASES,
) -> dict:
    """
    Compute each symbol's mean AS OF THE LATEST ROW IN `df`, per lag/side
    column, for persisting and reuse at live-inference time (where there's
    no in-process history to compute an expanding mean from — a live scoring
    run typically sees one row per symbol).

    This uses the mean of ALL of that symbol's rows in `df` (the full
    training frame) — i.e. it's the baseline a NEW row for that symbol,
    dated after everything in `df`, should be demeaned against. Not leakage:
    it's computed once, after the training frame is finalised, and is only
    ever applied to rows dated strictly after the data used to compute it.

    Returns: {symbol: {column_name: mean_value}}
    """
    cols = _matching_columns(df.columns, bases)
    if not cols:
        return {}

    out: dict = {}
    sym_series = pd.Series(symbols).astype(str).values
    tmp = df[cols].copy()
    tmp["_symbol"] = sym_series
    means = tmp.groupby("_symbol")[cols].mean()
    for sym, row in means.iterrows():
        out[sym] = {c: float(v) for c, v in row.items() if pd.notna(v)}
    return out


def save_symbol_baselines(baselines: dict, path: str = DEFAULT_BASELINE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"bases": list(DEFAULT_DEMEAN_BASES), "baselines": baselines}, f, indent=2)
    logger.info(f"[symbol-demean] saved baselines for {len(baselines)} symbol(s) -> {p}")


def load_symbol_baselines(path: str = DEFAULT_BASELINE_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        logger.info(
            f"[symbol-demean] no baseline file at {p} — live rows for these bases will "
            f"stay RAW (undemeaned) until the next retrain writes one"
        )
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("baselines", {})


def demean_live_features(
    X: pd.DataFrame,
    symbols: pd.Series,
    baselines: dict,
    bases: tuple = DEFAULT_DEMEAN_BASES,
) -> pd.DataFrame:
    """
    Inference-time counterpart to demean_training_features(): subtracts each
    symbol's PERSISTED baseline (written by save_symbol_baselines() at the
    last retrain) from the matching raw columns of a live feature frame.

    Symbols with no stored baseline (new to the model since the last
    retrain, or a fresh deployment with no baseline file yet) are left RAW —
    there's no real history to demean against yet, and falling back to a
    population-wide average would just reintroduce a diluted version of the
    same fingerprint effect this whole change exists to remove.
    """
    cols = _matching_columns(X.columns, bases)
    if not cols or not baselines:
        return X

    X = X.copy()
    sym_series = pd.Series(symbols).astype(str)
    unknown_symbols = set()

    for i in X.index:
        sym = sym_series.loc[i] if i in sym_series.index else None
        sym_baseline = baselines.get(sym) if sym is not None else None
        if not sym_baseline:
            if sym is not None:
                unknown_symbols.add(sym)
            continue
        for col in cols:
            if col in sym_baseline and pd.notna(X.at[i, col]):
                X.at[i, col] = X.at[i, col] - sym_baseline[col]

    if unknown_symbols:
        logger.info(
            f"[symbol-demean] {len(unknown_symbols)} symbol(s) had no stored baseline "
            f"(kept RAW for these bases): {sorted(unknown_symbols)[:10]}"
        )
    return X
