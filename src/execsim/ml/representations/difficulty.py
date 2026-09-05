"""Train-only causal baseline-difficulty reweighting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_difficulty_components(
    actual_remaining: np.ndarray,
    baseline_remaining: np.ndarray,
    actual_shape: np.ndarray,
    baseline_shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute causal baseline log-level and conditional cumulative-shape error."""
    actual_total = np.asarray(actual_remaining, dtype=float)
    baseline_total = np.asarray(baseline_remaining, dtype=float)
    actual_curve = np.asarray(actual_shape, dtype=float)
    baseline_curve = np.asarray(baseline_shape, dtype=float)
    if actual_total.shape != baseline_total.shape or actual_curve.shape != baseline_curve.shape:
        raise ValueError("Difficulty targets and causal baseline predictions must align.")
    if actual_curve.ndim != 2 or len(actual_curve) != len(actual_total):
        raise ValueError("Difficulty shapes must contain one curve per remaining-volume target.")
    if not all(
        np.isfinite(values).all()
        for values in (actual_total, baseline_total, actual_curve, baseline_curve)
    ):
        raise ValueError("Difficulty inputs must be finite.")
    level = np.abs(np.log1p(actual_total) - np.log1p(np.maximum(baseline_total, 0.0)))
    shape = np.mean(
        np.abs(np.cumsum(actual_curve, axis=1) - np.cumsum(baseline_curve, axis=1)), axis=1
    )
    return level, shape


def difficulty_weights(
    level_error: np.ndarray,
    shape_error: np.ndarray,
    as_of_strata: np.ndarray | None = None,
) -> np.ndarray:
    """Average separate empirical CDF ranks within each training as-of stratum."""
    level = np.asarray(level_error, dtype=float)
    shape = np.asarray(shape_error, dtype=float)
    if level.shape != shape.shape or level.ndim != 1 or not len(level):
        raise ValueError("Difficulty inputs must be matching non-empty vectors.")
    if not np.isfinite(level).all() or not np.isfinite(shape).all():
        raise ValueError("Difficulty inputs must be finite.")
    strata = (
        np.zeros(len(level), dtype=np.int64) if as_of_strata is None else np.asarray(as_of_strata)
    )
    if strata.shape != level.shape:
        raise ValueError("Difficulty as-of strata must align with training errors.")
    level_rank = np.empty(len(level), dtype=float)
    shape_rank = np.empty(len(level), dtype=float)
    for stratum in np.unique(strata):
        selected = np.flatnonzero(strata == stratum)
        level_rank[selected] = _empirical_cdf(level[selected])
        shape_rank[selected] = _empirical_cdf(shape[selected])
    rank = 0.5 * (level_rank + shape_rank)
    raw = 0.5 + rank
    return raw / raw.mean()


def difficulty_table(
    *,
    level_error: np.ndarray,
    shape_error: np.ndarray,
    as_of_strata: np.ndarray,
    baseline_forecast_id: str,
    training_cutoff: str,
) -> pd.DataFrame:
    """Persist raw errors, separate CDF ranks, strata, weights, and causal identity."""
    level = np.asarray(level_error, dtype=float)
    shape = np.asarray(shape_error, dtype=float)
    strata = np.asarray(as_of_strata)
    weights = difficulty_weights(level, shape, strata)
    level_rank = np.empty(len(level), dtype=float)
    shape_rank = np.empty(len(level), dtype=float)
    for stratum in np.unique(strata):
        selected = np.flatnonzero(strata == stratum)
        level_rank[selected] = _empirical_cdf(level[selected])
        shape_rank[selected] = _empirical_cdf(shape[selected])
    return pd.DataFrame(
        {
            "level_error": level,
            "shape_error": shape,
            "level_cdf_rank": level_rank,
            "shape_cdf_rank": shape_rank,
            "as_of_stratum": strata,
            "weight": weights,
            "baseline_forecast_id": baseline_forecast_id,
            "training_cutoff": training_cutoff,
        }
    )


def _empirical_cdf(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = (np.arange(len(values)) + 0.5) / len(values)
    return ranks
