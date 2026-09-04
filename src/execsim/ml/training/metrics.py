from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def regression_metrics(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    truth, estimate = _matching_arrays(actual, predicted)
    error = estimate - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
    }


def shape_metrics(
    actual: ArrayLike, predicted: ArrayLike, *, epsilon: float = 1e-12
) -> dict[str, float]:
    truth, estimate = _matching_arrays(actual, predicted)
    if np.any(truth < 0) or np.any(estimate < 0):
        raise ValueError("Shape values must be non-negative.")
    truth = truth / max(float(truth.sum()), epsilon)
    estimate = estimate / max(float(estimate.sum()), epsilon)
    error = estimate - truth
    return {
        "share_mae": float(np.mean(np.abs(error))),
        "share_rmse": float(math.sqrt(np.mean(error**2))),
        "cross_entropy": float(-np.sum(truth * np.log(estimate + epsilon))),
        "kl_divergence": float(np.sum(truth * np.log((truth + epsilon) / (estimate + epsilon)))),
        "cumulative_curve_error": float(np.sum(np.abs(np.cumsum(truth) - np.cumsum(estimate)))),
        "wasserstein_bucket_distance": float(
            np.sum(np.abs(np.cumsum(truth) - np.cumsum(estimate)))
        ),
    }


def _matching_arrays(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(actual, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 1 or not truth.size:
        raise ValueError("Metric inputs must be equal-length non-empty vectors.")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("Metric inputs must be finite.")
    return truth, estimate
