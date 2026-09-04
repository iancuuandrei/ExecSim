from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def stable_softmax(scores: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Softmax scores must be a non-empty finite vector.")
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def positive_volume_forecast(values: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    volumes = np.asarray(values, dtype=float)
    if volumes.ndim != 1 or not volumes.size or not np.isfinite(volumes).all():
        raise ValueError("Volume predictions must be a non-empty finite vector.")
    positive = np.maximum(volumes, 0.0)
    total = positive.sum()
    shares = positive / total if total else np.zeros_like(positive)
    return positive, shares
