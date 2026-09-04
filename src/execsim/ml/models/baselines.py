from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class MeanTargetBaseline:
    """Training-fold mean used as a deterministic statistical lower baseline."""

    family: str = "mean-target"
    _mean: float | None = field(default=None, init=False)

    def fit(self, features: NDArray[np.float64], target: NDArray[np.float64]) -> None:
        del features
        if target.ndim != 1 or not len(target) or not np.isfinite(target).all():
            raise ValueError("Training target must be a non-empty finite vector.")
        self._mean = float(target.mean())

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._mean is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return np.full(len(features), self._mean, dtype=float)

    def parameters(self) -> dict[str, object]:
        return {"mean": self._mean}
