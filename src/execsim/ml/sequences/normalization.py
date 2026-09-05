"""Training-fold-only robust feature normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RobustFoldNormalizer:
    """Clip to train quantiles and scale by train median and interquartile range."""

    lower: np.ndarray
    upper: np.ndarray
    median: np.ndarray
    iqr: np.ndarray
    epsilon: float = 1e-6

    @classmethod
    def fit(
        cls, values: np.ndarray, mask: np.ndarray, *, epsilon: float = 1e-6
    ) -> RobustFoldNormalizer:
        """Fit parameters from a declared training tensor only."""
        array = np.asarray(values, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if array.ndim != 3 or array.shape[:2] != valid.shape:
            raise ValueError("Normalizer expects [session, token, feature] values and token mask.")
        rows = array[valid]
        if not rows.size or not np.isfinite(rows).all():
            raise ValueError("Training normalization rows must be non-empty and finite.")
        lower, q25, median, q75, upper = np.quantile(rows, [0.005, 0.25, 0.5, 0.75, 0.995], axis=0)
        return cls(lower, upper, median, np.maximum(q75 - q25, epsilon), epsilon)

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply persisted train parameters and leave padding exactly zero."""
        array = np.asarray(values, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if array.shape[:-1] != valid.shape or array.shape[-1] != len(self.median):
            raise ValueError("Normalization input shape does not match fitted parameters.")
        result = np.zeros_like(array, dtype=np.float32)
        result[valid] = (np.clip(array[valid], self.lower, self.upper) - self.median) / np.maximum(
            self.iqr, self.epsilon
        )
        if not np.isfinite(result).all():
            raise ValueError("Normalization produced non-finite values.")
        return result

    def stable_payload(self) -> dict[str, object]:
        """Return JSON-compatible persisted parameters."""
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
            "epsilon": self.epsilon,
        }
