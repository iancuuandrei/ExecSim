from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ForecastModel(Protocol):
    @property
    def family(self) -> str: ...

    def fit(self, features: NDArray[np.float64], target: NDArray[np.float64]) -> None: ...

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def parameters(self) -> dict[str, object]: ...
