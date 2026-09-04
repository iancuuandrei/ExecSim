from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

ModelFamily = Literal["ridge", "elastic_net", "hist_gradient_boosting"]


@dataclass(slots=True)
class SklearnRegressorAdapter:
    family: ModelFamily
    model_parameters: dict[str, object] = field(default_factory=dict)
    random_seed: int = 17
    pipeline: object | None = field(default=None, init=False, repr=False)

    def fit(self, features: NDArray[np.float64], target: NDArray[np.float64]) -> None:
        if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
            raise ValueError("Training features must be 2D and align with the 1D target.")
        if not len(target) or not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("Training data must be non-empty and finite.")
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import ElasticNet, Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the 'ml' extra to fit model adapters.") from exc
        if self.family == "ridge":
            model = Ridge(**self.model_parameters)
        elif self.family == "elastic_net":
            model = ElasticNet(random_state=self.random_seed, **self.model_parameters)
        elif self.family == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(
                random_state=self.random_seed, **self.model_parameters
            )
        else:
            raise ValueError(f"Unknown scikit-learn model family: {self.family}")
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        self.pipeline.fit(features, target)  # type: ignore[attr-defined]

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before prediction.")
        raw_predictions = cast(Any, self.pipeline).predict(features)
        predictions = np.asarray(raw_predictions, dtype=np.float64)
        if not np.isfinite(predictions).all():
            raise RuntimeError("Model produced non-finite predictions.")
        return predictions

    def parameters(self) -> dict[str, object]:
        return {
            "family": self.family,
            "model_parameters": self.model_parameters,
            "random_seed": self.random_seed,
        }

    @property
    def fitted_preprocessor_mean(self) -> NDArray[np.float64]:
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before inspecting preprocessing.")
        scaler = self.pipeline.named_steps["scaler"]  # type: ignore[attr-defined]
        return np.asarray(scaler.mean_, dtype=float)
