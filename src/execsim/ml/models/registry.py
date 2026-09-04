from __future__ import annotations

from typing import cast

from execsim.ml.models.baselines import MeanTargetBaseline
from execsim.ml.models.protocol import ForecastModel
from execsim.ml.models.sklearn_adapters import ModelFamily, SklearnRegressorAdapter


def create_model(
    family: str, *, parameters: dict[str, object] | None = None, seed: int = 17
) -> ForecastModel:
    if family == "mean-target":
        return MeanTargetBaseline()
    if family in {"ridge", "elastic_net", "hist_gradient_boosting"}:
        return SklearnRegressorAdapter(
            family=cast(ModelFamily, family),
            model_parameters=parameters or {},
            random_seed=seed,
        )
    raise ValueError(f"Unknown model family: {family}")
