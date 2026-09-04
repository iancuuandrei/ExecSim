from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from execsim.ml.features.registry import FeatureRegistry
from execsim.ml.schemas import FeatureValue


def validate_feature_values(
    values: Iterable[FeatureValue], *, as_of: pd.Timestamp, registry: FeatureRegistry
) -> None:
    if as_of.tzinfo is None:
        raise ValueError("Sample as_of must be timezone-aware.")
    seen: set[str] = set()
    for feature in values:
        registry.get(feature.name)
        if feature.name in seen:
            raise ValueError(f"Duplicate feature value: {feature.name}")
        if feature.available_at > as_of:
            raise ValueError(
                f"Feature {feature.name} is available at {feature.available_at}, "
                f"after as_of {as_of}."
            )
        seen.add(feature.name)
