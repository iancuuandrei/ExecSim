from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

FeatureMode = Literal["static", "dynamic", "both"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    dtype: str
    description: str
    source_fields: tuple[str, ...]
    lookback: str
    transformation: str
    earliest_availability: str
    mode: FeatureMode
    missing_value_rule: str
    version: str
    leakage_notes: str
    rationale: str

    def __post_init__(self) -> None:
        values = (
            self.name,
            self.dtype,
            self.description,
            self.lookback,
            self.transformation,
            self.earliest_availability,
            self.missing_value_rule,
            self.version,
            self.leakage_notes,
            self.rationale,
        )
        if any(not value.strip() for value in values) or not self.source_fields:
            raise ValueError("Feature metadata fields must be non-empty.")


@dataclass(frozen=True, slots=True)
class FeatureValue:
    name: str
    value: object
    available_at: pd.Timestamp

    def __post_init__(self) -> None:
        if self.available_at.tzinfo is None:
            raise ValueError("Feature availability timestamps must be timezone-aware.")
