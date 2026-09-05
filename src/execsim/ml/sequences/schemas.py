"""Fixed-shape sequence and sample-index schemas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import numpy as np

PAPER_FEATURES = (
    "close_to_close_log_return",
    "open_to_close_log_return",
    "log_high_low_range",
    "realized_volatility",
    "share_volume_surprise",
    "dollar_volume_surprise",
    "trade_count_surprise",
    "cumulative_volume_ratio",
    "adv_bucket_ratio",
    "volume_surprise_change",
    "vwap_to_previous_close",
    "log_adv20",
    "time_sine",
    "time_cosine",
    "elapsed_fraction",
    "spy_return",
    "spy_volume_surprise",
    "spy_realized_volatility",
)
TOKEN_COUNT = 26
FEATURE_COUNT = 18
ENCODER_FEATURE_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 15, 16, 17)
CONDITIONING_FEATURE_INDICES = (7, 11, 12, 13, 14)
ENCODER_FEATURE_COUNT = len(ENCODER_FEATURE_INDICES)
CONDITIONING_FEATURE_COUNT = len(CONDITIONING_FEATURE_INDICES)
CONTEXT_LENGTH = 8
MIN_CONTEXT = 4
HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """Store one session tensor with targets and point-in-time provenance."""

    schema_version: ClassVar[str] = "paper-sequence-v2"
    session_id: str
    instrument_id: str
    symbol: str
    session_date: str
    features: np.ndarray
    token_mask: np.ndarray
    available_at_ns: np.ndarray
    raw_volume: np.ndarray
    raw_vwap: np.ndarray
    causal_baseline_volume: np.ndarray
    source_sha256: str
    cutoff: str
    training_cutoff: str
    market_information_as_of: str
    feature_history_end: str

    def __post_init__(self) -> None:
        if self.features.shape != (TOKEN_COUNT, FEATURE_COUNT):
            raise ValueError(f"features must have shape {(TOKEN_COUNT, FEATURE_COUNT)}")
        for name, value in (
            ("token_mask", self.token_mask),
            ("available_at_ns", self.available_at_ns),
            ("raw_volume", self.raw_volume),
            ("raw_vwap", self.raw_vwap),
            ("causal_baseline_volume", self.causal_baseline_volume),
        ):
            if value.shape != (TOKEN_COUNT,):
                raise ValueError(f"{name} must have shape {(TOKEN_COUNT,)}")
        valid = self.token_mask.astype(bool)
        if not np.isfinite(self.features[valid]).all():
            raise ValueError("Valid token features must be finite.")
        if (self.raw_volume[valid] < 0).any():
            raise ValueError("Raw target volume must be non-negative.")
        if (
            not np.isfinite(self.causal_baseline_volume[valid]).all()
            or (self.causal_baseline_volume[valid] < 0).any()
        ):
            raise ValueError("Causal baseline volume must be finite and non-negative.")
        if np.any(np.diff(self.available_at_ns[valid]) <= 0):
            raise ValueError("Valid token availability timestamps must increase strictly.")
        if np.any((~self.token_mask[:-1].astype(bool)) & self.token_mask[1:].astype(bool)):
            raise ValueError("Valid session tokens must form one contiguous prefix.")
        if date.fromisoformat(self.cutoff) >= date.fromisoformat(self.session_date):
            raise ValueError("Sequence cutoff must precede its target session.")
        if self.cutoff != self.feature_history_end:
            raise ValueError("Legacy sequence cutoff must equal feature_history_end.")
        import pandas as pd

        market_time = pd.Timestamp(self.market_information_as_of)
        if market_time.tzinfo is None or market_time.date().isoformat() != self.session_date:
            raise ValueError("market_information_as_of must be aware and within the session.")
        if len(self.source_sha256) != 64:
            raise ValueError("Sequence source_sha256 must contain a full SHA-256 digest.")


@dataclass(frozen=True, slots=True)
class SequenceSample:
    """Reference one causal context and its available forward targets."""

    sample_id: str
    session_id: str
    fold_id: str
    partition: str
    as_of_token: int
    context_start: int
    context_end: int
    target_indices: tuple[int, ...]
    target_mask: tuple[bool, ...]
    as_of_ns: int
    source_sequence_hash: str
    cutoff: str
    training_cutoff: str
    market_information_as_of: str
    feature_history_end: str

    def __post_init__(self) -> None:
        if self.context_end != self.as_of_token or self.context_start < 0:
            raise ValueError("Context must end at the exclusive as-of token.")
        if self.context_end - self.context_start > CONTEXT_LENGTH:
            raise ValueError("Context exceeds the locked eight-token window.")
        if len(self.target_indices) != len(HORIZONS) or len(self.target_mask) != len(HORIZONS):
            raise ValueError("Sample targets must match the four locked horizons.")
        if self.cutoff != self.feature_history_end:
            raise ValueError("Sample cutoff must identify the feature-history end.")
