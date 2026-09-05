"""Window extraction without overlapping sequence materialization."""

from __future__ import annotations

import numpy as np

from execsim.ml.sequences.schemas import (
    CONTEXT_LENGTH,
    FEATURE_COUNT,
    HORIZONS,
    SequenceRecord,
    SequenceSample,
)


def extract_window(record: SequenceRecord, sample: SequenceSample) -> dict[str, np.ndarray]:
    """Extract left-padded context and masked actual future tokens."""
    context = np.zeros((CONTEXT_LENGTH, FEATURE_COUNT), dtype=np.float32)
    context_mask = np.zeros(CONTEXT_LENGTH, dtype=bool)
    observed = record.features[sample.context_start : sample.context_end]
    context[-len(observed) :] = observed
    context_mask[-len(observed) :] = True
    targets = np.zeros((len(HORIZONS), FEATURE_COUNT), dtype=np.float32)
    target_mask = np.asarray(sample.target_mask, dtype=bool)
    for offset, (index, available) in enumerate(
        zip(sample.target_indices, target_mask, strict=True)
    ):
        if available:
            if record.available_at_ns[index] <= sample.as_of_ns:
                raise ValueError("Future target availability must follow sample as_of.")
            targets[offset] = record.features[index]
    return {
        "context": context,
        "context_mask": context_mask,
        "targets": targets,
        "target_mask": target_mask,
    }
