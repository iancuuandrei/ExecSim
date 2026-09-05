"""Point-in-time and shape validation for sequence artifacts."""

from __future__ import annotations

from execsim.ml.sequences.schemas import SequenceRecord, SequenceSample


def validate_sequence_sample(record: SequenceRecord, sample: SequenceSample) -> tuple[str, ...]:
    """Return violations of session identity, mask, and information time."""
    errors: list[str] = []
    if sample.session_id != record.session_id:
        errors.append("sample references a different session")
    if sample.as_of_token > int(record.token_mask.sum()):
        errors.append("sample as_of exceeds valid session tokens")
    if sample.as_of_token and record.available_at_ns[sample.as_of_token - 1] > sample.as_of_ns:
        errors.append("context contains information after as_of")
    for index, available in zip(sample.target_indices, sample.target_mask, strict=True):
        if available and record.available_at_ns[index] <= sample.as_of_ns:
            errors.append("future target does not follow as_of")
    return tuple(errors)
