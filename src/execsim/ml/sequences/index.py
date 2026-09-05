"""Deterministic causal sample indexes over session tensors."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from execsim.data.paper.partitions import resolve_fold_partition
from execsim.ml.sequences.schemas import (
    CONTEXT_LENGTH,
    HORIZONS,
    MIN_CONTEXT,
    SequenceRecord,
    SequenceSample,
)


def build_sample_index(
    record: SequenceRecord,
    *,
    fold_id: str,
    partition: str | None = None,
    source_sequence_hash: str,
) -> tuple[SequenceSample, ...]:
    """Enumerate the complete valid as-of grid starting at 10:30."""
    session_date = pd.Timestamp(record.session_date).date()
    resolved_partition = resolve_fold_partition(fold_id, session_date)
    if partition is not None and partition != resolved_partition:
        raise ValueError(
            f"Fold partition mismatch: expected {resolved_partition}, got {partition}."
        )
    samples = []
    valid_count = int(record.token_mask.sum())
    for as_of_token in range(MIN_CONTEXT, valid_count):
        indices = tuple(as_of_token + horizon - 1 for horizon in HORIZONS)
        target_mask = tuple(index < valid_count for index in indices)
        if not any(target_mask):
            continue
        payload = (
            f"{record.session_id}|{fold_id}|{resolved_partition}|{as_of_token}|{record.cutoff}"
        )
        samples.append(
            SequenceSample(
                sample_id=hashlib.sha256(payload.encode()).hexdigest()[:20],
                session_id=record.session_id,
                fold_id=fold_id,
                partition=resolved_partition,
                as_of_token=as_of_token,
                context_start=max(0, as_of_token - CONTEXT_LENGTH),
                context_end=as_of_token,
                target_indices=indices,
                target_mask=target_mask,
                as_of_ns=int(record.available_at_ns[as_of_token - 1]),
                source_sequence_hash=source_sequence_hash,
                cutoff=record.cutoff,
                training_cutoff=record.training_cutoff,
                market_information_as_of=str(
                    pd.Timestamp(record.available_at_ns[as_of_token - 1], tz="UTC")
                ),
                feature_history_end=record.feature_history_end,
            )
        )
    return tuple(samples)


def sample_training_positions(
    samples: tuple[SequenceSample, ...], *, epoch: int, seed: int
) -> tuple[SequenceSample, ...]:
    """Select exactly two deterministic as-of positions for one session and epoch."""
    if len(samples) < 2:
        raise ValueError("Training sessions require at least two valid as-of positions.")
    digest = hashlib.sha256(f"{samples[0].session_id}|{epoch}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    selected = np.sort(rng.choice(len(samples), size=2, replace=False))
    return tuple(samples[int(index)] for index in selected)


def write_sample_index(samples: tuple[SequenceSample, ...], path: Path) -> Path:
    """Persist the deterministic window index separately from session tensors."""
    if not samples:
        raise ValueError("Cannot persist an empty sequence sample index.")
    rows = [asdict(sample) for sample in sorted(samples, key=lambda item: item.sample_id)]
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
