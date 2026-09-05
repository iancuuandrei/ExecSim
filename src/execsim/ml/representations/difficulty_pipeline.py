"""Causal TRAIN-only difficulty ledger for sparse representation adaptation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import file_sha256, read_json
from execsim.ml.representations.difficulty import difficulty_table
from execsim.ml.sequences.manifests import read_sequence_record
from execsim.ml.sequences.streaming import _sample_from_row


def build_difficulty_ledger(sequence_manifest_path: Path, output: Path) -> pd.DataFrame:
    """Rank separate historical-profile level/shape errors within TRAIN as-of strata."""
    manifest = read_json(sequence_manifest_path)
    root = sequence_manifest_path.parent
    records = {
        path.stem: read_sequence_record(path)
        for value in manifest["sequence_files"]
        if "sessions/train/" in str(value).replace("\\", "/")
        for path in (root / str(value),)
    }
    samples = [
        _sample_from_row(row)
        for value in manifest["index_files"]
        if "indexes/train/" in str(value).replace("\\", "/")
        for row in pd.read_parquet(root / str(value)).itertuples(index=False)
    ]
    history: dict[tuple[str, int], list[np.ndarray]] = {}
    rows = []
    for sample in sorted(
        samples,
        key=lambda item: (records[item.session_id].session_date, item.session_id, item.as_of_token),
    ):
        record = records[sample.session_id]
        future = record.raw_volume[sample.as_of_token :]
        key = (record.instrument_id, sample.as_of_token)
        prior = history.setdefault(key, [])
        width = len(future)
        if prior:
            comparable = np.stack([values[:width] for values in prior if len(values) >= width])
            baseline = np.mean(comparable, axis=0)
        else:
            observed_mean = float(np.mean(record.raw_volume[: sample.as_of_token]))
            baseline = np.full(width, observed_mean)
        actual_total = float(future.sum())
        baseline_total = float(baseline.sum())
        actual_shape = future / actual_total if actual_total > 0 else np.zeros_like(future)
        baseline_shape = (
            baseline / baseline_total if baseline_total > 0 else np.zeros_like(baseline)
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "fold_id": sample.fold_id,
                "instrument_id": record.instrument_id,
                "session_date": record.session_date,
                "as_of_stratum": sample.as_of_token,
                "level_error": abs(np.log1p(baseline_total) - np.log1p(actual_total)),
                "shape_error": float(
                    np.mean(np.abs(np.cumsum(baseline_shape) - np.cumsum(actual_shape)))
                ),
            }
        )
        prior.append(future.copy())
    if not rows:
        raise ValueError("Difficulty ledger has no causally preceded TRAIN samples.")
    raw = pd.DataFrame(rows)
    ranked = difficulty_table(
        level_error=raw["level_error"].to_numpy(),
        shape_error=raw["shape_error"].to_numpy(),
        as_of_strata=raw["as_of_stratum"].to_numpy(),
        baseline_forecast_id="causal-historical-volume-profile-v1",
        training_cutoff=str(manifest["cutoff"]),
    )
    ledger = pd.concat((raw.reset_index(drop=True), ranked.drop(columns="as_of_stratum")), axis=1)
    ledger["paper_config_hash"] = str(manifest["config_hash"])
    ledger["sequence_manifest_hash"] = file_sha256(sequence_manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(output, index=False)
    return ledger
