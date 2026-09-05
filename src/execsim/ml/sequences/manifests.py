"""Sequence artifact identities and fixed-list Arrow schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from execsim.data.paper.manifests import stable_hash, write_json_atomic
from execsim.ml.sequences.normalization import RobustFoldNormalizer
from execsim.ml.sequences.schemas import FEATURE_COUNT, TOKEN_COUNT, SequenceRecord


@dataclass(frozen=True, slots=True)
class SequenceManifest:
    """Identify a fold-normalized sequence corpus and its causal cutoff."""

    schema_version: str
    feature_schema_version: str
    fold_id: str
    cutoff: str
    raw_hashes: tuple[str, ...]
    sequence_files: tuple[str, ...]
    normalization: dict[str, object]
    manifest_id: str
    universe_manifest_hash: str = ""
    corporate_action_manifest_hash: str = ""
    config_hash: str = ""
    index_files: tuple[str, ...] = ()
    partition_counts: dict[str, int] | None = None
    exclusions: tuple[dict[str, str], ...] = ()
    quality_protocol: str = "exact-minute-v1"


def write_sequence_manifest(
    path: Path,
    *,
    fold_id: str,
    cutoff: str,
    raw_hashes: tuple[str, ...],
    sequence_files: tuple[str, ...],
    normalizer: RobustFoldNormalizer,
    universe_manifest_hash: str = "",
    corporate_action_manifest_hash: str = "",
    config_hash: str = "",
    index_files: tuple[str, ...] = (),
    partition_counts: dict[str, int] | None = None,
    exclusions: tuple[dict[str, str], ...] = (),
    quality_protocol: str = "exact-minute-v1",
) -> SequenceManifest:
    """Persist source identities and train-only scaler values for one fold."""
    stable = {
        "schema_version": "paper-sequence-manifest-v2",
        "feature_schema_version": "paper-token-v2",
        "fold_id": fold_id,
        "cutoff": cutoff,
        "raw_hashes": tuple(sorted(raw_hashes)),
        "sequence_files": tuple(sorted(sequence_files)),
        "normalization": normalizer.stable_payload(),
        "universe_manifest_hash": universe_manifest_hash,
        "corporate_action_manifest_hash": corporate_action_manifest_hash,
        "config_hash": config_hash,
        "index_files": tuple(sorted(index_files)),
        "partition_counts": partition_counts or {},
        "exclusions": exclusions,
        "quality_protocol": quality_protocol,
    }
    manifest = SequenceManifest(
        schema_version="paper-sequence-manifest-v2",
        feature_schema_version="paper-token-v2",
        fold_id=fold_id,
        cutoff=cutoff,
        raw_hashes=tuple(sorted(raw_hashes)),
        sequence_files=tuple(sorted(sequence_files)),
        normalization=normalizer.stable_payload(),
        manifest_id="sequence-manifest-" + stable_hash(stable)[:16],
        universe_manifest_hash=universe_manifest_hash,
        corporate_action_manifest_hash=corporate_action_manifest_hash,
        config_hash=config_hash,
        index_files=tuple(sorted(index_files)),
        partition_counts=partition_counts or {},
        exclusions=exclusions,
        quality_protocol=quality_protocol,
    )
    write_json_atomic(path, asdict(manifest))
    return manifest


def sequence_arrow_schema() -> pa.Schema:
    """Return the one-row-per-session fixed-shape Arrow contract."""
    return pa.schema(
        [
            ("session_id", pa.string()),
            ("instrument_id", pa.string()),
            ("symbol", pa.string()),
            ("session_date", pa.date32()),
            ("features", pa.list_(pa.float32(), TOKEN_COUNT * FEATURE_COUNT)),
            ("token_mask", pa.list_(pa.bool_(), TOKEN_COUNT)),
            ("available_at_ns", pa.list_(pa.int64(), TOKEN_COUNT)),
            ("raw_volume", pa.list_(pa.float64(), TOKEN_COUNT)),
            ("raw_vwap", pa.list_(pa.float64(), TOKEN_COUNT)),
            ("causal_baseline_volume", pa.list_(pa.float64(), TOKEN_COUNT)),
            ("observed_bar_count", pa.list_(pa.int16(), TOKEN_COUNT)),
            ("provider_gap_count", pa.int16()),
            ("quality_protocol", pa.string()),
            ("source_sha256", pa.string()),
            ("cutoff", pa.string()),
            ("training_cutoff", pa.string()),
            ("market_information_as_of", pa.string()),
            ("feature_history_end", pa.string()),
        ]
    )


def sequence_cache_identity(payload: dict[str, object]) -> str:
    """Hash all source, schema, fold, normalization, and cutoff inputs."""
    required = {"raw_hash", "feature_schema", "fold_id", "cutoff", "normalization"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Sequence cache identity missing fields: {sorted(missing)}")
    return "sequence-" + stable_hash(payload)[:16]


def write_sequence_record(record: SequenceRecord, root: Path) -> Path:
    """Write one fixed-shape session row to year and instrument partitions."""
    year = record.session_date[:4]
    destination = (
        root
        / f"year={year}"
        / f"instrument_id={record.instrument_id}"
        / f"{record.session_id}.parquet"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "session_id": record.session_id,
                "instrument_id": record.instrument_id,
                "symbol": record.symbol,
                "session_date": date.fromisoformat(record.session_date),
                "features": record.features.reshape(-1).tolist(),
                "token_mask": record.token_mask.tolist(),
                "available_at_ns": record.available_at_ns.tolist(),
                "raw_volume": record.raw_volume.tolist(),
                "raw_vwap": record.raw_vwap.tolist(),
                "causal_baseline_volume": record.causal_baseline_volume.tolist(),
                "observed_bar_count": record.observed_bar_count.tolist(),
                "provider_gap_count": record.provider_gap_count,
                "quality_protocol": record.quality_protocol,
                "source_sha256": record.source_sha256,
                "cutoff": record.cutoff,
                "training_cutoff": record.training_cutoff,
                "market_information_as_of": record.market_information_as_of,
                "feature_history_end": record.feature_history_end,
            }
        ],
        schema=sequence_arrow_schema(),
    )
    pq.write_table(table, destination, compression="zstd")
    return destination


def read_sequence_record(path: Path) -> SequenceRecord:
    """Read one fixed-shape session and re-run schema validation."""
    table = pq.read_table(path, schema=sequence_arrow_schema())
    if table.num_rows != 1:
        raise ValueError("A sequence partition file must contain exactly one session row.")
    row = table.to_pylist()[0]
    return SequenceRecord(
        session_id=row["session_id"],
        instrument_id=row["instrument_id"],
        symbol=row["symbol"],
        session_date=str(row["session_date"]),
        features=np.asarray(row["features"], dtype=np.float32).reshape(TOKEN_COUNT, FEATURE_COUNT),
        token_mask=np.asarray(row["token_mask"], dtype=bool),
        available_at_ns=np.asarray(row["available_at_ns"], dtype=np.int64),
        raw_volume=np.asarray(row["raw_volume"], dtype=float),
        raw_vwap=np.asarray(row["raw_vwap"], dtype=float),
        causal_baseline_volume=np.asarray(row["causal_baseline_volume"], dtype=float),
        observed_bar_count=np.asarray(row["observed_bar_count"], dtype=np.int16),
        provider_gap_count=int(row["provider_gap_count"]),
        quality_protocol=row["quality_protocol"],
        source_sha256=row["source_sha256"],
        cutoff=row["cutoff"],
        training_cutoff=row["training_cutoff"],
        market_information_as_of=row["market_information_as_of"],
        feature_history_end=row["feature_history_end"],
    )
