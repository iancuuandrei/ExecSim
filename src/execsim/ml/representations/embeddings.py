"""Causal frozen-embedding export and cache identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from execsim.data.paper.manifests import stable_hash
from execsim.ml.representations.schemas import EmbeddingCacheKey
from execsim.ml.sequences.schemas import (
    CONDITIONING_FEATURE_INDICES,
    ENCODER_FEATURE_INDICES,
)


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactManifest:
    """Describe a checksummed partition of frozen causal embeddings."""

    artifact_id: str
    checkpoint_hash: str
    partition_identity: str
    row_count: int
    shape: tuple[int, int]
    components: tuple[str, ...]
    training_cutoff: str
    source_hashes: tuple[str, ...]
    parquet_sha256: str
    sequence_manifest_hash: str = ""
    normalization_hash: str = ""
    paper_config_hash: str = ""
    checkpoint_manifest_hash: str = ""
    torch_version: str = ""

    def __post_init__(self) -> None:
        hashes = (self.checkpoint_hash, self.parquet_sha256, *self.source_hashes)
        if any(len(value) != 64 for value in hashes):
            raise ValueError("Embedding artifact hashes must be full SHA-256 digests.")
        if self.row_count <= 0 or self.shape != (self.row_count, 644):
            raise ValueError("Embedding artifact shape must be [row_count, 644].")
        if self.components != (
            "current",
            "h1",
            "h2",
            "h4",
            "h8",
            "horizon_availability",
        ):
            raise ValueError("Embedding artifact components are incompatible.")
        for value in (
            self.sequence_manifest_hash,
            self.normalization_hash,
            self.paper_config_hash,
            self.checkpoint_manifest_hash,
        ):
            if value and len(value) != 64:
                raise ValueError("Embedding compatibility hashes must be full SHA-256 digests.")


def embedding_cache_identity(key: EmbeddingCacheKey) -> str:
    """Hash the complete embedding derivation identity."""
    return "embedding-" + stable_hash(asdict(key))[:20]


def compose_causal_embedding(
    current: np.ndarray, predicted: np.ndarray, availability: np.ndarray | None = None
) -> np.ndarray:
    """Concatenate causal latents and an explicit four-horizon availability mask."""
    current_values = np.asarray(current, dtype=np.float32)
    predicted_values = np.asarray(predicted, dtype=np.float32)
    if current_values.shape != (128,) or predicted_values.shape != (4, 128):
        raise ValueError("Embedding export requires one current and four predicted 128-D latents.")
    available = (
        np.ones(4, dtype=np.float32)
        if availability is None
        else np.asarray(availability, dtype=np.float32)
    )
    if available.shape != (4,) or not np.isin(available, (0.0, 1.0)).all():
        raise ValueError("Embedding availability must be a binary four-horizon vector.")
    predicted_values = predicted_values * available[:, None]
    output = np.concatenate((current_values, predicted_values.reshape(-1), available))
    if output.shape != (644,) or not np.isfinite(output).all():
        raise ValueError("Embedding export produced an invalid vector.")
    return output


def export_frozen_embedding(
    model: Any, context: Any, context_mask: Any, horizon_mask: Any | None = None
) -> np.ndarray:
    """Export current plus predicted latents without accepting any actual future token."""
    import torch

    if context.shape != (1, 8, 18) or context_mask.shape != (1, 8):
        raise ValueError("Frozen export requires one [8, 18] context and mask.")
    if int(context_mask.sum()) < 1:
        raise ValueError("Frozen export requires an observed context token.")
    model.eval()
    with torch.no_grad():
        dynamic = context[..., ENCODER_FEATURE_INDICES]
        _, linked = model.encode(dynamic)
        linked = linked * context_mask.unsqueeze(-1).to(linked.dtype)
        current_index = int(torch.nonzero(context_mask[0], as_tuple=False)[-1].item())
        current = linked[0, current_index]
        predictions = []
        for horizon in range(4):
            index = torch.tensor([horizon], dtype=torch.long, device=context.device)
            last = context[0, current_index][..., CONDITIONING_FEATURE_INDICES].unsqueeze(0)
            predictions.append(model.link(model.predictor(linked, context_mask, last, index))[0])
        predicted = torch.stack(predictions)
    availability = None if horizon_mask is None else horizon_mask[0].cpu().numpy()
    return compose_causal_embedding(current.cpu().numpy(), predicted.cpu().numpy(), availability)


def export_frozen_embedding_batch(
    model: Any, context: Any, context_mask: Any, horizon_mask: Any | None = None
) -> np.ndarray:
    """Export a device-resident batch without accepting future target tensors."""
    import torch

    if context.ndim != 3 or context.shape[1:] != (8, 18):
        raise ValueError("Frozen batch export requires [batch, 8, 18] context.")
    if context_mask.shape != context.shape[:2] or not bool(context_mask.any(dim=1).all()):
        raise ValueError("Frozen batch export requires one observed token in every row.")
    model.eval()
    with torch.no_grad():
        _, linked = model.encode(context[..., ENCODER_FEATURE_INDICES])
        linked = linked * context_mask.unsqueeze(-1).to(linked.dtype)
        positions = torch.arange(context_mask.shape[1], device=context_mask.device)
        last = torch.where(context_mask, positions[None, :], -1).max(dim=1).values
        current = linked[torch.arange(len(linked), device=linked.device), last]
        conditioning = context[torch.arange(len(context), device=context.device), last][
            ..., CONDITIONING_FEATURE_INDICES
        ]
        predicted = []
        for horizon in range(4):
            index = torch.full((len(linked),), horizon, dtype=torch.long, device=linked.device)
            predicted.append(model.link(model.predictor(linked, context_mask, conditioning, index)))
        available = (
            torch.ones((len(context), 4), dtype=current.dtype, device=current.device)
            if horizon_mask is None
            else horizon_mask.to(device=current.device, dtype=current.dtype)
        )
        predicted = [
            value * available[:, index : index + 1] for index, value in enumerate(predicted)
        ]
        output = torch.cat((current, *predicted, available), dim=1)
    if output.shape != (len(context), 644) or not bool(torch.isfinite(output).all()):
        raise ValueError("Frozen embedding batch is non-finite or has the wrong shape.")
    return output.float().cpu().numpy()


def validate_embedding_metadata(metadata: dict[str, object]) -> None:
    """Reject exports that omit causal or artifact provenance."""
    required = {
        "instrument_id",
        "symbol",
        "session_date",
        "as_of",
        "fold_id",
        "seed",
        "geometry",
        "adaptation",
        "predictor_family",
        "checkpoint_hash",
        "sequence_hash",
        "cutoff",
        "component_order",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Embedding metadata missing fields: {sorted(missing)}")
    if date.fromisoformat(str(metadata["cutoff"])) >= date.fromisoformat(
        str(metadata["session_date"])
    ):
        raise ValueError("Embedding cutoff must precede its target session.")
    expected_order = "current,h1,h2,h4,h8,horizon_availability"
    if str(metadata["component_order"]) != expected_order:
        raise ValueError(f"Embedding component order must be {expected_order}.")


def write_embedding_parquet(
    path: Path, *, embedding: np.ndarray, metadata: dict[str, object]
) -> Path:
    """Write one causal fixed-size 644-value embedding with complete provenance."""
    validate_embedding_metadata(metadata)
    values = np.asarray(embedding, dtype=np.float32)
    if values.shape != (644,) or not np.isfinite(values).all():
        raise ValueError("Embedding artifact requires one finite 644-value vector.")
    schema = pa.schema(
        [
            ("instrument_id", pa.string()),
            ("symbol", pa.string()),
            ("session_date", pa.string()),
            ("as_of", pa.string()),
            ("fold_id", pa.string()),
            ("seed", pa.int64()),
            ("geometry", pa.string()),
            ("adaptation", pa.string()),
            ("predictor_family", pa.string()),
            ("checkpoint_hash", pa.string()),
            ("sequence_hash", pa.string()),
            ("cutoff", pa.string()),
            ("component_order", pa.string()),
            ("embedding", pa.list_(pa.float32(), 644)),
        ]
    )
    payload = {name: metadata[name] for name in metadata}
    payload["seed"] = int(str(payload["seed"]))
    payload["component_order"] = str(payload["component_order"])
    payload["embedding"] = values.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([payload], schema=schema), path, compression="zstd")
    return path


def write_embedding_artifact_manifest(
    path: Path,
    *,
    artifact_id: str,
    checkpoint_hash: str,
    partition_identity: str,
    row_count: int,
    training_cutoff: str,
    source_hashes: tuple[str, ...],
    parquet_path: Path,
    sequence_manifest_hash: str = "",
    normalization_hash: str = "",
    paper_config_hash: str = "",
    checkpoint_manifest_hash: str = "",
    torch_version: str = "",
) -> EmbeddingArtifactManifest:
    """Write the compatibility identity for an exported embedding partition."""
    from execsim.data.paper.manifests import file_sha256, write_json_atomic

    manifest = EmbeddingArtifactManifest(
        artifact_id=artifact_id,
        checkpoint_hash=checkpoint_hash,
        partition_identity=partition_identity,
        row_count=row_count,
        shape=(row_count, 644),
        components=("current", "h1", "h2", "h4", "h8", "horizon_availability"),
        training_cutoff=training_cutoff,
        source_hashes=source_hashes,
        parquet_sha256=file_sha256(parquet_path),
        sequence_manifest_hash=sequence_manifest_hash,
        normalization_hash=normalization_hash,
        paper_config_hash=paper_config_hash,
        checkpoint_manifest_hash=checkpoint_manifest_hash,
        torch_version=torch_version,
    )
    write_json_atomic(path, asdict(manifest))
    return manifest
