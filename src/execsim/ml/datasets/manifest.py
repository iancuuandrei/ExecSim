from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    mode: str
    bucket_minutes: int
    timezone: str
    feature_schema_version: str
    target_schema_version: str
    data_classification: str
    row_count: int
    sample_count: int
    symbol_count: int
    symbols: tuple[str, ...]
    min_session_date: str
    max_session_date: str
    source_hashes: dict[str, str]
    partitions: tuple[str, ...]
    filters: dict[str, object]
    exclusions: tuple[dict[str, object], ...]
    git_commit: str | None
    built_at: str

    def stable_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("built_at")
        return payload

    def manifest_hash(self) -> str:
        encoded = json.dumps(self.stable_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def write(self, path: Path) -> None:
        payload = asdict(self)
        payload["manifest_sha256"] = self.manifest_hash()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_hash = payload.pop("manifest_sha256", None)
    payload["symbols"] = tuple(payload["symbols"])
    payload["partitions"] = tuple(payload["partitions"])
    payload["exclusions"] = tuple(payload["exclusions"])
    manifest = DatasetManifest(**payload)
    if claimed_hash is not None and claimed_hash != manifest.manifest_hash():
        raise ValueError("Dataset manifest checksum does not match its content.")
    return manifest
