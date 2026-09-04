from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    artifact_id: str
    model_family: str
    model_parameters: dict[str, object]
    feature_names: tuple[str, ...]
    feature_schema_version: str
    target_name: str
    target_schema_version: str
    source_manifest_hash: str
    split_id: str
    fold_id: str
    training_cutoff: str
    validation_range: tuple[str, str]
    test_range: tuple[str, str]
    random_seed: int
    bucket_minutes: int
    timezone: str
    forecast_horizon: int | None
    metrics: dict[str, float]
    downstream_tca: dict[str, float]
    dependency_versions: dict[str, str]
    package_version: str
    git_commit: str | None
    created_at: str
    model_checksum: str


@runtime_checkable
class ArtifactStore(Protocol):
    """Storage boundary for versioned model artifacts."""

    def save(self, model: object, metadata: ArtifactMetadata) -> Path: ...

    def load(
        self,
        artifact_id: str,
        *,
        feature_schema_version: str,
        target_schema_version: str,
        bucket_minutes: int,
        timezone: str,
        forecast_horizon: int | None = None,
    ) -> tuple[object, ArtifactMetadata]: ...


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, model: object, metadata: ArtifactMetadata) -> Path:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the 'ml' extra to persist model artifacts.") from exc
        artifact_dir = self.root / metadata.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        model_path = artifact_dir / "model.joblib"
        joblib.dump(model, model_path)
        checksum = _file_hash(model_path)
        if metadata.model_checksum and metadata.model_checksum != checksum:
            raise ValueError("Provided artifact checksum does not match the serialized model.")
        payload = asdict(metadata)
        payload["model_checksum"] = checksum
        (artifact_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return artifact_dir

    def load(
        self,
        artifact_id: str,
        *,
        feature_schema_version: str,
        target_schema_version: str,
        bucket_minutes: int,
        timezone: str,
        forecast_horizon: int | None = None,
    ) -> tuple[object, ArtifactMetadata]:
        import joblib

        artifact_dir = self.root / artifact_id
        payload = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        payload["feature_names"] = tuple(payload["feature_names"])
        payload["validation_range"] = tuple(payload["validation_range"])
        payload["test_range"] = tuple(payload["test_range"])
        metadata = ArtifactMetadata(**payload)
        expected = {
            "feature_schema_version": feature_schema_version,
            "target_schema_version": target_schema_version,
            "bucket_minutes": bucket_minutes,
            "timezone": timezone,
        }
        if forecast_horizon is not None:
            expected["forecast_horizon"] = forecast_horizon
        for field, value in expected.items():
            if getattr(metadata, field) != value:
                raise ValueError(
                    f"Artifact {artifact_id} is incompatible: "
                    f"{field}={getattr(metadata, field)!r}, "
                    f"expected {value!r}."
                )
        current_package_version = importlib.metadata.version("execution-cost-sim")
        if metadata.package_version != current_package_version:
            raise ValueError(
                f"Artifact {artifact_id} is incompatible: "
                f"package_version={metadata.package_version!r}, "
                f"expected {current_package_version!r}."
            )
        model_path = artifact_dir / "model.joblib"
        if _file_hash(model_path) != metadata.model_checksum:
            raise ValueError("Artifact model checksum does not match metadata.")
        return joblib.load(model_path), metadata


def base_artifact_metadata(**values: object) -> ArtifactMetadata:
    dependencies = {}
    for package in ("numpy", "pandas", "scikit-learn"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "NOT_INSTALLED"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return ArtifactMetadata(
        dependency_versions=dependencies,
        package_version=importlib.metadata.version("execution-cost-sim"),
        git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        created_at=datetime.now(UTC).isoformat(),
        model_checksum="",
        **values,  # type: ignore[arg-type]
    )


def artifact_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return "model-" + hashlib.sha256(encoded).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
