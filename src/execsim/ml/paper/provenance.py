"""Complete, non-feature provenance for paper runs."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def build_run_provenance(
    *,
    paper_run_id: str,
    supplied: dict[str, object] | None = None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Capture code, environment, hashes, and locked configuration identities."""
    root = repository_root or Path.cwd()
    commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    dependencies = {
        name: _package_version(name)
        for name in ("numpy", "pandas", "pyarrow", "torch", "safetensors", "lightgbm", "osqp")
    }
    payload: dict[str, object] = {
        "paper_run_id": paper_run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit or "UNAVAILABLE",
        "working_tree_status": "dirty" if dirty else "clean",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "UNAVAILABLE",
        "dependency_versions": dependencies,
        "cuda_version": _torch_runtime_value("version.cuda"),
        "cudnn_version": _torch_runtime_value("backends.cudnn.version"),
        "raw_data_hashes": [],
        "processed_data_hashes": [],
        "universe_manifest_hash": "NOT RUN",
        "corporate_action_manifest_hash": "NOT RUN",
        "sequence_manifest_hash": "NOT RUN",
        "fold_manifest_hash": "NOT RUN",
        "normalization_manifest_hash": "NOT RUN",
        "model_config_hash": "NOT RUN",
        "checkpoint_hashes": [],
        "embedding_hashes": [],
        "downstream_model_hashes": [],
        "tca_config_hash": "NOT RUN",
        "statistics_config_hash": "NOT RUN",
    }
    payload.update(supplied or {})
    return payload


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def _torch_runtime_value(path: str) -> object:
    try:
        import torch
    except ImportError:
        return "NOT INSTALLED"
    value: object = torch
    for part in path.split("."):
        value = getattr(value, part)
    return value() if callable(value) else value or "UNAVAILABLE"
