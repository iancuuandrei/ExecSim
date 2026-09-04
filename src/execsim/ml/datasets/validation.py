from __future__ import annotations

from pathlib import Path

import pandas as pd

from execsim.ml.datasets.manifest import DatasetManifest


def load_dataset_rows(manifest: DatasetManifest, dataset_dir: str | Path) -> pd.DataFrame:
    root = Path(dataset_dir)
    frames = [pd.read_parquet(root / partition) for partition in manifest.partitions]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_dataset_rows(rows: pd.DataFrame, manifest: DatasetManifest) -> list[str]:
    errors: list[str] = []
    required = {
        "sample_id",
        "symbol",
        "session_date",
        "as_of",
        "target_bucket_timestamp",
        "feature_available_at",
    }
    missing = required.difference(rows.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    as_of = pd.to_datetime(rows["as_of"])
    available = pd.to_datetime(rows["feature_available_at"])
    targets = pd.to_datetime(rows["target_bucket_timestamp"])
    if (available > as_of).any():
        errors.append("feature availability exceeds sample as_of")
    if (targets < as_of).any():
        errors.append("target bucket precedes sample as_of")
    if len(rows) != manifest.row_count:
        errors.append(f"row count {len(rows)} does not match manifest {manifest.row_count}")
    if rows["sample_id"].nunique() != manifest.sample_count:
        errors.append("sample count does not match manifest")
    if rows[["symbol", "session_date", "sample_id"]].isna().any().any():
        errors.append("sample identity contains missing values")
    return errors
