from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    initial_train_sessions: int = 252
    validation_sessions: int = 21
    test_sessions: int = 21
    step_sessions: int = 21
    embargo_sessions: int = 0

    def __post_init__(self) -> None:
        values = (
            self.initial_train_sessions,
            self.validation_sessions,
            self.test_sessions,
            self.step_sessions,
        )
        if any(value <= 0 for value in values) or self.embargo_sessions < 0:
            raise ValueError("Walk-forward windows must be positive and embargo non-negative.")


@dataclass(frozen=True, slots=True)
class FoldManifest:
    fold_id: str
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    dataset_id: str
    config: dict[str, int]
    folds: tuple[FoldManifest, ...]
    split_id: str

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        payload = asdict(self)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output


def load_split_manifest(path: str | Path) -> SplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = tuple(FoldManifest(**fold) for fold in payload["folds"])
    manifest = SplitManifest(
        dataset_id=payload["dataset_id"],
        config=payload["config"],
        folds=folds,
        split_id=payload["split_id"],
    )
    for fold in manifest.folds:
        _validate_fold(fold)
    return manifest


def create_walk_forward_splits(
    rows: pd.DataFrame, *, dataset_id: str, config: WalkForwardConfig
) -> SplitManifest:
    if "session_date" not in rows:
        raise ValueError("Dataset rows must contain session_date.")
    dates = sorted(pd.Series(rows["session_date"].astype(str).unique()).tolist())
    folds: list[FoldManifest] = []
    train_end = config.initial_train_sessions
    fold_number = 1
    while True:
        validation_start = train_end + config.embargo_sessions
        validation_end = validation_start + config.validation_sessions
        test_start = validation_end + config.embargo_sessions
        test_end = test_start + config.test_sessions
        if test_end > len(dates):
            break
        fold = FoldManifest(
            fold_id=f"fold-{fold_number:03d}",
            train_dates=tuple(dates[:train_end]),
            validation_dates=tuple(dates[validation_start:validation_end]),
            test_dates=tuple(dates[test_start:test_end]),
        )
        _validate_fold(fold)
        folds.append(fold)
        fold_number += 1
        train_end += config.step_sessions
    if not folds:
        raise ValueError(
            "Insufficient distinct session dates for one walk-forward fold: "
            f"available={len(dates)}."
        )
    config_payload = asdict(config)
    encoded = json.dumps(
        {
            "dataset_id": dataset_id,
            "config": config_payload,
            "folds": [asdict(fold) for fold in folds],
        },
        sort_keys=True,
    ).encode()
    return SplitManifest(
        dataset_id=dataset_id,
        config=config_payload,
        folds=tuple(folds),
        split_id="split-" + hashlib.sha256(encoded).hexdigest()[:12],
    )


def _validate_fold(fold: FoldManifest) -> None:
    partitions = [set(fold.train_dates), set(fold.validation_dates), set(fold.test_dates)]
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise ValueError(f"Split partitions overlap in {fold.fold_id}.")
    if not (max(fold.train_dates) < min(fold.validation_dates) < min(fold.test_dates)):
        raise ValueError(f"Split dates are not chronological in {fold.fold_id}.")
