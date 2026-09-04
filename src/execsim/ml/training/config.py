from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_manifest_path: Path
    split_manifest_path: Path
    feature_names: tuple[str, ...]
    target_name: str
    model_family: str = "ridge"
    hyperparameter_grid: tuple[dict[str, object], ...] = ({"alpha": 1.0},)
    artifact_root: Path = Path("artifacts/models")
    random_seed: int = 17
    refit_train_validation: bool = True
    allow_historical_training: bool = False
    run_downstream_execution_evaluation: bool = True

    def __post_init__(self) -> None:
        if not self.feature_names or not self.target_name:
            raise ValueError("Training features and target must be specified.")
        if not self.hyperparameter_grid:
            raise ValueError("Training hyperparameter_grid must be non-empty.")
