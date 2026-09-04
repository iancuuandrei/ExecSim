from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from execsim.ml.datasets import load_dataset_manifest, load_split_manifest
from execsim.ml.datasets.validation import load_dataset_rows, validate_dataset_rows
from execsim.ml.models.registry import create_model
from execsim.ml.training.artifacts import (
    LocalArtifactStore,
    artifact_id,
    base_artifact_metadata,
)
from execsim.ml.training.config import TrainingConfig
from execsim.ml.training.metrics import regression_metrics

DownstreamEvaluator = Callable[[pd.DataFrame], dict[str, float]]


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    dataset_id: str
    data_classification: str
    feature_schema: str
    target_schema: str
    feature_names: tuple[str, ...]
    target_name: str
    folds: tuple[str, ...]
    model_family: str
    hyperparameter_grid: tuple[dict[str, object], ...]
    estimated_rows: int
    artifact_destination: str
    execution_evaluation_plan: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    plan: TrainingPlan
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    artifact_paths: tuple[Path, ...]


def build_training_plan(config: TrainingConfig) -> TrainingPlan:
    dataset = load_dataset_manifest(config.dataset_manifest_path)
    splits = load_split_manifest(config.split_manifest_path)
    if splits.dataset_id != dataset.dataset_id:
        raise ValueError("Split manifest belongs to a different dataset.")
    warnings: list[str] = []
    if dataset.data_classification != "synthetic_fixture" and not config.allow_historical_training:
        warnings.append(
            "real-data fitting disabled; use --dry-run only until separately authorized"
        )
    if dataset.sample_count < 252:
        warnings.append("dataset has fewer than 252 point-in-time samples")
    return TrainingPlan(
        dataset_id=dataset.dataset_id,
        data_classification=dataset.data_classification,
        feature_schema=dataset.feature_schema_version,
        target_schema=dataset.target_schema_version,
        feature_names=config.feature_names,
        target_name=config.target_name,
        folds=tuple(fold.fold_id for fold in splits.folds),
        model_family=config.model_family,
        hyperparameter_grid=config.hyperparameter_grid,
        estimated_rows=dataset.row_count,
        artifact_destination=str(config.artifact_root),
        execution_evaluation_plan=(
            "generate out-of-sample volume predictions and compare forecast-driven policies "
            "on the same locked test sessions"
            if config.run_downstream_execution_evaluation
            else "disabled"
        ),
        warnings=tuple(warnings),
    )


def run_training(
    config: TrainingConfig,
    *,
    dry_run: bool = True,
    downstream_evaluator: DownstreamEvaluator | None = None,
) -> TrainingPlan | TrainingResult:
    plan = build_training_plan(config)
    if dry_run:
        return plan
    dataset = load_dataset_manifest(config.dataset_manifest_path)
    if dataset.data_classification != "synthetic_fixture" and not config.allow_historical_training:
        raise PermissionError(
            "Historical model fitting is disabled. This V1 task permits only dry runs or tiny "
            "synthetic-fixture fits."
        )
    splits = load_split_manifest(config.split_manifest_path)
    rows = load_dataset_rows(dataset, config.dataset_manifest_path.parent)
    errors = validate_dataset_rows(rows, dataset)
    if errors:
        raise ValueError("Invalid training dataset: " + "; ".join(errors))
    missing = set((*config.feature_names, config.target_name)).difference(rows.columns)
    if missing:
        raise ValueError(f"Training columns are absent: {sorted(missing)}")
    metrics_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    artifact_paths: list[Path] = []
    store = LocalArtifactStore(config.artifact_root)

    for fold in splits.folds:
        train = rows.loc[rows["session_date"].isin(fold.train_dates)].copy()
        validation = rows.loc[rows["session_date"].isin(fold.validation_dates)].copy()
        test = rows.loc[rows["session_date"].isin(fold.test_dates)].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError(f"Fold {fold.fold_id} contains an empty partition.")
        train_x, train_y = _arrays(train, config)
        validation_x, validation_y = _arrays(validation, config)
        best_parameters: dict[str, object] | None = None
        best_validation_rmse = float("inf")
        for parameters in config.hyperparameter_grid:
            candidate = create_model(
                config.model_family, parameters=parameters, seed=config.random_seed
            )
            candidate.fit(train_x, train_y)
            score = regression_metrics(validation_y, candidate.predict(validation_x))["rmse"]
            if score < best_validation_rmse:
                best_validation_rmse = score
                best_parameters = parameters
        if best_parameters is None:
            raise RuntimeError(f"No valid model candidate for {fold.fold_id}.")

        fit_rows = (
            pd.concat([train, validation], ignore_index=True)
            if config.refit_train_validation
            else train
        )
        fit_x, fit_y = _arrays(fit_rows, config)
        selected = create_model(
            config.model_family, parameters=best_parameters, seed=config.random_seed
        )
        selected.fit(fit_x, fit_y)
        test_x, test_y = _arrays(test, config)
        test_prediction = selected.predict(test_x)
        test_metrics = regression_metrics(test_y, test_prediction)
        prediction_rows = test.loc[
            :, ["sample_id", "symbol", "session_date", "as_of", "target_bucket_timestamp"]
        ].copy()
        prediction_rows["fold_id"] = fold.fold_id
        prediction_rows["actual"] = test_y
        prediction_rows["prediction"] = test_prediction
        predictions.append(prediction_rows)
        downstream = (
            downstream_evaluator(prediction_rows)
            if downstream_evaluator is not None and config.run_downstream_execution_evaluation
            else {}
        )
        metric_row: dict[str, object] = {
            "fold_id": fold.fold_id,
            "validation_rmse": best_validation_rmse,
            **{f"test_{name}": value for name, value in test_metrics.items()},
            "selected_parameters": str(best_parameters),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        }
        metrics_rows.append(metric_row)
        id_payload = {
            "dataset": dataset.dataset_id,
            "split": splits.split_id,
            "fold": fold.fold_id,
            "family": config.model_family,
            "parameters": best_parameters,
            "features": config.feature_names,
            "target": config.target_name,
            "seed": config.random_seed,
        }
        model_artifact_id = artifact_id(id_payload)
        metadata = base_artifact_metadata(
            artifact_id=model_artifact_id,
            model_family=config.model_family,
            model_parameters=best_parameters,
            feature_names=config.feature_names,
            feature_schema_version=dataset.feature_schema_version,
            target_name=config.target_name,
            target_schema_version=dataset.target_schema_version,
            source_manifest_hash=dataset.manifest_hash(),
            split_id=splits.split_id,
            fold_id=fold.fold_id,
            training_cutoff=max(
                fold.validation_dates if config.refit_train_validation else fold.train_dates
            ),
            validation_range=(min(fold.validation_dates), max(fold.validation_dates)),
            test_range=(min(fold.test_dates), max(fold.test_dates)),
            random_seed=config.random_seed,
            bucket_minutes=dataset.bucket_minutes,
            timezone=dataset.timezone,
            forecast_horizon=None,
            metrics={
                "validation_rmse": best_validation_rmse,
                **{f"test_{name}": value for name, value in test_metrics.items()},
            },
            downstream_tca=downstream,
        )
        artifact_paths.append(store.save(selected, metadata))
    return TrainingResult(
        plan=plan,
        fold_metrics=pd.DataFrame(metrics_rows),
        predictions=pd.concat(predictions, ignore_index=True),
        artifact_paths=tuple(artifact_paths),
    )


def plan_to_dict(plan: TrainingPlan) -> dict[str, object]:
    return asdict(plan)


def _arrays(rows: pd.DataFrame, config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    features = rows.loc[:, config.feature_names].to_numpy(dtype=float)
    target = rows[config.target_name].to_numpy(dtype=float)
    return features, target
