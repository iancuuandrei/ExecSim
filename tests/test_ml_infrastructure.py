from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from execsim.data.scenarios import ScenarioConfig, generate_scenario
from execsim.ml.datasets import DatasetBuildConfig, WalkForwardConfig, build_dataset
from execsim.ml.datasets.splits import create_walk_forward_splits
from execsim.ml.datasets.validation import load_dataset_rows, validate_dataset_rows
from execsim.ml.features import DEFAULT_FEATURE_REGISTRY, validate_feature_values
from execsim.ml.forecasts import positive_volume_forecast, stable_softmax
from execsim.ml.models import SklearnRegressorAdapter
from execsim.ml.schemas import FeatureValue
from execsim.ml.training import TrainingConfig, build_training_plan, run_training
from execsim.ml.training.artifacts import LocalArtifactStore
from execsim.ml.training.metrics import regression_metrics, shape_metrics


def _sessions(count: int = 10) -> pd.DataFrame:
    frames = []
    day = date(2026, 1, 5)
    while len(frames) < count:
        if day.weekday() < 5:
            frames.append(
                generate_scenario(
                    ScenarioConfig(
                        symbol="AAPL",
                        session_date=day,
                        n_buckets=4,
                        base_volume=100 + len(frames) * 10,
                        volume_scenario="u_shaped",
                        seed=len(frames),
                    )
                )
            )
        day += timedelta(days=1)
    return pd.concat(frames, ignore_index=True)


def test_feature_registry_is_complete_and_rejects_future_availability() -> None:
    spec = DEFAULT_FEATURE_REGISTRY.get("rolling_adv")
    as_of = pd.Timestamp("2026-01-06 09:30", tz="America/New_York")
    assert spec.rationale
    with pytest.raises(ValueError, match="after as_of"):
        validate_feature_values(
            [FeatureValue("rolling_adv", 100.0, as_of + pd.Timedelta(minutes=1))],
            as_of=as_of,
            registry=DEFAULT_FEATURE_REGISTRY,
        )


def test_static_dataset_is_point_in_time_normalized_and_reproducible(tmp_path: Path) -> None:
    config = DatasetBuildConfig(
        mode="static",
        bucket_minutes=1,
        require_calendar_complete=False,
        data_classification="synthetic_fixture",
    )
    first = build_dataset(output_root=tmp_path / "first", config=config, bars=_sessions())
    second = build_dataset(output_root=tmp_path / "second", config=config, bars=_sessions())
    loaded = load_dataset_rows(first.manifest, first.manifest_path.parent)

    assert first.manifest.dataset_id == second.manifest.dataset_id
    assert first.manifest.manifest_hash() == second.manifest.manifest_hash()
    assert not validate_dataset_rows(loaded, first.manifest)
    assert (pd.to_datetime(loaded["feature_available_at"]) < pd.to_datetime(loaded["as_of"])).all()
    shares = loaded.groupby("sample_id")["target_volume_share"].sum()
    assert (shares - 1.0).abs().max() < 1e-12


def test_dynamic_dataset_has_explicit_as_of_and_only_future_targets(tmp_path: Path) -> None:
    result = build_dataset(
        output_root=tmp_path,
        config=DatasetBuildConfig(
            mode="dynamic",
            bucket_minutes=1,
            require_calendar_complete=False,
            data_classification="synthetic_fixture",
        ),
        bars=_sessions(4),
    )
    rows = result.rows

    assert (pd.to_datetime(rows["feature_available_at"]) < pd.to_datetime(rows["as_of"])).all()
    assert (pd.to_datetime(rows["target_bucket_timestamp"]) >= pd.to_datetime(rows["as_of"])).all()
    conditional = rows.groupby("sample_id")["target_conditional_share"].sum()
    assert (conditional - 1.0).abs().max() < 1e-12


def test_parquet_builder_scans_by_symbol_without_materializing_result_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _sessions(4).to_parquet(source, index=False)

    result = build_dataset(
        output_root=tmp_path / "dataset",
        source_paths=(source,),
        materialize_result_rows=False,
        config=DatasetBuildConfig(
            mode="static",
            bucket_minutes=1,
            require_calendar_complete=False,
            data_classification="synthetic_fixture",
        ),
    )

    assert result.rows.empty
    assert result.manifest.row_count == 12
    loaded = load_dataset_rows(result.manifest, result.manifest_path.parent)
    assert len(loaded) == result.manifest.row_count


def test_walk_forward_splits_are_global_chronological_and_disjoint(tmp_path: Path) -> None:
    dataset = build_dataset(
        output_root=tmp_path,
        config=DatasetBuildConfig(
            mode="static",
            bucket_minutes=1,
            require_calendar_complete=False,
            data_classification="synthetic_fixture",
        ),
        bars=_sessions(10),
    )
    split = create_walk_forward_splits(
        dataset.rows,
        dataset_id=dataset.manifest.dataset_id,
        config=WalkForwardConfig(
            initial_train_sessions=4,
            validation_sessions=2,
            test_sessions=2,
            step_sessions=1,
        ),
    )

    assert split.folds
    for fold in split.folds:
        assert max(fold.train_dates) < min(fold.validation_dates) < min(fold.test_dates)
        assert not set(fold.train_dates) & set(fold.validation_dates)
        assert not set(fold.validation_dates) & set(fold.test_dates)


def test_forecast_normalization_and_metrics_are_finite() -> None:
    probabilities = stable_softmax([1_000.0, 999.0, -1_000.0])
    volumes, shares = positive_volume_forecast([-1.0, 2.0, 3.0])

    assert probabilities.sum() == pytest.approx(1.0)
    assert volumes.tolist() == [0.0, 2.0, 3.0]
    assert shares.sum() == pytest.approx(1.0)
    assert regression_metrics([1.0, 2.0], [1.5, 2.5])["mae"] == pytest.approx(0.5)
    assert shape_metrics([0.5, 0.5], [0.4, 0.6])["cumulative_curve_error"] > 0


@pytest.mark.parametrize("family", ["ridge", "elastic_net", "hist_gradient_boosting"])
def test_model_adapters_fit_only_tiny_synthetic_arrays(family: str) -> None:
    features = pd.DataFrame({"x": range(30), "z": [value % 3 for value in range(30)]}).to_numpy(
        dtype=float
    )
    target = features[:, 0] * 0.1 + features[:, 1]
    adapter = SklearnRegressorAdapter(family=family)  # type: ignore[arg-type]
    adapter.fit(features[:20], target[:20])

    assert adapter.predict(features[20:]).shape == (10,)
    assert adapter.fitted_preprocessor_mean == pytest.approx(features[:20].mean(axis=0))


def _training_setup(tmp_path: Path, classification: str = "synthetic_fixture"):
    dataset = build_dataset(
        output_root=tmp_path / "datasets",
        config=DatasetBuildConfig(
            mode="static",
            bucket_minutes=1,
            require_calendar_complete=False,
            data_classification=classification,
        ),
        bars=_sessions(10),
    )
    split = create_walk_forward_splits(
        dataset.rows,
        dataset_id=dataset.manifest.dataset_id,
        config=WalkForwardConfig(4, 2, 2, 2),
    )
    split_path = split.write(dataset.manifest_path.parent / "splits.json")
    config = TrainingConfig(
        dataset_manifest_path=dataset.manifest_path,
        split_manifest_path=split_path,
        feature_names=(
            "weekday",
            "month",
            "bucket_index",
            "previous_session_total_volume",
            "rolling_adv",
        ),
        target_name="target_volume_share",
        model_family="ridge",
        hyperparameter_grid=({"alpha": 0.1}, {"alpha": 1.0}),
        artifact_root=tmp_path / "artifacts",
        run_downstream_execution_evaluation=False,
    )
    return dataset, split, config


def test_training_dry_run_validates_plan_without_artifacts(tmp_path: Path) -> None:
    dataset, split, config = _training_setup(tmp_path)
    plan = build_training_plan(config)
    dry_run = run_training(config, dry_run=True)

    assert plan == dry_run
    assert plan.dataset_id == dataset.manifest.dataset_id
    assert plan.folds == tuple(fold.fold_id for fold in split.folds)
    assert not config.artifact_root.exists()


def test_tiny_synthetic_training_pipeline_and_artifact_compatibility(tmp_path: Path) -> None:
    dataset, _, config = _training_setup(tmp_path)
    result = run_training(config, dry_run=False)

    assert result.fold_metrics["test_rmse"].notna().all()
    assert not result.predictions.empty
    assert result.artifact_paths
    artifact_id = result.artifact_paths[0].name
    _, metadata = LocalArtifactStore(config.artifact_root).load(
        artifact_id,
        feature_schema_version=dataset.manifest.feature_schema_version,
        target_schema_version=dataset.manifest.target_schema_version,
        bucket_minutes=dataset.manifest.bucket_minutes,
        timezone=dataset.manifest.timezone,
    )
    assert metadata.model_family == "ridge"
    with pytest.raises(ValueError, match="incompatible"):
        LocalArtifactStore(config.artifact_root).load(
            artifact_id,
            feature_schema_version="wrong",
            target_schema_version=dataset.manifest.target_schema_version,
            bucket_minutes=dataset.manifest.bucket_minutes,
            timezone=dataset.manifest.timezone,
        )


def test_real_history_training_is_blocked_but_dry_run_is_available(tmp_path: Path) -> None:
    _, _, config = _training_setup(tmp_path, classification="historical")

    plan = run_training(config, dry_run=True)
    assert any("real-data fitting disabled" in warning for warning in plan.warnings)
    with pytest.raises(PermissionError, match="Historical model fitting is disabled"):
        run_training(config, dry_run=False)
    assert not config.artifact_root.exists()
