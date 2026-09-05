"""Deterministic dry-run expansion and compute planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from execsim.data.paper.manifests import read_json


@dataclass(frozen=True, slots=True)
class PaperComputePlan:
    """Describe the locked run matrix without launching work."""

    folds: int
    seeds: tuple[int, ...]
    geometries: tuple[str, ...]
    predictor_families: tuple[str, ...]
    representation_runs: int
    lightgbm_rows: tuple[str, ...]
    network_enabled: bool
    historical_training_enabled: bool
    full_run_enabled: bool
    development_runs: int
    appendix_runs: int
    maximum_representation_runs: int
    expected_primary_training_runs: int
    expected_sequence_sessions: str
    expected_batches_and_steps: str
    expected_embedding_rows: str
    expected_lightgbm_fits: int
    expected_mpc_simulations: str
    estimated_output_storage: str
    estimated_peak_host_memory: str
    estimated_gpu_hours: str
    resource_gate: str


@dataclass(frozen=True, slots=True)
class PaperKernelProfile:
    """Measured bounded profile; never an unverified historical runtime promise."""

    device: str
    rdmreg_batch_rows_per_second: float
    rdmreg_batch_seconds: float
    embedding_rows_per_second: float
    peak_gpu_bytes: int | None
    sequence_rows_per_second: float | None
    sequence_rows_measured: int
    tca_simulations_per_second: float
    profile_scope: str = "synthetic_or_bounded_local_profile"


def profile_paper_kernels(sequence_manifest: Path | None = None) -> PaperKernelProfile:
    """Measure core kernels on bounded local inputs without fitting historical models."""
    import torch

    from execsim.ml.representations.embeddings import export_frozen_embedding_batch
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.rdmreg import sliced_wasserstein_distance
    from execsim.ml.representations.schemas import RepresentationConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    values = torch.randn((256, 128), device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = perf_counter()
    sliced_wasserstein_distance(
        values,
        p=2.0,
        mu=-0.6744897501960817,
        sigma=1.0,
        projections=512,
        seed=13,
        rectify_target=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    rdm_seconds = perf_counter() - started
    model = PredictiveRepresentationModel(RepresentationConfig("sparse")).to(device)
    context = torch.randn((256, 8, 18), device=device)
    mask = torch.ones((256, 8), dtype=torch.bool, device=device)
    horizon_mask = torch.ones((256, 4), dtype=torch.bool, device=device)
    started = perf_counter()
    export_frozen_embedding_batch(model, context, mask, horizon_mask)
    if device == "cuda":
        torch.cuda.synchronize()
    embedding_seconds = perf_counter() - started
    peak_gpu = torch.cuda.max_memory_allocated() if device == "cuda" else None
    sequence_rate = None
    sequence_rows = 0
    if sequence_manifest is not None:
        from execsim.ml.sequences.streaming import PaperSequenceDataset, build_sequence_dataloader

        dataset = PaperSequenceDataset(
            sequence_manifest,
            partition="train",
            seed=13,
            cache_size=8,
            sample_train_positions=False,
        )
        loader = build_sequence_dataloader(
            dataset, batch_size=256, num_workers=0, device=device, prefetch_factor=1
        )
        started = perf_counter()
        for index, batch in enumerate(loader):
            sequence_rows += len(batch["sample_id"])
            if index == 7:
                break
        sequence_rate = sequence_rows / max(perf_counter() - started, 1e-12)
    tca_rate = _profile_tca_simulation()
    return PaperKernelProfile(
        device=device,
        rdmreg_batch_rows_per_second=256 / max(rdm_seconds, 1e-12),
        rdmreg_batch_seconds=rdm_seconds,
        embedding_rows_per_second=256 / max(embedding_seconds, 1e-12),
        peak_gpu_bytes=peak_gpu,
        sequence_rows_per_second=sequence_rate,
        sequence_rows_measured=sequence_rows,
        tca_simulations_per_second=tca_rate,
    )


def _profile_tca_simulation() -> float:
    from datetime import date, time

    import pandas as pd

    from execsim.forecasting.models import VolumeForecast
    from execsim.orders import ParentOrder
    from execsim.policies import ExecutionConstraints
    from execsim.simulator import simulate_policy

    from .tca import SegmentCommittedMPCPolicy

    timestamps = tuple(
        pd.date_range("2024-01-03 10:30", periods=30, freq="min", tz="America/New_York")
    )
    bars = pd.DataFrame(
        {
            "symbol": "PROFILE",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 10_000.0,
            "trade_count": 100,
            "vwap": 100.0,
        }
    )

    class ProfileProvider:
        provider_id = "bounded-profile"

        def forecast(self, **request: object) -> VolumeForecast:
            buckets: tuple[pd.Timestamp, ...] = tuple(
                pd.Timestamp(value) for value in cast(Any, request["bucket_timestamps"])
            )
            generated = pd.Timestamp(request["generated_at"])
            volumes = np.full(len(buckets), 10_000.0)
            return VolumeForecast(
                symbol="PROFILE",
                session_date=date(2024, 1, 3),
                generated_at=generated,
                first_forecast_bucket=buckets[0],
                bucket_timestamps=buckets,
                expected_volumes=tuple(volumes),
                normalized_shares=tuple(volumes / volumes.sum()),
                expected_remaining_volume=float(volumes.sum()),
                forecaster_id=self.provider_id,
                feature_schema_version="bounded-profile-v1",
                training_data_cutoff=date(2024, 1, 2),
                data_manifest_hash="a" * 64,
            )

    started = perf_counter()
    simulate_policy(
        parent_order=ParentOrder(
            "PROFILE", "buy", 1_000, date(2024, 1, 3), time(10, 30), time(11, 0)
        ),
        bars=bars,
        policy=SegmentCommittedMPCPolicy(half_spread=0.005, temporary_impact=0.1),
        constraints=ExecutionConstraints(0.1, 0.1),
        forecast_provider=ProfileProvider(),
    )
    return 1.0 / max(perf_counter() - started, 1e-12)


def build_compute_plan(
    *,
    network_enabled: bool = False,
    historical_training_enabled: bool = False,
    full_run_enabled: bool = False,
) -> PaperComputePlan:
    """Expand the primary design while leaving all expensive capabilities disabled."""
    seeds = (13, 29, 47)
    geometries = ("dense", "sparse")
    predictors = ("affine_ridge", "mlp_64", "mlp_256")
    return PaperComputePlan(
        folds=3,
        seeds=seeds,
        geometries=geometries,
        predictor_families=predictors,
        representation_runs=3 * len(seeds) * len(geometries),
        lightgbm_rows=("ewma", "raw", "untrained_neural", "dense", "sparse"),
        network_enabled=network_enabled,
        historical_training_enabled=historical_training_enabled,
        full_run_enabled=full_run_enabled,
        development_runs=6,
        appendix_runs=5,
        maximum_representation_runs=29,
        expected_primary_training_runs=18,
        expected_sequence_sessions=(
            "up to 100 stocks x eligible 2022-2025 XNYS sessions; exact after exclusions"
        ),
        expected_batches_and_steps=(
            "2 train samples/session/epoch; ceil(samples/256) x at most 40 epochs/run"
        ),
        expected_embedding_rows=(
            "up to 22 valid as-of rows/session/checkpoint; exact after horizon masks"
        ),
        expected_lightgbm_fits=384,
        expected_mpc_simulations=(
            "seed-specific matched methods x 30 stocks x locked test dates; exact after exclusions"
        ),
        estimated_output_storage=(
            "644 float32 values plus Parquet metadata per exported row; exact after manifests"
        ),
        estimated_peak_host_memory=(
            "bounded session cache + one DataLoader prefetch window + 1032x1032 ridge statistics"
        ),
        estimated_gpu_hours="NOT MEASURED until a representative authorized device profile exists",
        resource_gate="PASS: declared representation matrix is at the configured 29-run ceiling",
    )


def estimate_manifest_resources(
    manifests: tuple[Path, ...],
    *,
    batch_size: int,
    max_epochs: int,
    bounds: dict[str, int],
) -> dict[str, int | str]:
    """Derive fail-closed historical workload bounds from completed sequence manifests."""
    if len(manifests) != 3 or batch_size <= 0 or max_epochs <= 0:
        raise ValueError("Resource estimation requires three folds and positive training settings.")
    primary_runs = 18
    development_runs = 6
    appendix_runs = 5
    jepa_steps = 0
    shape_rows = 0
    embedding_rows = 0
    sessions = 0
    for path in manifests:
        payload = read_json(path)
        counts = {name: int(value) for name, value in payload["partition_counts"].items()}
        sessions += sum(counts.values())
        jepa_steps += int(np.ceil(2 * counts["train"] / batch_size)) * max_epochs * 6
        shape_rows += 88 * counts["train"] + 253 * (counts["validation"] + counts["test"])
        embedding_rows += 22 * sum(counts.values()) * 6
    embedding_bytes = embedding_rows * 644 * 4
    values = {
        "sequence_sessions_across_expanding_folds": sessions,
        "maximum_primary_jepa_steps": jepa_steps,
        "maximum_long_shape_rows": shape_rows,
        "embedding_rows": embedding_rows,
        "uncompressed_embedding_value_bytes": embedding_bytes,
        "primary_representation_runs": primary_runs,
        "development_representation_runs": development_runs,
        "appendix_representation_runs": appendix_runs,
        "maximum_representation_runs": primary_runs + development_runs + appendix_runs,
    }
    checks = {
        "maximum_representation_runs": values["maximum_representation_runs"],
        "maximum_jepa_steps": jepa_steps,
        "maximum_shape_rows": shape_rows,
        "maximum_embedding_bytes": embedding_bytes,
    }
    exceeded = [name for name, value in checks.items() if value > int(bounds[name])]
    if exceeded:
        raise RuntimeError(f"BLOCKED: paper resource estimate exceeds safe bounds: {exceeded}")
    return {**values, "resource_gate": "PASS"}


def predictor_capacity_smoke(
    *, seed: int = 13, repetitions: int = 3
) -> tuple[dict[str, object], ...]:
    """Measure the frozen P0/P1/P2 probe cost on a deterministic fixture."""
    import torch
    from torch import nn

    if repetitions <= 0:
        raise ValueError("Predictor benchmark repetitions must be positive.")
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn((8, 644), generator=generator)
    targets = features[:, :128].unsqueeze(1).repeat(1, 4, 1) + 0.1 * torch.randn(
        (8, 4, 128), generator=generator
    )
    rows = []
    for family, hidden in (("affine_ridge", None), ("mlp_64", 64), ("mlp_256", 256)):
        torch.manual_seed(seed)
        predictor = (
            nn.Linear(644, 128)
            if hidden is None
            else nn.Sequential(nn.Linear(644, hidden), nn.GELU(), nn.Linear(hidden, 128))
        ).eval()
        predictions = []
        timings = []
        with torch.no_grad():
            for _horizon in range(4):
                started = perf_counter()
                output = None
                for _ in range(repetitions):
                    output = predictor(features)
                timings.append((perf_counter() - started) * 1_000 / repetitions)
                if output is None:
                    raise RuntimeError("Predictor benchmark executed no forward pass.")
                predictions.append(output)
        stacked = torch.stack(predictions, dim=1)
        mse = torch.mean((stacked - targets) ** 2, dim=(0, 2)).numpy()
        identity = torch.mean((features[:, :128].unsqueeze(1) - targets) ** 2, dim=(0, 2)).numpy()
        macs = sum(
            module.in_features * module.out_features
            for module in predictor.modules()
            if isinstance(module, nn.Linear)
        )
        rows.append(
            {
                "predictor_family": family,
                "parameters": sum(parameter.numel() for parameter in predictor.parameters()),
                "approximate_linear_macs": macs,
                "median_forward_ms": float(np.median(timings)),
                "mse_by_horizon": mse.tolist(),
                "identity_mse_by_horizon": identity.tolist(),
            }
        )
    return tuple(rows)
