"""Frozen-representation accessibility metrics for the redirected paper protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class LatentNormalization:
    """TRAIN-only latent scale used to make geometry errors comparable."""

    covariance_trace: float
    per_dimension_variance: float
    latent_dimension: int


def fit_latent_normalization(train_latents: np.ndarray) -> LatentNormalization:
    """Fit the declared covariance scale from TRAIN latents only."""
    values = np.asarray(train_latents, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Latent normalization requires at least two finite TRAIN rows.")
    centered = values - values.mean(axis=0, keepdims=True)
    trace = float(np.square(centered).sum() / (len(values) - 1))
    if trace <= 0:
        raise ValueError("TRAIN latent covariance trace must be positive.")
    return LatentNormalization(trace, trace / values.shape[1], values.shape[1])


def normalized_latent_mse(
    actual: np.ndarray, predicted: np.ndarray, normalization: LatentNormalization
) -> float:
    """Return per-row squared latent error divided by TRAIN covariance trace."""
    target = np.asarray(actual, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if (
        target.shape != estimate.shape
        or target.ndim != 2
        or target.shape[1] != normalization.latent_dimension
    ):
        raise ValueError("Latent NMSE inputs do not match the fitted normalization.")
    return float(np.mean(np.square(target - estimate).sum(axis=1) / normalization.covariance_trace))


def representation_baselines(
    train_latents: np.ndarray,
    context_latents: np.ndarray,
    target_latents: np.ndarray,
    normalization: LatentNormalization,
) -> dict[str, float]:
    """Evaluate the fixed zero, TRAIN-mean, and persistence baselines."""
    target = np.asarray(target_latents, dtype=float)
    context = np.asarray(context_latents, dtype=float)
    mean = np.asarray(train_latents, dtype=float).mean(axis=0, keepdims=True)
    if context.shape != target.shape:
        raise ValueError("Persistence baseline requires aligned context and target latents.")
    return {
        "zero_nmse": normalized_latent_mse(target, np.zeros_like(target), normalization),
        "train_mean_nmse": normalized_latent_mse(
            target, np.broadcast_to(mean, target.shape), normalization
        ),
        "persistence_nmse": normalized_latent_mse(target, context, normalization),
    }


def complete_horizon_origin_mask(as_of_tokens: np.ndarray) -> np.ndarray:
    """Select RQ1 origins where h1, h2, h4, and h8 are simultaneously observable."""
    values = np.asarray(as_of_tokens, dtype=int)
    return (values >= 4) & (values + 8 - 1 < 26)


def future_volume_surprise(
    actual_bucket_volume: np.ndarray, causal_expected_bucket_volume: np.ndarray
) -> np.ndarray:
    """Construct the fixed observable probe target without representation input."""
    actual = np.asarray(actual_bucket_volume, dtype=float)
    expected = np.asarray(causal_expected_bucket_volume, dtype=float)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Future-volume surprise inputs must be aligned and finite.")
    if (actual < 0).any() or (expected < 0).any():
        raise ValueError("Future-volume surprise volumes must be non-negative.")
    return np.log1p(actual) - np.log1p(expected)


ProbeCapacity = Literal["affine_ridge", "mlp_64", "mlp_256"]


def fit_frozen_capacity_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    capacity: ProbeCapacity,
    seed: int,
    ridge_alpha: float = 1.0,
    epochs: int = 20,
) -> np.ndarray:
    """Fit one output-unlinked probe without changing the frozen representation."""
    train_x = np.asarray(train_features, dtype=np.float32)
    train_y = np.asarray(train_targets, dtype=np.float32)
    evaluate_x = np.asarray(evaluation_features, dtype=np.float32)
    if (
        train_x.ndim != 2
        or train_y.ndim != 2
        or len(train_x) != len(train_y)
        or evaluate_x.ndim != 2
        or evaluate_x.shape[1] != train_x.shape[1]
    ):
        raise ValueError("Frozen probe features and targets have incompatible shapes.")
    if capacity == "affine_ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=ridge_alpha, random_state=seed)
        model.fit(train_x, train_y)
        return np.asarray(model.predict(evaluate_x), dtype=np.float32)
    if capacity not in {"mlp_64", "mlp_256"} or epochs <= 0:
        raise ValueError(f"Unsupported frozen probe capacity: {capacity}")
    import torch
    from torch import nn

    hidden = 64 if capacity == "mlp_64" else 256
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(train_x.shape[1], hidden),
            nn.GELU(),
            nn.Linear(hidden, train_y.shape[1]),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x_tensor = torch.from_numpy(train_x)
        y_tensor = torch.from_numpy(train_y)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x_tensor) - y_tensor) ** 2)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Frozen capacity probe produced non-finite loss.")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            result = model(torch.from_numpy(evaluate_x)).numpy()
    return np.asarray(result, dtype=np.float32)
