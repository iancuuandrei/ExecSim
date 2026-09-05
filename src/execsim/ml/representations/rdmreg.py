"""Generalized-Gaussian targets and sliced-Wasserstein distribution matching."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import gamma, gammaincc
from torch import Tensor


def generalized_gaussian_moments(*, p: float, mu: float, sigma: float) -> tuple[float, float]:
    """Return analytical mean and variance for shape ``p`` and scale ``sigma``."""
    if p <= 0 or sigma <= 0 or not all(math.isfinite(value) for value in (p, mu, sigma)):
        raise ValueError("Generalized-Gaussian p and sigma must be finite and positive.")
    scale = p ** (1.0 / p) * sigma
    variance = scale**2 * float(gamma(3.0 / p) / gamma(1.0 / p))
    return mu, variance


def rectified_generalized_gaussian_moments(
    *, p: float, mu: float, sigma: float
) -> tuple[float, float]:
    """Return E[ReLU(X)] and E[ReLU(X)^2] for a shifted generalized Gaussian.

    The symmetric pre-link density uses scale ``a = p**(1/p) * sigma``. Integer
    positive-part moments reduce to upper incomplete gamma tails. The branch for
    positive ``mu`` subtracts the negative tail from the full raw moment, which
    avoids numerical integration and remains stable at ``mu=0``.
    """
    if p <= 0 or sigma <= 0 or not all(math.isfinite(value) for value in (p, mu, sigma)):
        raise ValueError("Generalized-Gaussian p and sigma must be finite and positive.")
    scale = p ** (1.0 / p) * sigma
    threshold = abs(mu) / scale
    tail_argument = threshold**p
    normalizer = float(gamma(1.0 / p))

    def upper(order: int) -> float:
        shape = (order + 1.0) / p
        return float(gamma(shape) * gammaincc(shape, tail_argument))

    tail_probability_term = upper(0)
    tail_first = scale * upper(1)
    tail_second = scale**2 * upper(2)
    if mu <= 0:
        first = (mu * tail_probability_term + tail_first) / (2.0 * normalizer)
        second = (mu**2 * tail_probability_term + 2.0 * mu * tail_first + tail_second) / (
            2.0 * normalizer
        )
    else:
        _, variance = generalized_gaussian_moments(p=p, mu=mu, sigma=sigma)
        negative_first = (mu * tail_probability_term - tail_first) / (2.0 * normalizer)
        negative_second = (mu**2 * tail_probability_term - 2.0 * mu * tail_first + tail_second) / (
            2.0 * normalizer
        )
        first = mu - negative_first
        second = mu**2 + variance - negative_second
    return max(first, 0.0), max(second, 0.0)


def generalized_gaussian_samples(
    shape: tuple[int, ...], *, p: float, mu: float, sigma: float, seed: int
) -> np.ndarray:
    """Sample the symmetric generalized-Gaussian target with a local RNG."""
    if p <= 0 or sigma <= 0:
        raise ValueError("Generalized-Gaussian p and sigma must be positive.")
    rng = np.random.default_rng(seed)
    magnitude = rng.gamma(shape=1.0 / p, scale=1.0, size=shape) ** (1.0 / p)
    sign = rng.choice(np.asarray([-1.0, 1.0]), size=shape)
    scale = p ** (1.0 / p) * sigma
    return (mu + scale * sign * magnitude).astype(np.float32)


def sliced_wasserstein_distance(
    latents: Tensor,
    *,
    p: float,
    mu: float,
    sigma: float,
    projections: int,
    seed: int,
    rectify_target: bool = False,
    target_rms: float | None = None,
) -> Tensor:
    """Compute mean squared one-dimensional Wasserstein distance over unit projections."""
    if latents.ndim != 2 or len(latents) < 2 or projections <= 0:
        raise ValueError("RDMReg requires at least two [sample, latent] rows and projections.")
    with torch.autocast(device_type=latents.device.type, enabled=False):
        values = latents.float()
        generator = torch.Generator(device=values.device).manual_seed(seed)
        directions = torch.randn(
            (projections, values.shape[1]),
            generator=generator,
            device=values.device,
            dtype=torch.float32,
        )
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        target_np = generalized_gaussian_samples(
            tuple(values.shape), p=p, mu=mu, sigma=sigma, seed=seed + 1_000_003
        )
        if rectify_target:
            if target_rms is None:
                _, second_moment = rectified_generalized_gaussian_moments(p=p, mu=mu, sigma=sigma)
                target_rms = math.sqrt(second_moment)
            if target_rms <= 0:
                raise ValueError("Rectified target RMS must be positive.")
            target_np = np.maximum(target_np, 0.0) / target_rms
        target = torch.as_tensor(target_np, device=values.device, dtype=torch.float32)
        projected = torch.sort(values @ directions.T, dim=0).values
        target_projected = torch.sort(target @ directions.T, dim=0).values
        return torch.mean((projected - target_projected) ** 2)


def calibrate_rdm_lambda(
    prediction_gradient_norm: float, rdm_gradient_norm: float, *, target_ratio: float = 0.1
) -> float:
    """Scale initial RDM gradients to a bounded fraction of prediction gradients."""
    if prediction_gradient_norm < 0 or rdm_gradient_norm <= 0 or target_ratio <= 0:
        raise ValueError("Gradient norms and calibration ratio must be positive.")
    return float(np.clip(target_ratio * prediction_gradient_norm / rdm_gradient_norm, 1e-3, 1e3))
