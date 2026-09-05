"""Collapse and sparsity diagnostics for linked latent matrices."""

from __future__ import annotations

import numpy as np


def representation_diagnostics(
    latents: np.ndarray, *, zero_tolerance: float = 0.0
) -> dict[str, float]:
    """Measure finite values, rank, exact support, and Hoyer sparsity."""
    values = np.asarray(latents, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("Diagnostics require a non-empty [sample, latent] matrix.")
    finite = float(np.isfinite(values).all())
    safe = np.nan_to_num(values)
    zeros = np.abs(safe) <= zero_tolerance
    variance = np.var(safe, axis=0)
    singular = np.linalg.svd(safe - safe.mean(axis=0), compute_uv=False)
    energy = singular**2
    probabilities = energy / max(float(energy.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12))))
    dimension = safe.shape[1]
    l1 = np.linalg.norm(safe, ord=1, axis=1)
    l2 = np.linalg.norm(safe, ord=2, axis=1)
    hoyer = (np.sqrt(dimension) - l1 / np.maximum(l2, 1e-12)) / (np.sqrt(dimension) - 1)
    activation = (~zeros).mean(axis=0)
    support_probability = np.mean(~zeros, axis=0)
    active_count = (~zeros).sum(axis=1)
    support_entropy = -support_probability * np.log(support_probability + 1e-12) - (
        1 - support_probability
    ) * np.log(1 - support_probability + 1e-12)
    return {
        "finite": finite,
        "mean_variance": float(variance.mean()),
        "effective_rank": effective_rank,
        "zero_fraction": float(zeros.mean()),
        "active_dimension_fraction": float((activation > 0).mean()),
        "dead_dimension_fraction": float((activation == 0).mean()),
        "always_on_dimension_fraction": float((activation == 1).mean()),
        "mean_hoyer_sparsity": float(np.mean(hoyer)),
        "mean_support_entropy": float(np.mean(support_entropy)),
        "mean_active_dimensions": float(np.mean(active_count)),
        "median_active_dimensions": float(np.median(active_count)),
        "p95_active_dimensions": float(np.quantile(active_count, 0.95)),
        "activation_frequency_q05": float(np.quantile(activation, 0.05)),
        "activation_frequency_q50": float(np.quantile(activation, 0.50)),
        "activation_frequency_q95": float(np.quantile(activation, 0.95)),
    }


def support_transition_diagnostics(latents: np.ndarray, regimes: np.ndarray) -> dict[str, object]:
    """Measure consecutive support changes using independently supplied regimes."""
    values = np.asarray(latents, dtype=float)
    labels = np.asarray(regimes)
    if values.ndim != 2 or len(values) < 2 or labels.shape != (len(values),):
        raise ValueError("Support transitions require ordered latents and one regime per row.")
    support = values > 0
    marginal = support.mean(axis=0)
    chance_intersection = float(np.square(marginal).sum())
    chance_union = float((2 * marginal - np.square(marginal)).sum())
    intersections = np.logical_and(support[:-1], support[1:]).sum(axis=1)
    unions = np.logical_or(support[:-1], support[1:]).sum(axis=1)
    jaccard = np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=float),
        where=unions > 0,
    )
    transition = 1.0 - jaccard
    dimension_transitions = {
        "inactive_to_inactive": int((~support[:-1] & ~support[1:]).sum()),
        "inactive_to_active": int((~support[:-1] & support[1:]).sum()),
        "active_to_inactive": int((support[:-1] & ~support[1:]).sum()),
        "active_to_active": int((support[:-1] & support[1:]).sum()),
    }
    regime_jaccard: dict[str, float] = {}
    matrix: dict[str, dict[str, int]] = {}
    for index, _value in enumerate(jaccard):
        source = str(labels[index])
        target = str(labels[index + 1])
        matrix.setdefault(source, {})[target] = matrix.setdefault(source, {}).get(target, 0) + 1
        if source == target:
            regime_jaccard.setdefault(source, 0.0)
    for regime in tuple(regime_jaccard):
        selected = (labels[:-1] == regime) & (labels[1:] == regime)
        regime_jaccard[regime] = float(np.mean(jaccard[selected]))
    return {
        "mean_consecutive_support_jaccard": float(np.mean(jaccard)),
        "support_transition_rate": float(np.mean(transition)),
        "chance_support_jaccard": chance_intersection / max(chance_union, 1e-12),
        "support_state_transition_matrix": dimension_transitions,
        "per_regime_support_jaccard": regime_jaccard,
        "regime_transition_matrix": matrix,
    }


def predictive_identity_ratio(prediction_error: np.ndarray, identity_error: np.ndarray) -> float:
    """Compare predictive error with the frozen last-latent identity reference."""
    prediction = np.asarray(prediction_error, dtype=float)
    identity = np.asarray(identity_error, dtype=float)
    if prediction.shape != identity.shape or not len(prediction):
        raise ValueError("Identity comparison requires matching non-empty errors.")
    if not np.isfinite(prediction).all() or not np.isfinite(identity).all():
        raise ValueError("Identity comparison errors must be finite.")
    return float(np.mean(prediction) / max(float(np.mean(identity)), 1e-12))


def sparse_acceptance(
    diagnostics: dict[str, float], *, target_zero_fraction: float = 0.75
) -> tuple[str, ...]:
    """Return failed locked sparse-representation gates."""
    failures = []
    if abs(diagnostics["zero_fraction"] - target_zero_fraction) > 0.05:
        failures.append("zero fraction is more than five percentage points from target")
    if diagnostics["active_dimension_fraction"] < 0.8:
        failures.append("fewer than 80% of dimensions were ever active")
    if diagnostics["dead_dimension_fraction"] >= 0.2:
        failures.append("at least 20% of dimensions are dead")
    if diagnostics["finite"] != 1.0:
        failures.append("latents contain non-finite values")
    return tuple(failures)
