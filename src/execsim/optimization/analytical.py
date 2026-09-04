from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def almgren_chriss_continuous_schedule(
    quantity: int,
    n_buckets: int,
    *,
    temporary_quadratic_coefficient: float,
    volatility: float,
    risk_aversion: float,
) -> NDArray[np.float64]:
    """Constant-parameter discrete analytical schedule for the V1 objective."""

    if quantity <= 0 or n_buckets <= 0:
        raise ValueError("quantity and n_buckets must be positive.")
    if not math.isfinite(temporary_quadratic_coefficient) or (temporary_quadratic_coefficient <= 0):
        raise ValueError("temporary_quadratic_coefficient must be finite and positive.")
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be finite and non-negative.")
    if not math.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("risk_aversion must be finite and non-negative.")

    ratio = risk_aversion * volatility**2 / (2.0 * temporary_quadratic_coefficient)
    if ratio <= 1e-14:
        return np.full(n_buckets, quantity / n_buckets, dtype=float)

    kappa = math.acosh(1.0 + ratio)
    indices = np.arange(n_buckets + 1, dtype=float)
    # The scaled exponential form avoids overflow for large kappa * N.
    remaining_steps = n_buckets - indices
    denominator = -math.expm1(-2.0 * kappa * n_buckets)
    inventory = (
        quantity
        * np.exp(-kappa * indices)
        * (-np.expm1(-2.0 * kappa * remaining_steps))
        / denominator
    )
    inventory[-1] = 0.0
    return inventory[:-1] - inventory[1:]
