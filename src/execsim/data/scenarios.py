from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Literal

import numpy as np
import pandas as pd

VolumeScenario = Literal[
    "uniform",
    "u_shaped",
    "front_loaded",
    "back_loaded",
    "low_liquidity",
    "midday_drought",
    "opening_spike",
    "late_spike",
]
PriceScenario = Literal["constant", "favorable_trend", "adverse_trend", "volatility_shock"]


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    symbol: str = "SYNTH"
    session_date: date = date(2026, 1, 5)
    side: str = "buy"
    n_buckets: int = 390
    start_time: time = time(9, 30)
    timezone: str = "America/New_York"
    base_price: float = 100.0
    base_volume: int = 10_000
    volume_scenario: VolumeScenario = "u_shaped"
    price_scenario: PriceScenario = "constant"
    trend_return: float = 0.01
    shock_return: float = 0.02
    noise_fraction: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_buckets <= 0 or self.base_volume <= 0 or self.base_price <= 0:
            raise ValueError("Scenario buckets, volume, and price must be positive.")
        if self.side not in {"buy", "sell"}:
            raise ValueError("Scenario side must be 'buy' or 'sell'.")
        if self.noise_fraction < 0:
            raise ValueError("noise_fraction must be non-negative.")


def generate_scenario(config: ScenarioConfig) -> pd.DataFrame:
    """Generate deterministic canonical minute bars for controlled research."""
    rng = np.random.default_rng(config.seed)
    position = (np.arange(config.n_buckets, dtype=float) + 0.5) / config.n_buckets
    volume_shape = _volume_shape(position, config.volume_scenario)
    if config.noise_fraction:
        volume_shape *= np.maximum(
            0.05, 1.0 + rng.normal(0.0, config.noise_fraction, config.n_buckets)
        )
    volume = np.maximum(0, np.rint(config.base_volume * volume_shape)).astype(np.int64)
    price = _price_path(config, position, rng)
    local_start = pd.Timestamp.combine(config.session_date, config.start_time).tz_localize(
        config.timezone
    )
    timestamps = pd.date_range(local_start, periods=config.n_buckets, freq="min")
    local_range = np.full(config.n_buckets, config.base_price * 0.0005)
    if config.price_scenario == "volatility_shock":
        middle = config.n_buckets // 2
        radius = max(1, config.n_buckets // 12)
        local_range[max(0, middle - radius) : middle + radius] *= 5.0
    return pd.DataFrame(
        {
            "symbol": config.symbol.upper(),
            "timestamp": timestamps,
            "open": price,
            "high": price + local_range,
            "low": np.maximum(0.01, price - local_range),
            "close": price,
            "volume": volume,
            "trade_count": np.maximum(1, np.ceil(volume / 100).astype(np.int64)),
            "vwap": price,
            "scenario_volume": config.volume_scenario,
            "scenario_price": config.price_scenario,
            "scenario_seed": config.seed,
        }
    )


def _volume_shape(position: np.ndarray, scenario: VolumeScenario) -> np.ndarray:
    if scenario == "uniform":
        return np.ones_like(position)
    if scenario == "u_shaped":
        return 0.55 + 2.2 * (2.0 * position - 1.0) ** 2
    if scenario == "front_loaded":
        return 0.4 + 2.0 * (1.0 - position) ** 2
    if scenario == "back_loaded":
        return 0.4 + 2.0 * position**2
    if scenario == "low_liquidity":
        return np.full_like(position, 0.15)
    if scenario == "midday_drought":
        return 1.0 - 0.85 * np.exp(-(((position - 0.5) / 0.14) ** 2))
    if scenario == "opening_spike":
        return 0.65 + 4.0 * np.exp(-position / 0.06)
    if scenario == "late_spike":
        return 0.65 + 4.0 * np.exp(-(1.0 - position) / 0.06)
    raise ValueError(f"Unknown volume scenario: {scenario}")


def _price_path(
    config: ScenarioConfig, position: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    direction = 1.0 if config.side == "buy" else -1.0
    if config.price_scenario == "constant":
        returns = np.zeros(config.n_buckets)
    elif config.price_scenario == "adverse_trend":
        returns = direction * config.trend_return * position
    elif config.price_scenario == "favorable_trend":
        returns = -direction * config.trend_return * position
    elif config.price_scenario == "volatility_shock":
        innovations = rng.normal(
            0.0, config.shock_return / np.sqrt(config.n_buckets), config.n_buckets
        )
        middle = config.n_buckets // 2
        radius = max(1, config.n_buckets // 12)
        innovations[max(0, middle - radius) : middle + radius] *= 4.0
        returns = np.cumsum(innovations)
    else:
        raise ValueError(f"Unknown price scenario: {config.price_scenario}")
    return config.base_price * np.exp(returns)
