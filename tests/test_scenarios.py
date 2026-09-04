from __future__ import annotations

import pandas as pd

from execsim.data.scenarios import ScenarioConfig, generate_scenario


def test_scenarios_are_deterministic_and_canonical() -> None:
    config = ScenarioConfig(n_buckets=30, noise_fraction=0.1, seed=42)
    first = generate_scenario(config)
    second = generate_scenario(config)
    pd.testing.assert_frame_equal(first, second)
    assert {"symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"} <= set(
        first.columns
    )
    assert first["timestamp"].dt.tz is not None


def test_volume_scenarios_have_expected_qualitative_shapes() -> None:
    front = generate_scenario(ScenarioConfig(n_buckets=60, volume_scenario="front_loaded"))
    back = generate_scenario(ScenarioConfig(n_buckets=60, volume_scenario="back_loaded"))
    drought = generate_scenario(ScenarioConfig(n_buckets=60, volume_scenario="midday_drought"))
    assert front["volume"].iloc[:10].mean() > front["volume"].iloc[-10:].mean()
    assert back["volume"].iloc[:10].mean() < back["volume"].iloc[-10:].mean()
    assert drought["volume"].iloc[25:35].mean() < drought["volume"].iloc[:10].mean()


def test_adverse_trends_are_side_aware() -> None:
    buy = generate_scenario(
        ScenarioConfig(n_buckets=10, side="buy", price_scenario="adverse_trend")
    )
    sell = generate_scenario(
        ScenarioConfig(n_buckets=10, side="sell", price_scenario="adverse_trend")
    )
    assert buy["close"].iloc[-1] > buy["close"].iloc[0]
    assert sell["close"].iloc[-1] < sell["close"].iloc[0]
