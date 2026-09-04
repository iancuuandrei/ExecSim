from __future__ import annotations

import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict
from datetime import date
from datetime import time as wall_time
from pathlib import Path

import numpy as np
import pandas as pd

from execsim.data.scenarios import ScenarioConfig, generate_scenario
from execsim.experiments import ExperimentRunner, ExperimentSpec
from execsim.forecasting import HistoricalProfileForecaster
from execsim.ml.datasets import DatasetBuildConfig, build_dataset
from execsim.orders import ParentOrder
from execsim.policies import AdaptiveMPCPolicy, ExecutionConstraints, TwapPolicy, create_policy
from execsim.simulator import simulate_policy


def _measure(callable_, repetitions: int) -> dict[str, float | int]:
    durations: list[float] = []
    tracemalloc.start()
    for _ in range(repetitions):
        started = time.perf_counter()
        callable_()
        durations.append(time.perf_counter() - started)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "repetitions": repetitions,
        "median_seconds": statistics.median(durations),
        "minimum_seconds": min(durations),
        "peak_traced_bytes": peak,
    }


def main() -> int:
    history = [
        generate_scenario(ScenarioConfig(session_date=date(2026, 1, day), seed=day))
        for day in range(5, 10)
    ]
    bars = history[-1]
    all_bars = pd.concat(history, ignore_index=True)
    order = ParentOrder("SYNTH", "buy", 50_000, date(2026, 1, 9), wall_time(10), wall_time(11))
    constraints = ExecutionConstraints(0.1, 0.1)

    def static() -> object:
        return simulate_policy(
            parent_order=order,
            bars=bars,
            policy=TwapPolicy(),
            constraints=constraints,
        )

    def adaptive() -> object:
        return simulate_policy(
            parent_order=order,
            bars=bars,
            policy=AdaptiveMPCPolicy(),
            constraints=constraints,
            forecast_provider=HistoricalProfileForecaster(all_bars, lookback_sessions=4),
        )

    def comparison() -> None:
        for name in ("twap", "vwap", "pov", "optimal", "mpc"):
            simulate_policy(
                parent_order=order,
                bars=bars,
                policy=create_policy(name),
                constraints=constraints,
                forecast_provider=(
                    HistoricalProfileForecaster(all_bars, lookback_sessions=4)
                    if name in {"vwap", "optimal", "mpc"}
                    else None
                ),
            )

    experiment = ExperimentRunner(
        ExperimentSpec(
            symbols=("SYNTH",),
            trade_dates=(date(2026, 1, 8), date(2026, 1, 9)),
            quantities=(10_000,),
            strategies=("twap", "vwap", "pov", "optimal", "mpc"),
            profile_lookback_sessions=3,
            risk_aversions=(0.0,),
        ),
        {"SYNTH": all_bars},
    )

    with tempfile.TemporaryDirectory(prefix="execsim-benchmark-") as temporary:
        source = Path(temporary) / "bars.parquet"
        all_bars.to_parquet(source, index=False)

        def dataset_build() -> object:
            return build_dataset(
                output_root=Path(temporary) / "datasets",
                source_paths=(source,),
                config=DatasetBuildConfig(
                    bucket_minutes=5,
                    require_calendar_complete=False,
                    data_classification="synthetic_fixture",
                ),
            )

        ml_result = _measure(dataset_build, 3)

    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scenario": asdict(ScenarioConfig()),
        "static_simulation": _measure(static, 20),
        "adaptive_mpc_simulation": _measure(adaptive, 3),
        "repeated_strategy_comparison": _measure(comparison, 2),
        "multi_day_experiment": _measure(lambda: experiment.run(write_outputs=False), 2),
        "ml_parquet_scan_and_build": ml_result,
        "claim_boundary": "local synthetic throughput; not production capacity",
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
