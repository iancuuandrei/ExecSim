from __future__ import annotations

import argparse
import cProfile
import pstats
from datetime import date
from datetime import time as wall_time
from pathlib import Path

import pandas as pd

from execsim.data.scenarios import ScenarioConfig, generate_scenario
from execsim.experiments import ExperimentRunner, ExperimentSpec
from execsim.forecasting import HistoricalProfileForecaster
from execsim.orders import ParentOrder
from execsim.policies import AdaptiveMPCPolicy, ExecutionConstraints
from execsim.simulator import simulate_policy


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, ParentOrder, ExecutionConstraints]:
    history = [
        generate_scenario(ScenarioConfig(session_date=date(2026, 1, day), seed=day))
        for day in range(5, 10)
    ]
    bars = history[-1]
    all_bars = pd.concat(history, ignore_index=True)
    order = ParentOrder("SYNTH", "buy", 50_000, date(2026, 1, 9), wall_time(10), wall_time(11))
    return bars, all_bars, order, ExecutionConstraints(0.1, 0.1)


def _adaptive_mpc_workload() -> None:
    bars, all_bars, order, constraints = _fixture()
    simulate_policy(
        parent_order=order,
        bars=bars,
        policy=AdaptiveMPCPolicy(),
        constraints=constraints,
        forecast_provider=HistoricalProfileForecaster(all_bars, lookback_sessions=4),
    )


def _experiment_workload() -> None:
    _, all_bars, _, _ = _fixture()
    runner = ExperimentRunner(
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
    runner.run(write_outputs=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a deterministic synthetic ExecSim workload."
    )
    parser.add_argument("--workload", choices=("mpc", "experiment"), default="mpc")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional pstats output path. Generated profiles are not research artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workload = _adaptive_mpc_workload if args.workload == "mpc" else _experiment_workload
    profiler = cProfile.Profile()
    profiler.runcall(workload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.output)
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
