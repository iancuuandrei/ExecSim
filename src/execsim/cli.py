from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, time
from pathlib import Path
from typing import Any, cast

import yaml

from execsim.config import ExecSimConfig, load_config, load_project_dotenv
from execsim.orders import OrderSide, ParentOrder

STRATEGIES = ("twap", "vwap", "pov", "almgren-chriss", "optimal", "mpc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="execsim",
        description="Optimal-execution research CLI.",
        epilog=(
            "Deployable policies use only point-in-time observations and forecasts: "
            "twap, vwap, pov, almgren-chriss, optimal, and mpc. Oracle behavior is "
            "evaluation-only and not exposed by simulate. No trained ML artifact is included; "
            "prepare manifests and inspect a future run with `execsim ml training-plan`, then "
            "launch an authorized fit with `execsim ml train` without --dry-run."
        ),
    )
    commands = parser.add_subparsers(dest="command")
    for name, description in (
        ("show-config", "Load the project configuration."),
        ("smoke", "Run an installation check."),
        ("download-data", "Download and validate Alpaca minute bars."),
        ("build-manifest", "Build the processed-data manifest."),
        ("validate-data", "Validate processed minute bars."),
    ):
        command = commands.add_parser(name, help=description)
        _add_config_argument(command)

    simulate = commands.add_parser("simulate", help="Simulate one execution policy.")
    _add_simulation_arguments(simulate, include_strategy=True)
    legacy = commands.add_parser("simulate-twap", help="Alias for simulate --strategy twap.")
    _add_simulation_arguments(legacy, include_strategy=False)

    experiment = commands.add_parser("experiment", help="Run or inspect an experiment.")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    run = experiment_commands.add_parser("run", help="Run a YAML experiment grid.")
    run.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    report = experiment_commands.add_parser("report", help="Locate an existing report.")
    report.add_argument("--run-id", required=True)
    report.add_argument("--reports-root", type=Path, default=Path("reports/runs"))

    scenario = commands.add_parser("scenario", help="Generate deterministic synthetic bars.")
    scenario.add_argument("--volume-shape", default="u_shaped")
    scenario.add_argument("--price-path", default="constant")
    scenario.add_argument("--seed", type=int, default=17)
    scenario.add_argument("--output", type=Path, required=True)

    ml = commands.add_parser("ml", help="Operate point-in-time ML datasets and training plans.")
    ml_commands = ml.add_subparsers(dest="ml_command", required=True)
    build = ml_commands.add_parser("build-dataset")
    _add_config_argument(build)
    build.add_argument("--mode", choices=("static", "dynamic"), default="static")
    build.add_argument("--bucket-minutes", type=int, choices=(1, 5, 15), default=5)
    build.add_argument("--output-root", type=Path, default=Path("data/ml"))
    build.add_argument("--source", type=Path, action="append", default=[])
    build.add_argument("--allow-incomplete-sessions", action="store_true")
    for command_name in ("validate-dataset", "inspect-dataset"):
        command = ml_commands.add_parser(command_name)
        command.add_argument("--manifest", type=Path, required=True)
    split = ml_commands.add_parser("create-splits")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--initial-train-sessions", type=int, default=252)
    split.add_argument("--validation-sessions", type=int, default=21)
    split.add_argument("--test-sessions", type=int, default=21)
    split.add_argument("--step-sessions", type=int, default=21)
    split.add_argument("--embargo-sessions", type=int, default=0)
    for command_name in ("training-plan", "train"):
        command = ml_commands.add_parser(command_name)
        command.add_argument("--config", type=Path, default=Path("configs/ml.yaml"))
        command.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return _dispatch(args)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "show-config":
        print(yaml.safe_dump(load_config(args.config).to_dict(), sort_keys=False).rstrip())
        return 0
    if args.command == "smoke":
        config = load_config(args.config)
        print(f"smoke: ok | project={config.project_name} | timezone={config.timezone}")
        return 0
    if args.command == "download-data":
        from execsim.data.download import download_and_prepare_data

        result = download_and_prepare_data(load_config(args.config))
        for item in result.symbols:
            print(f"{item.symbol}: raw_rows={item.raw_rows} processed_rows={item.processed_rows}")
        print(f"manifest_path={result.manifest_path}")
        return 0
    if args.command == "build-manifest":
        from execsim.data.manifest import build_dataset_manifest

        config = load_config(args.config)
        manifest = build_dataset_manifest(config)
        print(f"manifest: rows={len(manifest)} path={config.resolved_manifest_path}")
        return 0
    if args.command == "validate-data":
        return _validate_market_data(load_config(args.config))
    if args.command in {"simulate", "simulate-twap"}:
        forced = "twap" if args.command == "simulate-twap" else None
        return _simulate(args, forced_strategy=forced)
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "scenario":
        return _scenario(args)
    if args.command == "ml":
        return _ml(args)
    raise ValueError(f"Unknown command: {args.command}")


def _simulate(args: argparse.Namespace, forced_strategy: str | None) -> int:
    from execsim.costs import CostParameter, LinearTemporaryImpactModel, ParameterProvenance
    from execsim.data.loaders import load_processed_symbol_bars, slice_processed_symbol_bars
    from execsim.forecasting import HistoricalProfileForecaster
    from execsim.policies import ExecutionConstraints, create_policy
    from execsim.simulator import simulate_policy

    config = load_config(args.config)
    order, participation = _build_parent_order_from_args(args, config)
    strategy = forced_strategy or args.strategy
    all_bars = load_processed_symbol_bars(config, order.symbol)
    day_bars = slice_processed_symbol_bars(all_bars, order.symbol, order.trade_date)
    provider = None
    if strategy in {"vwap", "optimal", "mpc"}:
        provider = HistoricalProfileForecaster(all_bars, lookback_sessions=args.lookback_sessions)
    result = simulate_policy(
        parent_order=order,
        bars=day_bars,
        policy=create_policy(
            strategy,
            risk_aversion=args.risk_aversion,
            half_spread=args.half_spread,
            temporary_impact=args.temporary_impact,
            volatility=args.volatility,
            pov_target_rate=args.pov_target_rate,
        ),
        constraints=ExecutionConstraints(participation, participation, config.timezone),
        cost_model=LinearTemporaryImpactModel(
            CostParameter(args.half_spread, ParameterProvenance.ASSUMED, "CLI assumption"),
            CostParameter(args.temporary_impact, ParameterProvenance.ASSUMED, "CLI assumption"),
        ),
        forecast_provider=provider,
    )
    _print_simulation_result(result, as_json=args.json)
    return 0


def _experiment(args: argparse.Namespace) -> int:
    if args.experiment_command == "report":
        report = args.reports_root / args.run_id / "REPORT.md"
        if not report.exists():
            raise FileNotFoundError(f"Experiment report not found: {report}")
        print(report.resolve())
        return 0
    from execsim.data.loaders import load_processed_symbol_bars
    from execsim.experiments import ExperimentRunner, ExperimentSpec

    payload = _read_yaml_mapping(args.config)
    project = load_config(payload.get("project_config"))
    values = dict(payload.get("experiment", payload))
    spec = ExperimentSpec(
        symbols=tuple(str(value).upper() for value in values["symbols"]),
        trade_dates=tuple(_parse_date(str(value)) for value in values["trade_dates"]),
        quantities=tuple(int(value) for value in values["quantities"]),
        sides=tuple(values.get("sides", ["buy"])),
        strategies=tuple(values.get("strategies", STRATEGIES)),
        start_time=_parse_time(str(values.get("start_time", "10:00"))),
        end_time=_parse_time(str(values.get("end_time", "11:00"))),
        planned_participation_rate=float(values.get("planned_participation_rate", 0.1)),
        hard_participation_rate=float(values.get("hard_participation_rate", 0.1)),
        pov_target_rate=float(values.get("pov_target_rate", 0.05)),
        half_spread=float(values.get("half_spread", 0.01)),
        temporary_impacts=tuple(float(value) for value in values.get("temporary_impacts", [0.1])),
        volatility=float(values.get("volatility", 0.01)),
        risk_aversions=tuple(float(value) for value in values.get("risk_aversions", [0.0, 0.01])),
        profile_estimator=str(values.get("profile_estimator", "mean")),
        profile_lookback_sessions=int(values.get("profile_lookback_sessions", 20)),
        seed=int(values.get("seed", 17)),
        include_oracle=bool(values.get("include_oracle", False)),
    )
    bars = {symbol: load_processed_symbol_bars(project, symbol) for symbol in spec.symbols}
    outputs = ExperimentRunner(spec, bars, Path(project.reports_dir) / "runs").run()
    print(
        json.dumps(
            {
                "run_id": outputs.run_id,
                "output_dir": str(outputs.output_dir),
                "runs": len(outputs.results),
            },
            indent=2,
        )
    )
    return 0


def _scenario(args: argparse.Namespace) -> int:
    from execsim.data.scenarios import ScenarioConfig, generate_scenario

    bars = generate_scenario(
        ScenarioConfig(
            volume_scenario=args.volume_shape,
            price_scenario=args.price_path,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(bars), "seed": args.seed}, indent=2))
    return 0


def _ml(args: argparse.Namespace) -> int:
    from execsim.ml.datasets import (
        DatasetBuildConfig,
        WalkForwardConfig,
        build_dataset,
        create_walk_forward_splits,
        load_dataset_manifest,
    )
    from execsim.ml.datasets.validation import load_dataset_rows, validate_dataset_rows

    if args.ml_command == "build-dataset":
        config = load_config(args.config)
        sources = args.source or [config.processed_symbol_path(symbol) for symbol in config.symbols]
        result = build_dataset(
            output_root=args.output_root,
            source_paths=sources,
            materialize_result_rows=False,
            config=DatasetBuildConfig(
                mode=args.mode,
                bucket_minutes=args.bucket_minutes,
                timezone=config.timezone,
                require_calendar_complete=not args.allow_incomplete_sessions,
            ),
        )
        print(
            json.dumps(
                {
                    "dataset_id": result.manifest.dataset_id,
                    "manifest": str(result.manifest_path),
                    "rows": result.manifest.row_count,
                },
                indent=2,
            )
        )
        return 0
    if args.ml_command in {"validate-dataset", "inspect-dataset", "create-splits"}:
        manifest = load_dataset_manifest(args.manifest)
        rows = load_dataset_rows(manifest, args.manifest.parent)
        if args.ml_command == "validate-dataset":
            errors = validate_dataset_rows(rows, manifest)
            print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
            return 1 if errors else 0
        if args.ml_command == "inspect-dataset":
            print(
                json.dumps(
                    {**manifest.stable_payload(), "manifest_sha256": manifest.manifest_hash()},
                    indent=2,
                    default=str,
                )
            )
            return 0
        split = create_walk_forward_splits(
            rows,
            dataset_id=manifest.dataset_id,
            config=WalkForwardConfig(
                args.initial_train_sessions,
                args.validation_sessions,
                args.test_sessions,
                args.step_sessions,
                args.embargo_sessions,
            ),
        )
        split.write(args.output)
        print(
            json.dumps(
                {"split_id": split.split_id, "folds": len(split.folds), "output": str(args.output)},
                indent=2,
            )
        )
        return 0
    from execsim.ml.training import build_training_plan, run_training

    training_config = _training_config(args.config)
    training_result = (
        build_training_plan(training_config)
        if args.ml_command == "training-plan"
        else run_training(training_config, dry_run=args.dry_run)
    )
    print(json.dumps(_to_dict(training_result), indent=2, default=str))
    return 0


def _training_config(path: Path) -> Any:
    from execsim.ml.training import TrainingConfig

    values = _read_yaml_mapping(path)
    return TrainingConfig(
        dataset_manifest_path=Path(values["dataset_manifest_path"]),
        split_manifest_path=Path(values["split_manifest_path"]),
        feature_names=tuple(values["feature_names"]),
        target_name=str(values["target_name"]),
        model_family=str(values.get("model_family", "ridge")),
        hyperparameter_grid=tuple(values.get("hyperparameter_grid", [{"alpha": 1.0}])),
        artifact_root=Path(values.get("artifact_root", "artifacts/models")),
        random_seed=int(values.get("random_seed", 17)),
        allow_historical_training=bool(values.get("allow_historical_training", False)),
    )


def _validate_market_data(config: ExecSimConfig) -> int:
    from execsim.data.loaders import load_processed_symbol_bars
    from execsim.data.validation import validate_processed_bars

    reports = [
        validate_processed_bars(load_processed_symbol_bars(config, symbol), symbol=symbol)
        for symbol in config.symbols
    ]
    for report in reports:
        print("\n".join(report.to_lines()))
    return 0 if all(report.is_valid for report in reports) else 1


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)


def _add_simulation_arguments(parser: argparse.ArgumentParser, *, include_strategy: bool) -> None:
    _add_config_argument(parser)
    if include_strategy:
        parser.add_argument("--strategy", choices=STRATEGIES, default="twap")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--side", choices=("buy", "sell"), default=None)
    parser.add_argument("--quantity", type=int, default=None)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--max-bar-participation-rate", type=float, default=None)
    parser.add_argument("--pov-target-rate", type=float, default=0.05)
    parser.add_argument("--half-spread", type=float, default=0.01)
    parser.add_argument("--temporary-impact", type=float, default=0.1)
    parser.add_argument("--volatility", type=float, default=0.01)
    parser.add_argument("--risk-aversion", type=float, default=0.0)
    parser.add_argument("--lookback-sessions", type=int, default=20)
    parser.add_argument("--json", action="store_true")


def _build_parent_order_from_args(
    args: argparse.Namespace, config: ExecSimConfig
) -> tuple[ParentOrder, float]:
    defaults = config.demo_twap
    order = ParentOrder(
        symbol=args.symbol or defaults.symbol,
        side=cast(OrderSide, args.side or defaults.side),
        quantity=args.quantity if args.quantity is not None else defaults.quantity,
        trade_date=_parse_date(args.trade_date) if args.trade_date else defaults.trade_date,
        start_time=_parse_time(args.start_time) if args.start_time else defaults.start_time,
        end_time=_parse_time(args.end_time) if args.end_time else defaults.end_time,
    )
    rate = (
        args.max_bar_participation_rate
        if args.max_bar_participation_rate is not None
        else defaults.max_bar_participation_rate
    )
    return order, rate


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def _parse_time(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid time {value!r}; expected HH:MM.") from exc
    if parsed.tzinfo is not None:
        raise ValueError("Execution times must be timezone-naive.")
    return parsed


def _print_simulation_result(result: Any, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "summary": asdict(result.summary),
                    "execution_log_head": result.execution_log.head().to_dict(orient="records"),
                },
                indent=2,
                default=str,
            )
        )
        return
    summary = result.summary
    print(
        f"{summary.strategy}: {summary.symbol} {summary.side} "
        f"requested={summary.requested_qty} filled={summary.filled_qty} "
        f"completion={summary.completion_rate:.4f}"
    )
    print(
        f"implementation_shortfall_bps={summary.implementation_shortfall_bps} "
        f"total_modeled_cost={summary.total_modeled_execution_cost:.6f}"
    )


def _read_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"YAML must contain a mapping: {resolved}")
    return payload


def _to_dict(value: object) -> dict[str, object]:
    return asdict(cast(Any, value)) if is_dataclass(value) else {"result": str(value)}


if __name__ == "__main__":
    raise SystemExit(main())
