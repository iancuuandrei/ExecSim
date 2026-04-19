from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, time
from pathlib import Path

import yaml

from execsim.config import load_config, load_project_dotenv
from execsim.orders import ParentOrder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="execsim",
        description="Execution-cost simulator CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    show_config_parser = subparsers.add_parser(
        "show-config",
        help="Load the YAML config and print it.",
    )
    _add_config_argument(show_config_parser)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run a minimal smoke check without any market data.",
    )
    _add_config_argument(smoke_parser)

    download_parser = subparsers.add_parser(
        "download-data",
        help="Download, clean, validate, and save minute-bar data from Alpaca.",
    )
    _add_config_argument(download_parser)

    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Build a manifest CSV from processed per-symbol parquet files.",
    )
    _add_config_argument(manifest_parser)

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Run validation checks against processed per-symbol parquet files.",
    )
    _add_config_argument(validate_parser)

    simulate_twap_parser = subparsers.add_parser(
        "simulate-twap",
        help="Run a minimal TWAP simulation on processed minute bars.",
    )
    _add_config_argument(simulate_twap_parser)
    simulate_twap_parser.add_argument("--symbol", default=None, help="Symbol to simulate.")
    simulate_twap_parser.add_argument(
        "--trade-date",
        default=None,
        help="Trade date in YYYY-MM-DD format.",
    )
    simulate_twap_parser.add_argument(
        "--side",
        choices=("buy", "sell"),
        default=None,
        help="Parent order side.",
    )
    simulate_twap_parser.add_argument(
        "--quantity",
        type=int,
        default=None,
        help="Parent order share quantity.",
    )
    simulate_twap_parser.add_argument(
        "--start-time",
        default=None,
        help="Execution-window start time in HH:MM format, inclusive.",
    )
    simulate_twap_parser.add_argument(
        "--end-time",
        default=None,
        help="Execution-window end time in HH:MM format, exclusive.",
    )
    simulate_twap_parser.add_argument(
        "--max-bar-participation-rate",
        type=float,
        default=None,
        help="Per-bar cap as a fraction of observed bar volume.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_project_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "show-config":
        config = load_config(args.config)
        print(yaml.safe_dump(config.to_dict(), sort_keys=False).rstrip())
        return 0

    if args.command == "smoke":
        config = load_config(args.config)
        print(
            "smoke: ok "
            f"| project={config.project_name} "
            f"| timeframe={config.default_bar_timeframe} "
            f"| timezone={config.timezone}"
        )
        return 0

    if args.command == "download-data":
        from execsim.data.download import download_and_prepare_data

        config = load_config(args.config)
        result = download_and_prepare_data(config)
        for item in result.symbols:
            print(
                f"{item.symbol}: raw_rows={item.raw_rows} "
                f"processed_rows={item.processed_rows} "
                f"raw_path={item.raw_path} "
                f"processed_path={item.processed_path}"
            )
            for line in item.validation_report.to_lines()[2:]:
                print(line)
        print(f"manifest_path={result.manifest_path}")
        return 0

    if args.command == "build-manifest":
        from execsim.data.manifest import build_dataset_manifest

        config = load_config(args.config)
        manifest = build_dataset_manifest(config)
        print(
            f"manifest: rows={len(manifest)} "
            f"path={config.resolved_manifest_path}"
        )
        return 0

    if args.command == "validate-data":
        from execsim.data.loaders import load_processed_symbol_bars
        from execsim.data.validation import validate_processed_bars

        config = load_config(args.config)
        reports = [
            validate_processed_bars(load_processed_symbol_bars(config, symbol), symbol=symbol)
            for symbol in config.symbols
        ]
        for report in reports:
            for line in report.to_lines():
                print(line)
        return 0 if all(report.is_valid for report in reports) else 1

    if args.command == "simulate-twap":
        from execsim.data.loaders import load_processed_window_bars
        from execsim.simulator import simulate_twap

        config = load_config(args.config)
        try:
            order, max_participation_rate = _build_parent_order_from_args(args, config)
        except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        try:
            bars = load_processed_window_bars(
                config=config,
                symbol=order.symbol,
                trade_date=order.trade_date,
                start_time=order.start_time,
                end_time=order.end_time,
            )
            result = simulate_twap(
                parent_order=order,
                bars=bars,
                max_bar_participation_rate=max_participation_rate,
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_simulation_result(result)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML config file. Defaults to configs/base.yaml.",
    )


def _build_parent_order_from_args(
    args: argparse.Namespace,
    config,
) -> tuple[ParentOrder, float]:
    defaults = config.demo_twap
    order = ParentOrder(
        symbol=args.symbol or defaults.symbol,
        side=args.side or defaults.side,
        quantity=args.quantity if args.quantity is not None else defaults.quantity,
        trade_date=(
            _parse_date(args.trade_date)
            if args.trade_date is not None
            else defaults.trade_date
        ),
        start_time=(
            _parse_time(args.start_time)
            if args.start_time is not None
            else defaults.start_time
        ),
        end_time=(
            _parse_time(args.end_time)
            if args.end_time is not None
            else defaults.end_time
        ),
    )
    max_participation_rate = (
        args.max_bar_participation_rate
        if args.max_bar_participation_rate is not None
        else defaults.max_bar_participation_rate
    )
    return order, max_participation_rate


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def _parse_time(value: str) -> time:
    try:
        parsed_time = time.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid time '{value}'. Expected HH:MM."
        ) from exc

    if parsed_time.tzinfo is not None:
        raise argparse.ArgumentTypeError("Execution times must be timezone-naive.")
    return parsed_time


def _print_simulation_result(result) -> None:
    summary = result.summary
    average_price = (
        f"{summary.average_fill_price:.4f}"
        if summary.average_fill_price is not None
        else "n/a"
    )
    print(
        "TWAP simulation: "
        f"{summary.symbol} {summary.side} requested_qty={summary.requested_qty}"
    )
    print(
        f"window={summary.start_timestamp} -> {summary.end_timestamp} "
        f"| bars={summary.n_bars_in_window}"
    )
    print(
        f"filled_qty={summary.filled_qty} "
        f"| unfilled_qty={summary.unfilled_qty} "
        f"| completion_rate={summary.completion_rate:.4f} "
        f"| realized_participation={summary.realized_participation:.6f} "
        f"| average_fill_price={average_price}"
    )
    print("execution_log_head:")
    print(result.execution_log.head().to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
