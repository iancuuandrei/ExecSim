from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from execsim.config import load_config, load_project_dotenv


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

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML config file. Defaults to configs/base.yaml.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
