# Implementation Log

## Current Iteration

Current iteration = `Iteration 1`

## What Exists Now

- Python project bootstrapped with `pyproject.toml`
- `src/` layout with a minimal `execsim` package
- CLI commands: `show-config`, `smoke`, `download-data`, `build-manifest`, and `validate-data`
- YAML config loading from `configs/base.yaml`
- `execsim.data` package for schema, download, cleaning, validation, loaders, and manifest generation
- Alpaca environment-variable credential handling via `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`
- Config-driven multi-symbol minute-bar download parameters
- Raw per-symbol Parquet storage plus cleaned processed per-symbol Parquet outputs
- Processed-data validation and manifest CSV generation
- Smoke, config, cleaning, and validation tests
- Durable project context and implementation log documents
- Placeholder directories for future data, notebook, and report artifacts

## What Does Not Exist Yet

- No simulator engine
- No execution strategies
- No cost model
- No experiment runner
- No plotting, dashboard, or notebook analysis workflow
- No broker order submission or live trading integration
- No notebook-based research analysis beyond the packaged dataset pipeline

## Next Planned Iteration

`Iteration 2` should define the execution inputs and baseline scheduling objects:

- Parent-order representation with side, quantity, start time, and end time
- Simple schedule outputs for TWAP, VWAP-profile, and POV baselines
- Clear interfaces between cleaned market data, order definitions, and later simulator logic
- Preserve the lightweight, research-oriented code structure

## Changelog

- `2026-04-15`: Created Iteration 0 bootstrap with package scaffold, CLI, YAML config loader, tests, README, and durable docs.
- `2026-04-15`: Added Iteration 1 data ingestion, cleaning, validation, processed Parquet output, and dataset manifest generation for Alpaca minute bars.

## Maintenance Rule

Every future Codex iteration that changes repository scope, code, configuration, or workflow assumptions must update this file before the task is considered complete.
