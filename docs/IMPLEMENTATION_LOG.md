# Implementation Log

## Current Iteration

Current iteration = `Iteration 2`

## What Exists Now

- Python project bootstrapped with `pyproject.toml`
- `src/` layout with a minimal `execsim` package
- CLI commands: `show-config`, `smoke`, `download-data`, `build-manifest`, `validate-data`, and `simulate-twap`
- YAML config loading from `configs/base.yaml`
- `execsim.data` package for schema, download, cleaning, validation, loaders, and manifest generation
- Alpaca environment-variable credential handling via `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`
- Config-driven multi-symbol minute-bar download parameters
- Raw per-symbol Parquet storage plus cleaned processed per-symbol Parquet outputs
- Processed-data validation and manifest CSV generation
- Parent-order and single-day execution-window dataclasses
- Minimal TWAP scheduler that produces integer child quantities summing to the parent quantity before caps
- Processed-bar loader helpers for one symbol, one trade date, and one intraday execution window
- Basic simulator loop over processed 1-minute bars
- Per-bar max participation cap using `floor(rate * bar_volume)`
- Fill-price rule using bar `vwap` with OHLC average fallback
- Structured execution log with scheduled quantity, cap, fill quantity, bar volume, and fill price
- Simulation summary with fill completion, average fill price, and realized participation
- Smoke, config, cleaning, and validation tests
- Focused TWAP scheduling and simulator behavior tests
- Current implementation specifications in `docs/SPECIFICATIONS.md`
- Durable project context and implementation log documents
- Placeholder directories for future data, notebook, and report artifacts

## What Does Not Exist Yet

- No cost model
- No POV strategy
- No VWAP-profile strategy
- No adaptive strategy
- No market-impact overlay
- No implementation shortfall report
- No experiment runner
- No plotting, dashboard, or notebook analysis workflow
- No order-book or quote simulation
- No broker order submission or live trading integration
- No notebook-based research analysis beyond the packaged dataset pipeline

## Next Planned Iteration

`Iteration 3` should add simplified execution-quality and transaction-cost accounting:

- Arrival-price reference handling
- Implementation-shortfall calculations in currency units and basis points
- Slippage versus simple intraday references such as arrival price or session VWAP
- Keep the existing TWAP simulator small and preserve the processed-data pipeline

## Changelog

- `2026-04-15`: Created Iteration 0 bootstrap with package scaffold, CLI, YAML config loader, tests, README, and durable docs.
- `2026-04-15`: Added Iteration 1 data ingestion, cleaning, validation, processed Parquet output, and dataset manifest generation for Alpaca minute bars.
- `2026-04-19`: Added Iteration 2 parent-order model, TWAP scheduler, processed-window loader, basic simulator loop, execution log, summary object, `simulate-twap` CLI command, and focused tests.
- `2026-04-19`: Added `docs/SPECIFICATIONS.md` as a living implementation contract for the config, data pipeline, order model, TWAP scheduler, simulator, CLI, tests, and current non-goals.

## Maintenance Rule

Every future Codex iteration that changes repository scope, code, configuration, or workflow assumptions must update this file before the task is considered complete.
