# Execution Cost Simulator Specifications

This document records the current implementation contracts for future contributors. It is subordinate to `docs/PROJECT_CONTEXT.md`, which remains the stable project contract, and should be updated whenever code behavior changes.

Current coverage: Iterations 0 through 2.

## System Purpose

`execution-cost-sim` is an offline, single-asset educational simulator for studying parent-order execution over intraday minute-bar data. The current system can download and normalize historical bars, validate processed datasets, and run a minimal TWAP simulation for one symbol on one trade date over one intraday window.

The project is not a live trading system, brokerage connector, order router, high-frequency market simulator, or alpha model.

## Package And Runtime

- Source layout: `src/execsim`
- Python: 3.11+
- Core dependencies: `pandas`, `pyarrow`, `PyYAML`, `python-dotenv`, `alpaca-py`
- Test dependency: `pytest`
- CLI entry point: `execsim = execsim.cli:main`

The repository expects local file paths to resolve relative to the project root unless an absolute path is supplied.

## Configuration Contract

Default config path: `configs/base.yaml`

Loaded by: `execsim.config.load_config`

Required top-level fields:

- `project_name`
- `symbols`
- `start_date`
- `end_date`
- `timezone`
- `data_provider`
- `alpaca_feed`
- `alpaca_adjustment`
- `data_root`
- `raw_data_dir`
- `processed_data_dir`
- `manifest_path`
- `reports_dir`
- `default_bar_timeframe`
- `log_level`

Current constraints:

- `start_date <= end_date`
- `default_bar_timeframe == "1min"`
- `data_provider == "alpaca"`
- `symbols` are normalized to uppercase
- relative paths are resolved against the project root

Optional demo simulation block:

```yaml
demo_twap:
  symbol: AAPL
  trade_date: 2026-03-16
  side: buy
  quantity: 5000
  start_time: "10:00"
  end_time: "10:30"
  max_bar_participation_rate: 0.05
```

`demo_twap` constraints:

- `symbol` must be included in `symbols`
- `trade_date` must be within the configured date range
- `side` must be `buy` or `sell`
- `quantity` must be a positive integer
- `start_time < end_time`
- `max_bar_participation_rate` must be between `0` and `1`

## Environment Contract

`execsim.config.load_project_dotenv` loads `.env` from the project root by default.

Alpaca downloads require:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`

The simulator itself does not require network access or Alpaca credentials once processed data exists locally.

## Data Schema

Canonical processed bar columns:

- `symbol`
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `trade_count`
- `vwap`

Required non-null columns for validation:

- `open`
- `high`
- `low`
- `close`
- `volume`

Current session assumptions:

- Bar timeframe is one minute.
- Processed timestamps are timezone-aware.
- Processed bars are regular-hours only.
- Regular-hours window is `09:30 <= time < 16:00`.
- A full regular-hours trading day has 390 bars.

## Data Pipeline Specification

The Iteration 1 pipeline downloads raw Alpaca minute bars, normalizes them into the canonical bar schema, filters regular trading hours, writes per-symbol raw and processed Parquet files, validates processed output, and builds a manifest CSV.

Current CLI commands:

- `download-data`: download, clean, validate, save, and refresh manifest
- `validate-data`: validate configured processed symbol files
- `build-manifest`: rebuild the processed-data manifest

Processed data paths are per symbol:

```text
data/processed/alpaca/minute_bars/<SYMBOL>.parquet
```

## Processed Data Loading

Primary loader helpers:

- `load_processed_symbol_bars(config, symbol)`
- `load_processed_symbol_day_bars(config, symbol, trade_date)`
- `load_processed_window_bars(config, symbol, trade_date, start_time, end_time)`
- `slice_processed_symbol_bars(bars, symbol, trade_date, start_time=None, end_time=None)`

Window slicing semantics:

- symbol comparison is uppercase
- date match uses `timestamp.dt.date == trade_date`
- start time is inclusive
- end time is exclusive
- output is sorted by timestamp and reset to a zero-based index

The simulator can also slice bars internally, but CLI use should load an already sliced processed window.

## Parent Order Specification

Model: `execsim.orders.ParentOrder`

Fields:

- `symbol`
- `side`
- `quantity`
- `trade_date`
- `start_time`
- `end_time`

Validation and normalization:

- `symbol` is stripped and uppercased
- `side` is stripped, lowercased, and must be `buy` or `sell`
- `quantity` must be a positive integer share quantity
- `trade_date` must be a `datetime.date`
- `start_time` and `end_time` must be timezone-naive `datetime.time` values
- `start_time < end_time`

This iteration supports only one symbol and one trading day per parent order.

## Execution Window Specification

Model: `execsim.orders.ExecutionWindow`

Fields:

- `trade_date`
- `start_time`
- `end_time`

An execution window is a single-day, local market-time interval. Multi-day orders are not supported.

## Strategy Interface

Protocol: `execsim.strategies.base.SchedulingStrategy`

Required method:

```python
generate_schedule(parent_order: ParentOrder, bars: pd.DataFrame) -> pd.DataFrame
```

Expected output:

- one row per input bar
- includes `timestamp`
- includes integer `scheduled_qty`

Strategy schedules are target quantities before simulator participation caps are applied.

## TWAP Strategy Specification

Implementation: `execsim.strategies.twap.TwapStrategy`

Inputs:

- `ParentOrder`
- processed and execution-window-sliced bars

Behavior:

- requires a non-empty bar DataFrame with `timestamp`
- distributes parent quantity evenly across all bars in the execution window
- uses integer share quantities
- allocates remainder shares to the earliest bars
- guarantees `sum(scheduled_qty) == parent_order.quantity` before caps

Example:

- quantity `10` over `4` bars produces `[3, 3, 2, 2]`
- quantity `2` over `5` bars produces `[1, 1, 0, 0, 0]`

No catch-up, redistribution, volume awareness, or adaptivity is included.

## Simulator Specification

Primary functions:

- `execsim.simulator.simulate_twap(parent_order, bars, max_bar_participation_rate)`
- `execsim.simulator.simulate_order(parent_order, bars, strategy, max_bar_participation_rate)`

Required simulation bar columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

If a `symbol` column is present, rows are filtered to the parent order symbol.

Window filter:

- `timestamp.dt.date == parent_order.trade_date`
- `parent_order.start_time <= timestamp.dt.time < parent_order.end_time`

Per-bar participation cap:

```text
max_allowed_qty = floor(max_bar_participation_rate * bar_volume)
```

Child fill quantity:

```text
filled_qty = min(scheduled_qty, remaining_parent_qty, max_allowed_qty)
```

Fill price:

- use bar `vwap` when present and non-null
- otherwise use `(open + high + low + close) / 4`

The simulator records one execution-log row per bar in the execution window, including bars with zero fill.

## Execution Log Specification

Execution log type: `pandas.DataFrame`

Columns:

- `symbol`
- `timestamp`
- `side`
- `scheduled_qty`
- `max_allowed_qty`
- `filled_qty`
- `bar_volume`
- `fill_price`

The log is ordered by bar timestamp.

## Simulation Summary Specification

Model: `execsim.simulator.models.SimulationSummary`

Fields:

- `symbol`
- `side`
- `requested_qty`
- `filled_qty`
- `unfilled_qty`
- `average_fill_price`
- `completion_rate`
- `realized_participation`
- `start_timestamp`
- `end_timestamp`
- `n_bars_in_window`

Definitions:

```text
average_fill_price = sum(filled_qty_i * fill_price_i) / sum(filled_qty_i)
completion_rate = filled_qty / requested_qty
realized_participation = filled_qty / sum(bar_volume over execution window)
```

If no shares are filled:

- `average_fill_price` is `None`
- `completion_rate` is `0.0`
- `realized_participation` is `0.0` when window volume is zero, otherwise still `0.0`

## CLI Specification

Available commands:

- `show-config`
- `smoke`
- `download-data`
- `build-manifest`
- `validate-data`
- `simulate-twap`

All commands accept:

```text
--config <path>
```

`simulate-twap` accepts:

- `--symbol`
- `--trade-date`
- `--side`
- `--quantity`
- `--start-time`
- `--end-time`
- `--max-bar-participation-rate`

If omitted, these values are read from `demo_twap` in the loaded config.

The command prints:

- short TWAP simulation summary
- first few execution-log rows

Example:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli simulate-twap --symbol AAPL --trade-date 2026-03-16 --side buy --quantity 5000 --start-time 10:00 --end-time 10:30 --max-bar-participation-rate 0.05
```

## Testing Specification

Current test categories:

- config loading and `.env` loading behavior
- bar cleaning behavior
- processed-data validation behavior
- smoke CLI behavior
- TWAP schedule behavior
- simulator cap, side, incomplete-fill, and weighted-average behavior

Expected full-suite command:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

New features should include focused tests at the same level of granularity before broad experiment or reporting code is added.

## Current Non-Goals

These are intentionally not implemented yet:

- POV strategy
- VWAP-profile strategy
- adaptive strategy behavior
- market-impact model
- spread model
- implementation-shortfall report
- transaction-cost accounting
- experiment runner
- plotting or dashboard system
- order-book or quote simulation
- live broker execution
- multi-asset or multi-day parent orders

## Extension Guidance

- Prefer dataclasses and simple functions.
- Keep strategy output separate from simulator fills.
- Do not hide assumptions in notebooks; encode reusable behavior in `src/execsim`.
- Use processed local data for simulation paths.
- Keep one new abstraction per real need.
- Update this specification, `docs/IMPLEMENTATION_LOG.md`, and README workflow notes when behavior changes.
