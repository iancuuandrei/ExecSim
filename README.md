# execution-cost-sim

`execution-cost-sim` is a student quantitative-finance research repository for studying stock execution quality under a simplified intraday transaction-cost framework. The project is intended to compare benchmark execution styles such as TWAP, VWAP-profile, and POV for large parent orders, with a focus on execution costs and market-impact intuition rather than alpha generation or live trading.

## Current Status

This repository is at `Iteration 2`: a minimal end-to-end TWAP simulator on top of processed minute bars.

- Minimal `src/` Python package
- Config-driven Alpaca minute-bar download pipeline
- Cleaning to `America/New_York` regular trading hours
- Processed per-symbol Parquet outputs plus dataset manifest CSV
- Parent order and single-day execution-window models
- Minimal TWAP scheduler with integer child quantities
- Simulation loop with per-bar participation caps and fill-price logging
- Smoke, config, cleaning, and validation tests
- Durable project context and implementation log documents

## Repo Structure

```text
.
|-- configs/
|   `-- base.yaml
|-- data/
|   |-- processed/
|   `-- raw/
|-- docs/
|   |-- IMPLEMENTATION_LOG.md
|   `-- PROJECT_CONTEXT.md
|-- notebooks/
|-- reports/
|-- src/
|   `-- execsim/
|       |-- __init__.py
|       |-- cli.py
|       |-- config.py
|       |-- orders.py
|       |-- data/
|       |   |-- cleaning.py
|       |   |-- download.py
|       |   |-- loaders.py
|       |   |-- manifest.py
|       |   |-- schema.py
|       |   `-- validation.py
|       |-- simulator/
|       |   |-- core.py
|       |   `-- models.py
|       `-- strategies/
|           |-- base.py
|           `-- twap.py
`-- tests/
    |-- test_simulator_basic.py
    |-- test_twap_schedule.py
    |-- test_data_cleaning.py
    |-- test_data_validation.py
    |-- test_config.py
    `-- test_smoke.py
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
python -m execsim.cli --help
python -m execsim.cli download-data
python -m execsim.cli validate-data
python -m execsim.cli build-manifest
python -m execsim.cli simulate-twap
pytest
```

The installed console script `execsim --help` is also available after editable install.

## Data Pipeline

The Iteration 1 pipeline uses Alpaca historical US stock bars with:

- provider: `alpaca`
- feed: `sip`
- adjustment: `raw`
- frequency: `1min`
- session filter: `09:30 <= time < 16:00` in `America/New_York`

Credentials must be present in the environment:

```bash
set APCA_API_KEY_ID=your_key
set APCA_API_SECRET_KEY=your_secret
```

Then run:

```bash
python -m execsim.cli download-data
python -m execsim.cli validate-data
python -m execsim.cli build-manifest
```

`download-data` downloads one symbol at a time from the config universe, saves raw Parquet under `data/raw/...`, writes cleaned processed Parquet under `data/processed/...`, validates the cleaned output, and refreshes the manifest CSV.

## TWAP Simulation

Iteration 2 adds one runnable simulator command:

```bash
python -m execsim.cli simulate-twap
```

By default, this uses the small `demo_twap` block in `configs/base.yaml`. You can override the parent order and cap from the CLI:

```bash
python -m execsim.cli simulate-twap --symbol AAPL --trade-date 2026-03-16 --side buy --quantity 5000 --start-time 10:00 --end-time 10:30 --max-bar-participation-rate 0.05
```

The simulator loads processed per-symbol Parquet data, slices one trade date and one intraday window, builds a fixed TWAP schedule, applies a per-bar participation cap, and prints a summary plus the first few execution-log rows.

## Roadmap

- Iteration 0: bootstrap repository, config system, CLI, tests, and project docs
- Iteration 1: package the Alpaca ingestion, cleaning, validation, and manifest workflow
- Iteration 2: add parent-order representation, TWAP scheduling, and a minimal processed-bar simulator
- Iteration 3: add transaction-cost accounting and comparison metrics
- Iteration 4: add experiment workflows, reporting outputs, and validation checks

## Disclaimer

This is an educational execution-cost simulator for offline research. It is not a live execution engine, not trading advice, and not a production brokerage or order-routing system.

