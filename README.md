# ExecSim

ExecSim is an offline, single-asset quantitative research framework for comparing intraday parent-order execution policies. It provides causal volume forecasts, static and adaptive policies, a transparent spread-and-impact model, constrained optimization, transaction-cost analysis (TCA), reproducible experiments, and leakage-safe ML infrastructure.

ExecSim is educational research software. It is not a live trading system, broker, order router, alpha model, or claim of production readiness.

## Install the project

Create an isolated environment and install every V1 capability:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Verify the installation:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli smoke
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest -q
```

## Run a simulation

Run any deployable policy through one command:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli simulate --strategy optimal --symbol AAPL --trade-date 2026-03-23 --quantity 5000 --start-time 10:00 --end-time 11:00 --json
```

Supported policies are `twap`, `vwap`, `pov`, `almgren-chriss`, `optimal`, and `mpc`. The original `simulate-twap` command remains a compatibility alias.

## Run a reproducible experiment

Review `configs/experiment.yaml`, then run the configured grid:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli experiment run --config configs/experiment.yaml
```

Each stable run ID receives raw Parquet results, aggregate and paired CSV statistics, a configuration snapshot, provenance, figures, and a Markdown report under `reports/runs/`.

## Prepare ML research data

Build and validate point-in-time rows without fitting historical data:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli ml build-dataset --mode static --bucket-minutes 5 --output-root data/ml
.\.venv\Scripts\python.exe -m execsim.cli ml validate-dataset --manifest data/ml/DATASET_ID/manifest.json
```

Replace `DATASET_ID` with the ID printed by the build command. Historical model fitting is disabled by default; the test suite fits only tiny synthetic fixtures.

## Navigate the repository

Use the manifest-backed context selector to find ownership, specifications, and focused checks:

```powershell
.\.venv\Scripts\python.exe scripts/repo_context.py --list
.\.venv\Scripts\python.exe scripts/repo_context.py --path src/execsim/optimization/qp.py --json
```

Start with these documents:

- `docs/standards/implementation.md` defines code and documentation practice and records engineering directions.
- `docs/SPECIFICATIONS.md` is the normative component contract.
- `docs/MATHEMATICAL_MODEL.md` defines formulas, units, constraints, and signs.
- `docs/DATA_LEAKAGE_CONTRACT.md` defines the point-in-time boundary.
- `docs/ML_DESIGN.md` defines ML data, splits, training, and artifacts.
- `docs/NAVIGATION.md` maps repository areas.

## Understand the evidence boundary

Historical replay keeps market bars exogenous and applies assumed costs to simulated fills. Minute bars do not expose quotes, queue position, within-bar paths, or counterfactual market response. Reports describe only the selected bars and assumptions; they do not establish strategy superiority or expected live performance.
