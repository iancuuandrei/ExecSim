# Implementation log

## V1 completion pass — 2026-09-04

The repository advanced from a TWAP-only simulator to the V1 quantitative research framework:

- Established `AGENTS.md`, deterministic manifest navigation, and the living implementation standard.
- Added point-in-time decision contexts and historical volume forecasts.
- Added TWAP, historical VWAP, POV, analytical Almgren–Chriss, constrained QP, adaptive MPC, and evaluation-only oracle policies.
- Added explicit OSQP matrices, feasibility handling, diagnostics, warm starts, and deterministic integer projection.
- Added side-aware half-spread and linear-in-participation impact with parameter provenance.
- Expanded execution logs, decision traces, TCA summaries, cost reconciliation, and capacity reporting.
- Added deterministic liquidity and price scenarios.
- Added experiment grids, stable run IDs, Parquet/CSV/JSON/Markdown artifacts, figures, bootstrap intervals, paired differences, regimes, and win rates.
- Added ML feature metadata, point-in-time datasets, calendar filters, checksummed manifests, walk-forward splits, model adapters, training plans, synthetic fitting, inference, and compatibility-checked artifacts.
- Added a hierarchical CLI, example configurations, Ruff, mypy, pytest, repository-contract checks, and a Python 3.11/3.13 CI matrix.

## Acceptance evidence

The completed acceptance checks are:

| Check | Result |
|---|---|
| Pre-change test baseline after dependency installation | `PASS` — 23 tests |
| Final local suite | `PASS` — 67 tests |
| Coverage report | `PASS`; critical optimization, policy, reporting, artifact, simulation, and partitioned ML paths have focused tests |
| Fresh-environment editable install and suite | `PASS` — 67 tests, mypy, and Ruff |
| Ruff lint and format | `PASS` |
| mypy | `PASS` — 62 source files |
| Repository manifest contract | `PASS` — 10 areas |
| Existing AAPL, MSFT, and NVDA sample validation | `PASS` — 8,190 rows per symbol |
| All six deployable policies on one historical order | `PASS` — each completed 5,000 shares |
| Historical multi-strategy experiment | `PASS` — run `run-ab075520bd15`, 384 units |
| Deterministic synthetic scenario | `PASS` — 390 rows |
| Historical point-in-time ML dataset | `PASS` — 4,680 rows, 60 samples, three symbols |
| Walk-forward split manifest | `PASS` — two folds for the deliberately small acceptance windows |
| Historical ML training dry run | `PASS` — emitted cutoff and data-sufficiency warnings without artifacts |
| Historical ML fitting | `NOT RUN` — prohibited by the controlling task |
| Synthetic fixture fitting | `PASS` — Ridge, Elastic Net, and histogram gradient boosting in tests only |

No `artifacts/` directory or fitted historical model was created. Generated experiment, scenario, and ML acceptance outputs are ignored research artifacts.

## Maintenance rule

Update this log when a milestone changes repository capability or evidence. Update `docs/standards/implementation.md` in the same change whenever a material engineering direction changes.
