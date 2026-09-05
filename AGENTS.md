# ExecSim agent operating contract

This repository is an offline, single-asset, minute-bar execution-research system. Its purpose is to make execution assumptions, point-in-time information, optimization constraints, and cost attribution inspectable. It is not a live trading, alpha, brokerage, order-book, web, or reinforcement-learning system.

## Authority order

1. The active user brief.
2. This file.
3. `docs/standards/implementation.md` for implementation directions and repository-wide code and documentation style.
4. `docs/ADRs/README.md` and accepted ADRs for the rationale and consequences of durable decisions.
5. `repo_manifest.yaml` and the nearest package documentation for ownership and navigation.
6. `docs/SPECIFICATIONS.md` and its linked focused documents for implementation contracts.
7. Tests and current code for executable evidence.

When sources disagree, do not silently choose the convenient interpretation. Reconcile the authority documents and executable behavior in the same change.

## Required workflow

- Before editing, run `python scripts/repo_context.py --path <target> --json` or select the relevant area with `--area`.
- Keep point-in-time boundaries explicit. Policies and features must not receive unrestricted future-session data.
- Keep planning, realized fills, forecasting, costs, and reporting separate.
- Every production module must be represented in `repo_manifest.yaml` and specified in `docs/SPECIFICATIONS.md` or a linked focused specification. Write code and documentation according to `docs/standards/implementation.md`, and update its direction record with every material direction change.
- Record a new ADR for every material architecture, dependency, mathematical, information-boundary, artifact-compatibility, or non-obvious performance decision. Add it to `docs/ADRs/README.md` and update `docs/standards/implementation.md` in the same change.
- Preserve decision history. Supersede an accepted ADR with a new record instead of rewriting its rationale, and link both records.
- Update tests and specifications with behavior changes. An ADR explains why; it does not replace an executable specification or test.
- Report validation as `PASS`, `FAIL`, `NOT RUN`, or `BLOCKED`. Never convert missing dependencies, unavailable data, or skipped external work into a pass.
- Do not delete, skip, weaken, or narrow tests to make a gate pass.
- Preserve deterministic ordering, seeds, hashes, integer reconciliation, and artifact provenance.
- Never train an ML model on the repository's real historical data unless the user separately authorizes that action. Tiny synthetic test fits are allowed.
- Treat paper data acquisition, historical representation or LightGBM fitting, and the full paper evaluation as three separate privileged operations. Each requires its matching configuration switch and command-line enable flag; a dry run or synthetic fixture never grants that authority.
- Preserve the paper boundary in `docs/PAPER_DESIGN.md`: learned components may emit only point-in-time `VolumeForecast` inputs and cannot select trades, relax constraints, consume actual future tokens at inference, or alter replay and cost mathematics.

## Standard checks

Use the repository environment once installed:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe scripts\repo_context.py --check
```

Run focused tests during implementation and the complete gates before completion. Validate CLI examples exactly as documented.

## Scope boundaries

Core code may depend on NumPy, pandas, PyArrow, YAML, and the chosen sparse QP solver. Reporting, exchange-calendar, and ML dependencies must remain optional groups and must not be imported by core simulation paths. Do not add Nx, Node, distributed infrastructure, or a service framework to coordinate this single Python package unless profiling or repository scale later demonstrates a concrete need.

Generated reports, ML datasets, and fitted artifacts are untracked by default. Small deterministic fixtures may be tracked when they materially support reproducibility.
