# ADR 0001: Use the Python research stack and manifest-backed navigation

- Status: Accepted
- Date: 2026-09-04
- Owners: ExecSim maintainers

## Context

ExecSim is one offline quantitative-research package. It needs numerical arrays, labeled time-series tables, columnar artifacts, a sparse optimizer, deterministic repository discovery, and commands that work locally and in continuous integration. Adding a JavaScript task graph would create a second package ecosystem without a measured coordination problem.

## Decision

Use Python 3.11 or later with NumPy, pandas, SciPy, PyArrow, and OSQP for the implemented V1 boundaries. Use `AGENTS.md`, `repo_manifest.yaml`, `docs/NAVIGATION.md`, and `scripts/repo_context.py` for ownership, discovery, and verification. Do not add Nx or Node unless repository scale and profiling establish a concrete need.

## Rationale

The selected libraries match the numerical, time-series, columnar, and convex-optimization workload. A checked manifest makes ownership and validation commands inspectable without duplicating Python packaging in another build system.

## Consequences

Core simulation must not import optional reporting or ML dependencies. Every production module must appear in the manifest and in a specification. The repository accepts a small custom navigation checker instead of Nx features such as a generalized task graph or affected-project calculation.

## Alternatives considered

- Nx was rejected because one Python package has no measured multi-project orchestration bottleneck.
- An undocumented directory convention was rejected because agents and reviewers need deterministic ownership and verification lookup.

## Verification

`tests/test_repository_contract.py` and `python scripts/repo_context.py --check` verify the navigation contract.
