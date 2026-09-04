# ADR 0004: Restrict V1 ML to point-in-time input forecasts

- Status: Accepted
- Date: 2026-09-04
- Owners: ExecSim maintainers

## Context

ExecSim needs an ML-ready research path without turning model outputs into opaque trading decisions or making unsupported predictive claims from a small historical sample.

## Decision

Use ML only to forecast point-in-time optimizer inputs, initially volume. Keep trade selection, completion, risk, and participation constraints in the explicit optimizer. Disable real historical fitting unless a separate authorization enables it; synthetic fixture fits prove plumbing only.

## Rationale

This boundary preserves attribution and lets forecast quality be evaluated separately from optimization behavior. It also prevents an unrestricted learned policy from bypassing execution constraints.

## Consequences

Datasets and artifacts must carry feature availability, cutoffs, schema versions, checksums, and walk-forward split metadata. V1 cannot claim predictive quality or production readiness from synthetic tests.

## Alternatives considered

- End-to-end learned policies and reinforcement learning were rejected as outside V1 scope and interpretability requirements.
- Training on repository history by default was rejected because authorization and evidence are absent.

## Verification

`docs/ML_DESIGN.md`, `docs/DATA_LEAKAGE_CONTRACT.md`, and `tests/test_ml_infrastructure.py` define the boundary.
