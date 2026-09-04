# ADR 0002: Enforce a point-in-time policy boundary

- Status: Accepted
- Date: 2026-09-04
- Owners: ExecSim maintainers

## Context

Historical replay contains the complete realized session, but a deployable decision could not have observed future buckets. Passing unrestricted target-session bars to policies would make leakage easy and research comparisons unreliable. Minute bars also cannot identify the market's counterfactual response to simulated executions.

## Decision

Give policies a `DecisionContext` containing only observations strictly earlier than the decision, declared future timestamps, remaining inventory, constraints, and a point-in-time forecast. Keep the replayed market path exogenous. Mark hindsight providers as `EVALUATION_ONLY`.

## Rationale

The boundary makes causality structural and testable. Exogenous replay avoids inventing market-response dynamics unsupported by minute-bar data.

## Consequences

Forecasts must declare their generation time, first forecast bucket, training cutoff, schema, and provenance. Adaptive policies may use elapsed observations but not future target-session volume or prices. The simulator does not claim counterfactual price impact realism.

## Alternatives considered

- Passing the entire session DataFrame was rejected because conventions alone do not prevent leakage.
- Endogenous market response was deferred because the available data cannot identify it.

## Verification

`docs/DATA_LEAKAGE_CONTRACT.md`, `tests/test_forecasts.py`, and `tests/test_policies.py` define and test the boundary.
