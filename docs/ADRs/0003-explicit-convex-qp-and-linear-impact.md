# ADR 0003: Use an explicit convex QP and linear participation impact

- Status: Accepted
- Date: 2026-09-04
- Owners: ExecSim maintainers

## Context

Optimal and adaptive policies need visible completion, capacity, temporary-impact, tracking, and inventory-risk terms. The implementation must expose feasibility and numerical diagnostics rather than hiding them behind a strategy-specific heuristic.

## Decision

Represent constrained execution as an explicit convex quadratic program solved by OSQP. Keep the analytical Almgren-Chriss schedule as a separate reference. Model temporary execution-price displacement as linear in participation, which produces a convex quadratic total temporary-impact cost. Apply deterministic integer projection after the continuous solve.

## Rationale

The formulation is auditable, supports hard per-bucket capacities, and yields solver residuals and status. Linear-in-participation displacement is transparent and consistent between planning and realized cost accounting.

## Consequences

Temporary-impact coefficients must be strictly positive. OSQP remains an optional optimization dependency. Assumed parameters cannot be described as calibrated. Integer schedules preserve the feasible total and capacities but are a deterministic projection of the continuous optimum.

## Alternatives considered

- A closed-form schedule alone was rejected because it cannot express the complete capacity-constrained problem.
- Nonlinear or permanent-impact models were deferred because V1 has no calibration evidence for them.

## Verification

`docs/MATHEMATICAL_MODEL.md`, `tests/test_optimization.py`, and `tests/test_cost_models.py` specify and test the formulas and invariants.
