# ExecSim implementation specifications

This document defines implemented V1 behavior. The focused mathematical, point-in-time, ML, and writing contracts linked here are normative. Code and documentation changes must update the applicable contract in the same change.

## System boundary

ExecSim simulates one equity parent order on one local-market session using timezone-aware minute bars. It accepts policy decisions, enforces inventory and participation constraints, applies a cost model, and produces an execution log, decision trace, and summary. It does not submit orders, mutate source bars, predict returns, model an order book, or infer live performance.

## Public substitution boundaries

The framework exposes typed protocols:

| Protocol | Responsibility |
|---|---|
| `SchedulingPolicy` | Create one static integer schedule from a `ParentOrder` and `DecisionContext` |
| `AdaptiveExecutionPolicy` | Reset state and choose one current-bucket quantity per causal decision |
| `ExecutionCostModel` | Quote a side-aware execution price and cost attribution for one quantity |
| `ArtifactStore` | Save and compatibility-check a fitted model plus immutable metadata |
| `VolumeForecastProvider` | Generate a forecast using the declared point-in-time information set |
| `ForecastModel` | Fit and predict through a replaceable regression adapter |

Concrete registries reject unknown names. Oracle VWAP is `EVALUATION_ONLY` and requires explicit opt-in in experiments.

## Information and time contract

`DecisionContext.observations` contains timestamps strictly earlier than `current_timestamp`. Its forecast must be generated no later than the decision and cover exactly the declared future timestamps. Historical profiles use complete sessions whose dates precede the target session. A historical provider may cache its causally filtered session-by-bucket matrix only by symbol scope and target date; it derives each requested window from that matrix. Adaptive MPC receives only elapsed observations and a point-in-time forecast.

[The data leakage contract](DATA_LEAKAGE_CONTRACT.md) defines static and dynamic sample semantics, feature availability, chronological split ordering, embargo, and prohibited inputs.

## Order, data, and fill contract

A `ParentOrder` contains an uppercase symbol, `buy` or `sell` side, positive integer shares, a date, and a timezone-naive half-open execution window `[start_time, end_time)`. Simulation timestamps must be timezone-aware, unique, and ordered after normalization. Prices must be finite and positive; volume must be finite and non-negative.

The reference price is bar VWAP when finite and otherwise the OHLC mean. Each bucket uses:

```text
actual_capacity = floor(hard_participation_rate * actual_market_volume)
executed_quantity = min(planned_quantity, remaining_inventory, actual_capacity)
```

Zero-volume bars have zero capacity. Static schedule misses are not silently redistributed. POV observes current bucket volume as it materializes; its trace names that convention. MPC re-solves over the remaining horizon and may warm-start OSQP. `OptimalExecutionWorkspace` caches OSQP setups by exact horizon and updates objective and bound values when a horizon recurs. Warm starts are shifted to the new dimension and clipped to inventory and per-bucket capacity bounds.

## Policy contract

The policy registry supports these deployable policies:

| Name | Behavior |
|---|---|
| `twap` | Equal bucket weights with deterministic earliest-bucket integer remainder |
| `vwap` | Historical point-in-time volume-profile weights |
| `pov` | Floor of target participation times observable current bucket volume |
| `almgren-chriss` | Analytical constant-parameter risk-impact schedule |
| `optimal` | One constrained convex quadratic program over the full horizon |
| `mpc` | Constrained quadratic program re-solved at every decision |

Schedules use non-negative integers, preserve deterministic timestamp order, and reconcile to their feasible planned quantity. Constrained policies report forecast capacity shortfall rather than treating an infeasible order as complete.

## Mathematical and cost contract

[The mathematical model](MATHEMATICAL_MODEL.md) defines notation, units, QP matrices, risk and tracking terms, linear-in-participation temporary impact, half-spread, feasibility relaxation, OSQP acceptance, integer projection, benchmarks, and cost reconciliation. Assumed parameters remain labeled `assumed`; code does not call them calibrated.

## Result and research-output contract

The execution log records order identity, plan, actual capacity, executed quantity, inventory transition, forecast and decision IDs, reference and execution prices, and spread, impact, and timing attribution for every bucket. `SimulationSummary` reports requested, feasible, filled, and unfilled shares; completion plus overall, average-bucket, and maximum participation; benchmarks and side-aware slippage; modeled costs; capacity shortfall; and optimizer telemetry.

Experiment run IDs hash the canonical specification. A run reuses historical providers and compatible policy workspaces within that run. It writes Parquet results, CSV aggregates and paired differences, JSON config and provenance, a Markdown report, and figures. Statistics include sample size, dispersion, quantiles, seeded bootstrap intervals, paired differences versus TWAP, and win rates. Wall-clock solver timing is nondeterministic telemetry. Optimizer traces separate matrix construction, setup, numeric update, eigenvalue validation, solve, and integer projection timing.

## ML infrastructure contract

[The ML design](ML_DESIGN.md) specifies static and dynamic point-in-time rows, targets, feature metadata, 1/5/15-minute buckets, calendar filtering, partitioned Parquet, checksummed manifests, walk-forward folds, model adapters, training-only preprocessing, validation-only selection, locked tests, downstream TCA hooks, and compatibility-checked artifacts.

Historical training is disabled unless `allow_historical_training=true`; V1 acceptance does not grant that authorization. `--dry-run` resolves manifests, schemas, folds, model grids, artifacts, warnings, and evaluation intent without fitting. Tests may fit tiny `synthetic_fixture` datasets.

## Sparse predictive-representation paper contract

[The sparse predictive-representation paper specification](PAPER_DESIGN.md) defines the frozen Alpaca SIP universe, exact XNYS validation, corpus-wide 26-by-18 session tensors with a 13-feature encoder and five predictor-conditioning fields, matched dense-Gaussian and sparse-rectified-Gaussian JEPA, exact RepReLU, derived rectified RDMReg moments, bounded FP32 distribution diagnostics, streaming device-aware training, frozen capacity and observable probes, residual-scale and bounded long-form-shape LightGBM targets, compatibility-complete artifacts, locked folds, 15-minute committed execution, continuous realized-volume oracle regret, and matched fold-safe inference. These software paths are `IMPLEMENTED` and use the same APIs in the deterministic multi-session fixture and future historical run. Provider acquisition is `DATA NOT ACQUIRED`; historical fitting is `TRAINING NOT RUN`; empirical results are `EMPIRICAL RESULT NOT AVAILABLE`. Each remains separately authorized.

[The paper implementation report](PAPER_IMPLEMENTATION_REPORT.md) records the final local gate evidence, numerical fixtures, performance telemetry, and the explicit `NOT RUN` empirical boundary.

## Command contract

The CLI provides these command groups:

- data: `download-data`, `validate-data`, and `build-manifest`;
- policies: `simulate --strategy ...` and the `simulate-twap` compatibility alias;
- research: `experiment run`, `experiment report`, and `scenario`;
- ML: `build-dataset`, `validate-dataset`, `inspect-dataset`, `create-splits`, `training-plan`, `train --dry-run`, and the nested manifest-driven `paper` workflow, including `paper run`;
- diagnostics: `show-config` and `smoke`.

Commands return zero on success. Validation returns one for invalid data. Contract errors fail with a concise parser error. `--json` emits machine-readable simulation output.

## Determinism and compatibility

Stable ordering, explicit seeds, canonical JSON hashing, checksummed sources and models, exact integer reconciliation, and named numerical tolerances make substantive outputs reproducible. Build timestamps, Git commits, dependency versions, and timing telemetry are provenance, not model inputs. Artifact loading fails closed on checksum or feature schema, target schema, bucket size, or timezone mismatch.

## Verification contract

Run all local gates from the repository root:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe scripts/repo_context.py --check
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions repeats these checks on Python 3.11 and 3.13. Tests cover formulas, monotonicity, constraints, causal cutoffs, leakage rejection, diagnostics, integer invariants, buy/sell signs, partial fills, synthetic regimes, statistics, manifests, splits, preprocessing, artifacts, and CLI workflows.

## Known limitations

Minute-bar replay omits quotes, spread measurement, queue position, within-bar paths, permanent impact, market reaction, auctions, fees, multi-asset coupling, and multi-day orders. The historical sample is small. Cost and risk inputs are assumptions unless their provenance states otherwise. No real historical model fit or predictive performance claim is part of V1 acceptance.
