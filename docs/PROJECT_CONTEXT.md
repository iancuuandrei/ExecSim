# Project Context

## Project Goal

`execution-cost-sim` is an educational research repository for studying the execution of large parent stock orders over an intraday window under a simplified market-impact and transaction-cost framework. The primary goal is to compare execution quality across benchmark schedule styles in a clean, reproducible, student-friendly codebase.

## What The Project Is Not

- Not a return-prediction or alpha-generation project
- Not a live trading or order-routing system
- Not a brokerage integration layer
- Not a high-frequency microstructure simulator
- Not a production risk, compliance, or execution stack

## V1 Scope

The intended first meaningful version should support a single-asset, offline intraday simulation workflow with:

- Local historical intraday bar data
- A parent order definition with side, size, and execution window
- Baseline strategies such as TWAP, VWAP-profile, and POV
- A simplified cost model for temporary and possibly permanent impact
- Execution-quality metrics for comparing strategy outputs
- Reproducible experiments driven by config files

## Postponed Scope

The following are intentionally out of scope until the bootstrap and baseline research loop are stable:

- Live market connectivity
- Broker APIs and real order submission
- Data download automation
- Multi-asset portfolio scheduling
- Queue-position modeling and limit-order-book simulation
- Sophisticated impact calibration
- Dashboards, web apps, and production deployment

## Intended Data Choice

The expected data source for early iterations is intraday equity bar data, likely at the one-minute level, normalized into a local project-owned format. A practical initial choice is a vendor such as Alpaca or Polygon for minute bars, but the repository should remain vendor-agnostic once data is stored locally. Volume data from these bars will serve as the simplified market activity proxy for schedule construction and participation measurement.

## Intended Strategies

- TWAP: spread quantity evenly across the execution window
- VWAP-profile: follow a target intraday volume curve or session volume profile
- POV: trade as a fraction of observed market volume

## Intended Metrics

- Implementation shortfall in currency units and basis points
- Slippage versus arrival price
- Slippage versus session VWAP
- Fill completion rate by end of window
- Realized participation rate
- Execution-price path summary statistics

## Core Mathematical Definitions

### Arrival Price

The arrival price is the market reference price observed when the parent order decision is made or when the execution window begins. If the order start time is `t0`, then the arrival price is denoted `P_arr = P(t0)`.

### Average Execution Price

If child fills occur at prices `p_i` and quantities `q_i`, then the quantity-weighted average execution price is:

`P_exec_avg = (sum_i q_i p_i) / (sum_i q_i)`

### Implementation Shortfall

For a buy order with total executed quantity `Q`, one simple implementation shortfall measure is:

`IS = Q * (P_exec_avg - P_arr)`

In basis points for a buy order:

`IS_bps = 10^4 * (P_exec_avg - P_arr) / P_arr`

For sells, the sign convention should be flipped so worse execution still corresponds to higher cost.

### Session VWAP

If bar prices are represented by `P_t` and corresponding traded market volumes are `V_t`, then session VWAP is:

`VWAP_session = (sum_t V_t P_t) / (sum_t V_t)`

### Realized Participation

If the strategy executes quantity `Q_exec` over a window in which observed market volume is `V_mkt`, then realized participation is:

`Participation_realized = Q_exec / V_mkt`

## Coding And Repo Conventions

- Use Python 3.11+ with a `src/` layout
- Keep dependencies light and justified
- Prefer simple, explicit data structures over framework-heavy abstractions
- Separate configuration, research logic, and reporting artifacts
- Make experiment assumptions explicit in configs and docs
- Keep iteration logs current as scope evolves
- Write tests for basic behavior before adding simulation complexity

## Milestone Roadmap Summary

- Iteration 0: bootstrap repo, package, CLI, config loading, tests, and durable documentation
- Iteration 1: define local data schema, ingestion contract, and validation helpers
- Iteration 2: add parent-order representation and schedule-generation baselines
- Iteration 3: add simplified execution and transaction-cost accounting
- Iteration 4: add experiment runners, comparison outputs, and reports
- Iteration 5: refine modeling assumptions, edge cases, and calibration workflow
