# Performance and benchmark contract

ExecSim optimizes measured research hot paths after correctness, point-in-time integrity, deterministic output, and readable mathematics.

## Benchmark method

Run the tracked synthetic benchmark from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py
```

The benchmark uses deterministic 390-minute synthetic sessions, a 60-minute execution window, a 50,000-share parent order, and Python `tracemalloc`. Timings include validation, forecast generation, optimization, simulation, and tracing. They are environment-dependent telemetry.

## Baseline and optimized result

The 2026-09-04 checkpoints used Python 3.13.7, NumPy 2.3.5, and Windows 11. The same tracked benchmark reported:

| Path | Repetitions | Baseline median | Optimized median | Median ratio |
|---|---:|---:|---:|---:|
| Static TWAP simulation | 20 | 0.0304 s | 0.0727 s | 0.42x |
| Adaptive MPC simulation | 3 | 5.0630 s | 1.3228 s | 3.83x |
| Five-strategy comparison, one symbol-day | 2 | 5.5745 s | 3.5655 s | 1.56x |
| Five-strategy, two-day experiment | 2 | 12.0693 s | 4.3976 s | 2.74x |
| Five-session Parquet scan and 5-minute ML build | 3 | 0.1456 s | 0.3178 s | 0.46x |

The target MPC path exceeded the 3x median goal. Unchanged TWAP and ML paths varied materially between runs, which demonstrates why absolute wall-clock values are telemetry rather than a cross-machine or load-independent guarantee. No optimization claim is made for those paths.

## Profile evidence

Run the deterministic profiler independently of `tracemalloc`:

```powershell
.\.venv\Scripts\python.exe scripts/profile_performance.py --workload mpc --top 40
.\.venv\Scripts\python.exe scripts/profile_performance.py --workload experiment --top 40
```

The MPC profile fell from 4.544 seconds to 0.554 seconds, an 8.20x ratio. The two-day experiment profile fell from 7.813 seconds to 2.463 seconds, a 3.17x ratio. Before the change, historical forecasting consumed 3.763 seconds in the MPC profile and 5.885 seconds in the experiment profile. After the change, those totals were 0.052 seconds and 0.184 seconds.

The warmed 60-decision MPC component sample reported 0.0057 seconds for matrix construction, 0 seconds for setup, 0.0097 seconds for numeric solver updates, 0.0007 seconds for structural eigenvalue validation, 0.0195 seconds for OSQP solves, and 0.0138 seconds for integer projection. All 60 exact-horizon setups were reused in that sample.

## Current implementation choices

The implementation caches causally filtered NumPy volume matrices by symbol scope and target date, reuses providers and compatible policies within an experiment, constructs risk curvature from tail sums, and caches one OSQP setup per exact horizon. It also caches OSQP's selected algebra backend within a workspace so repeated solver construction does not repeat backend discovery. Shifted warm starts are dimensionally matched and clipped to inventory and capacity bounds. Fixed tolerances, status checks, residuals, completion checks, and deterministic integer projection preserve the numerical contract.

Adaptive MPC uses structural convexity validation: strictly positive temporary-impact curvature plus positive-semidefinite risk and tracking terms provide an analytical lower bound. Static, standalone, and mathematical test solves retain the full eigenvalue check.

PyArrow scanners apply column projection and bounded symbol discovery to ML source files. CLI builds scan and transform one symbol at a time, write that partition immediately, and do not retain result rows. Direct library callers may request returned rows for small interactive or test datasets; they should disable materialization for a large universe.

## Deferred acceleration

Rust, C++, Cython, Numba, Polars, DuckDB, distributed execution, and GPU infrastructure remain outside V1. Adopt one only after a representative profile identifies a bottleneck and the change preserves numerical equivalence and point-in-time behavior.
