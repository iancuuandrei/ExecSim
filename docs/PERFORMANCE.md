# Performance and benchmark contract

ExecSim optimizes measured research hot paths after correctness, point-in-time integrity, deterministic output, and readable mathematics.

## Benchmark method

Run the tracked synthetic benchmark from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py
```

The benchmark uses deterministic 390-minute synthetic sessions, a 60-minute execution window, a 50,000-share parent order, and Python `tracemalloc`. Timings include validation, forecast generation, optimization, simulation, and tracing. They are environment-dependent telemetry.

## Local result

The 2026-09-04 checkpoint used Python 3.13.7, NumPy 2.3.5, and Windows 11. Results were:

| Path | Repetitions | Median | Minimum | Peak traced memory |
|---|---:|---:|---:|---:|
| Static TWAP simulation | 20 | 0.0285 s | 0.0280 s | 0.49 MB |
| Adaptive MPC simulation | 3 | 4.9626 s | 4.7657 s | 3.53 MB |
| Five-strategy comparison, one symbol-day | 2 | 6.5067 s | 6.4871 s | 1.51 MB |
| Five-strategy, two-day experiment | 2 | 15.0489 s | 14.4044 s | 1.83 MB |
| Five-session Parquet scan and 5-minute ML build | 3 | 0.1439 s | 0.1338 s | 1.79 MB |

The complete 384-unit historical experiment finished successfully in approximately 90 seconds on the same machine. These measurements establish local operability, not production capacity or a cross-machine guarantee.

## Current implementation choices

The implementation caches loaded symbol bars within an experiment, uses vectorized NumPy formulas, warm-starts shrinking MPC horizons when compatible, writes columnar artifacts, and fixes ordering and seeds. OSQP adaptive penalty updates handle objective scaling while fixed tolerances and post-solve acceptance checks preserve the numerical contract.

PyArrow scanners apply column projection and bounded symbol discovery to ML source files. CLI builds scan and transform one symbol at a time, write that partition immediately, and do not retain result rows. Direct library callers may request returned rows for small interactive or test datasets; they should disable materialization for a large universe.

## Deferred acceleration

Rust, C++, Cython, Numba, Polars, DuckDB, distributed execution, and GPU infrastructure remain outside V1. Adopt one only after a representative profile identifies a bottleneck and the change preserves numerical equivalence and point-in-time behavior.
