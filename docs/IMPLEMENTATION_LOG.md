# Implementation log

## Sparse-JEPA empirical corpus gate — 2026-09-05

- Verified that GitHub squash commit `0c9b25ee70611ca7c1c98de070a3f293f27ca3d7` has the same tree as accepted PR #2 head `22963d351d5e0b195249662d9115f0b3f6b0ca0c`.
- Classified the merged design-freeze hash mismatch as a protocol-preserving implementation defect. Rebound `sparse-jepa-v1` before empirical inspection to the accepted tree, all six YAML files, and all normative documents, and added an enforced checksum sidecar.
- Enabled only the separately authorized network stage. Historical training and full evaluation remain disabled.
- Added a pinned, checksummed formation-constituent source, deterministic stable share-class identities, sourced ticker intervals, a pre-request storage/request plan, and a paginated Alpaca SIP entitlement probe.
- Corrected paper bar acquisition to request the symbol identity as of each interval and normalize retained raw-adjustment bars to the exact `America/New_York` regular-session grid before receipt validation.
- Corrected acquisition semantics discovered on APD: nonempty but incomplete monthly bars are retained with zero validated sessions for explicit exclusion accounting, while genuinely zero-row chunks remain failed receipts and do not count as complete.
- Corrected empty post-termination provider frames discovered on TIF so they produce schema-valid, explicit zero-row failure receipts instead of failing normalization with a missing timestamp column.
- Replaced the formation-universe all-corpus concatenation after a measured 6.63 GiB process footprint with a fail-closed, per-instrument monthly scan. The streaming calculation preserves the original price, completeness, and dollar-volume statistics while bounding peak corpus memory.
- Corrected the formation-completeness denominator to exclude the one predeclared 2021 early close, matching the primary 390-minute session contract. The rebuilt locked scan retained 62 of 505 candidates at the unchanged 95% threshold, still below the required universe size of 100. Classified the remaining shortfall as a Class C data limitation, rebuilt `CORPUS_QUALITY_REPORT.md`, and stopped before target acquisition, corporate-action processing, fold construction, or historical training.
- Restored the tracked network authorization default to `false` after the separately authorized acquisition. Rebound only the operational configuration/checksum fields of the tracked freeze; the acquisition-time config hash remains preserved in every acquisition receipt and in the quality report, and no scientific field changed.
- Recorded the initial merged-baseline result as `FAIL` (131 passed, 3 stale-freeze failures); the focused corrected gate is `PASS`. No historical model fit or test-model result was produced.

## Sparse-JEPA v2 resolution correction — 2026-09-05

- Preserved v1 as a blocked protocol with 62 of 505 eligible names, a 251-session corrected denominator, source-commit-bound normative hashes, and no target acquisition, historical training, test inspection, or TCA.
- Accepted ADR 0009. V2 measures formation quality from direct SIP daily bars, representation quality from 26 fixed observed-only 15-minute tokens, and TCA quality from the exact 300 consumed minutes from 10:30 through 15:29.
- Defined token OHLC, sum volume and trade count, volume-weighted VWAP, and observed-close realized variation without inserting, interpolating, or zero-filling absent provider minutes. Early closes remain daily-valid when observed and outside the primary 26-token sample.
- Added v2-only configuration, acquisition, formation-statistics, and resolution-quality code. Provider-semantics research passed; formation acquisition and counts remain `NOT RUN` until the protocol freeze is finalized.
- Historical model training, locked-test inspection, and historical TCA remain `NOT RUN`. Target acquisition remains prohibited before the v2 formation approval gate.

## Sparse predictive-representation paper framework — 2026-09-04

- Accepted ADR 0006 and added a focused paper specification covering the hypothesis, causal boundary, locked folds, artifacts, statistical method, and rejected alternatives.
- Added a frozen formation-universe schema, resumable immutable Alpaca SIP acquisition units, atomic receipts, point-in-time split adjustment, and regular-session validation. No provider request ran during implementation.
- Added one-session `26 x 18` sequence tensors, causal sample indexes, deterministic two-position train sampling, and training-fold robust normalization.
- Added matched dense and sparse shared-encoder JEPA models, exact-forward RepReLU with GELU derivatives, direct horizon predictors P0/P1/P2, generalized-Gaussian targets, rectified RMS-normalized RDMReg, diagnostics, difficulty weights, and safetensors checkpoints.
- Added native LightGBM scale/shape forecasting, a dimension-matched random projection control, causal 15-minute-to-minute `VolumeForecast` expansion, fold-safe moving-block bootstrap, and deterministic paper-bundle generation.
- Redirected the paper protocol to frozen representation accessibility and information retention: 13 dynamic encoder fields plus five predictor-conditioning fields, dense versus rectified-Gaussian sparse geometry, common RDM selection, horizon-specific frozen probes, an untrained nonlinear placebo, residual scale targets, bounded shape origins, causal held-out histories, 15-minute committed execution, oracle-relative regret, and main/appendix reporting separation. The prior adaptation and random-projection rows remain non-paper legacy functionality.
- Added six paper configuration files and the `execsim ml paper` command group. Network acquisition, historical training, and full evaluation require separate configuration and command-line enablement.
- Added deterministic CPU synthetic tests. These tests prove software plumbing and numerical invariants only; historical predictive quality and paper conclusions remain `NOT RUN`.
- Expanded safe checkpoint, embedding, and run provenance to cover the complete compatibility contract. Added executable synthetic fixtures for representation export, forecast and representation evaluation, MPC/TCA, and the exact paper output bundle.
- Closed all 20 software acceptance gates with 101 passing tests, 76% branch-aware coverage, Ruff, mypy over 104 source files, and 14-area repository validation. The detailed classifications are in `docs/PAPER_IMPLEMENTATION_REPORT.md`.

## Performance hardening and ADR governance — 2026-09-04

- Added indexed ADRs for the Python stack and navigation, point-in-time boundary, convex QP and impact model, forecast-only ML boundary, and performance workspace design.
- Updated `AGENTS.md` to require ADRs for material decisions and to preserve accepted history through superseding records.
- Replaced repeated historical DataFrame filtering, timestamp formatting, and pivot construction with a target-date-keyed causal NumPy matrix cache.
- Added `OptimalExecutionWorkspace`, exact-horizon OSQP setup/update reuse, cached backend selection, structured tail-risk Hessian construction, structural production validation, and component-level timing diagnostics.
- Preserved the pre-change deterministic 60-bucket MPC integer schedule on the benchmark fixture.
- Added deterministic profiling and extended benchmark output. The MPC benchmark median improved from 5.0630 seconds to 1.3228 seconds; the profiler improved from 4.544 seconds to 0.554 seconds.

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
