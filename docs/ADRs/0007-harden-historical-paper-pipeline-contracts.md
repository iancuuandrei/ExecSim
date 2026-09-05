# ADR 0007: Harden the historical paper pipeline contracts

- Status: Superseded by [ADR 0008](0008-redirect-paper-to-representation-accessibility.md)
- Date: 2026-09-04
- Owners: ExecSim maintainers
- Refines: [ADR 0006](0006-sparse-predictive-representation-paper-framework.md)

Implementation note: ADR 0008 retains the historical-pipeline hardening adopted here but supersedes the primary target geometry, RDM coefficient selection, predictor-capacity, placebo, adaptation, supervised-target, seed-aggregation, TCA-clock, and reporting decisions.

## Context

ADR 0006 established the controlled paper subsystem, but its first implementation proved only synthetic primitives. Audit of PR #2 found material gaps between those primitives and an executable historical experiment: sparse normalization was valid only at one operating point, padded encoder activations could reach predictors, the sparse branch received a unique bias, callers could contradict fold dates, 390-row sessions could contain the wrong minutes, and historical stages still required manual per-session glue. Checkpoint, LightGBM, adaptation, statistics, and reporting contracts also did not carry enough identity or structure for the locked experiment.

## Decision

Keep the research question and model families from ADR 0006, and adopt these executable contracts:

- Derive rectified generalized-Gaussian moments from `(p, mu, sigma)`. Normalize sparse RDMReg with `sqrt(E[ReLU(X)^2])`, and persist the moments and RMS.
- Mask linked context latents after encoding and before every predictor. Use identical encoder and predictor initialization distributions in the primary dense/sparse comparison.
- Derive partitions from `(fold_id, session_date)`. Validate exact XNYS regular-session minutes and OHLCV invariants before positional aggregation.
- Build the universe, sourced symbol history, corporate actions, SPY corpus, fold sequence stores, exclusions, train-only normalizers, and indexes through manifests. Missing provider identity metadata is `BLOCKED`; aliases are never inferred.
- Train historical JEPA through bounded PyTorch datasets and data loaders, device-aware mixed precision, exactly 32 training batches for RDMReg calibration, complete validation grids, collapse-gated selection, and complete trusted continuation state.
- Require full checkpoint compatibility, including geometry, adaptation, fold, cutoff, upstream hashes, target moments, RDMReg settings, configuration, and PyTorch policy.
- Rank difficulty level and shape errors separately within TRAIN as-of strata. Adapt across the weighted loader for `ceil(0.10 * base_steps)` steps and weight predictive loss only.
- Fit LightGBM from pandas frames with a frozen training symbol vocabulary. Use one scale model and one long-form shape model; normalize predictions only across valid future buckets. Select the two heads independently on their declared validation metrics from the same eight-point grid.
- Reuse each learned forecast for the 15-minute interval and truncate its original minute curve without restarting the within-token profile. Keep `VolumeForecast -> deterministic MPC -> TCA` unchanged.
- Form exact complete-case intersections before date aggregation. Resample blocks within each fold so every fold retains its observed contribution.
- Separate synthetic and historical report writers. Historical reports accept named result schemas and measured intervals only.
- Add one development-only rectified-Gaussian sparse control. It is not a main paper row.

## Rationale

These contracts remove avoidable causal and experimental confounds while retaining the matched shared-encoder study. Manifest-derived orchestration makes the future authorized run reproducible and idempotent. Full compatibility checks prevent shape-compatible but scientifically incompatible artifacts from being reused. Long-form shape prediction represents shrinking horizons directly instead of encoding an absolute output column as a separate model family.

## Consequences

The paper subsystem has more artifacts and validation steps, and the serious local fixture is slower than primitive unit tests. Ordinary ExecSim still does not import PyTorch or LightGBM. Network acquisition, historical fitting, and empirical evaluation remain separately privileged and are `NOT RUN` in this change. The software can be merge-ready without making an effectiveness claim only when the full synthetic multi-session acceptance fixture and repository gates pass.

## Alternatives considered

- Keep a hard-coded sparse RMS and separate shape model per output column. Rejected because both fail outside one target/horizon geometry.
- Mask raw padded tokens only. Rejected because encoder bias can recreate a nonzero latent.
- Trust caller-provided partitions or 390-row counts. Rejected because both allow silent causal or timestamp corruption.
- Encode ticker aliases heuristically. Rejected because unsourced identity repair can silently change the corpus.
- Load all sequences into one tensor. Rejected because the historical corpus must remain bounded in memory and Windows worker-safe.
- Globally pool bootstrap block starts. Rejected because folds with more possible starts receive unintended extra influence.

## Verification

`docs/PAPER_DESIGN.md` and `docs/SPECIFICATIONS.md` define the current contracts. `tests/test_paper_data.py`, `tests/test_paper_sequences.py`, `tests/test_paper_representations.py`, and `tests/test_paper_pipeline.py` cover their mathematical, causal, artifact, streaming, supervised, forecast, matching, and reporting invariants. Historical results remain unavailable until separately authorized data acquisition, fitting, and evaluation complete.
