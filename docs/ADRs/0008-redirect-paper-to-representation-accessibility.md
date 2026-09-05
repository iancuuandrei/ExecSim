# ADR 0008: Redirect the paper to representation accessibility

- Status: Accepted
- Date: 2026-09-05
- Owners: ExecSim maintainers
- Supersedes: [ADR 0006](0006-sparse-predictive-representation-paper-framework.md) and [ADR 0007](0007-harden-historical-paper-pipeline-contracts.md)

## Context

The first PR #2 design combined representation geometry, forecast usefulness, difficulty adaptation, and execution value too early. It also encoded exogenous clock and scale fields, used a Laplace target as the primary sparse geometry, selected RDMReg strength independently by gradient calibration, treated a linear random projection as the placebo, and averaged seed forecasts before the primary execution comparison. Those choices made a negative or positive downstream result difficult to interpret.

The historical orchestration and fail-closed artifact work from ADR 0007 remains necessary. The paper needs a narrower hierarchy that first asks whether information is accessible from frozen representations, then whether it helps a strong supervised forecaster, and only then whether forecast differences change a fixed execution decision.

## Decision

Adopt the following active protocol:

- Treat representation accessibility as RQ1, information retention and supervised forecast value as RQ2, and modeled execution decision value as the application analysis. Keep support and regime results secondary and scaling axes future-only.
- Keep the stored 26 by 18 observation tensor, but encode only 13 dynamic fields. Supply the five cumulative-volume, ADV, clock, and elapsed fields from the current observed token to the predictor as conditioning. Mask linked padded latents before every predictor and distribution calculation.
- Compare a dense standard-Gaussian target with a sparse rectified-Gaussian target using `p=2`, `mu=Phi^-1(0.25)`, `sigma=1`, RepReLU, and analytically derived RMS. Move the matched rectified-Laplace target and the 50%, 75%, and 87.5% sparsity sweep to bounded appendix analyses.
- Use one shared JEPA MLP predictor with input dimension 1,053 and equal per-example predictive loss across each example's valid horizons. Compute RDMReg in FP32 over actual valid linked latents with 512 training projections. Compute the 2,048-projection diagnostic over a deterministic uniform sample capped at 2,048 valid latents so validation memory is corpus-bounded.
- Select one common RDM coefficient from `{0.1, 1, 10}` using Fold 1 validation, seed 13, equal dense and sparse budgets, required collapse gates, and mean fixed-observable probe error. Freeze that coefficient for the main 18 representation runs.
- Train each representation once and freeze it. Measure accessibility with horizon-specific affine/ridge, 64-unit MLP, and 256-unit MLP probes. Normalize latent prediction error by TRAIN latent covariance trace and report zero, TRAIN-mean, and persistence baselines. Use actual future bucket-volume surprise against the causal seasonal expectation as the fixed observable probe.
- Give raw LightGBM every causal field. Replace the random linear control with one frozen, untrained nonlinear network having the JEPA encoder and predictor architecture. Export 640 causal latent values plus four explicit horizon-availability flags; unavailable predicted horizons are zero.
- Predict the remaining-volume residual relative to the causal baseline and reconstruct physical volume exactly. Bound shape training to one deterministic origin from each of four predeclared as-of bands per training session, with persisted inclusion probabilities and inverse-probability case weights divided over valid future rows. Keep every scale origin and the complete validation and test grids.
- Keep difficulty adaptation implemented as future work, but exclude it from every paper selection, main table, statistical contrast, and primary run count.
- Distinguish model-training cutoff, market-information as-of, and feature-history end in sequence, feature, model, and artifact provenance. Held-out ADV, prior close, seasonal history, and known effective corporate actions advance causally; fitted transforms, category vocabularies, and model weights remain frozen.
- Define regimes from the same independently observed historical-baseline statistic and TRAIN thresholds. Report sparse support activity, transition, and chance-Jaccard references without using embeddings to define regimes.
- Evaluate each seed as a primary replicate and average metric effects, not forecasts. Restrict seed ensembling to a labeled appendix.
- At each 15-minute decision boundary, solve the constrained allocation once, commit the next 15 one-minute planned quantities, and replay fills minute by minute without re-solving. Use a realized-future-volume oracle under the same continuous constrained allocation. Report normalized allocation regret and absolute modeled impact cost as primary execution endpoints; implementation shortfall is secondary.
- Form exact complete cases before aggregation. Use fold-stratified five-date moving blocks as primary inference, one- and ten-date blocks as sensitivity, and Holm correction for the five predeclared confirmatory contrasts if formal claims are made. Remove arbitrary binary success thresholds.
- Freeze the source-hashed protocol before model comparison and write a second checksummed parameter-selection freeze after validation-only RDM and LightGBM selection. Locked-test stages refuse to run without it.
- Derive resource bounds from sequence manifests before training and keep measured device throughput separate from projected historical workload. Main reporting contains four tables and four figures; support, regime, block-length, sparsity, and order-size analyses remain appendix artifacts.
- Freeze protocol choices in a machine-readable design artifact now. Record parameter selection separately after Fold 1 validation only. Keep dormant schema axes for future capacity, dimension, sparsity, data-scale, and market-complexity work without running them in this paper.

## Rationale

The dynamic/conditioning split prevents a representation from appearing predictive mainly because it encoded deterministic clock or scale state. A rectified Gaussian isolates non-negativity and support sparsity from tail shape; the Laplace appendix then tests tail geometry without changing the main comparison. One common RDM coefficient and matched budgets reduce geometry-specific tuning freedom.

The capacity ladder measures accessibility after representation training instead of retraining three representation predictors. Covariance-normalized error makes dense and sparse scales comparable and fixed baselines show whether a probe does more than reproduce a mean or persist the current latent. The observable volume-surprise probe connects latent accessibility to a market quantity without letting the downstream LightGBM result define RQ1.

Residual scale prediction anchors the learned forecast to the same causal baseline available to every method. Bounded shape sampling controls corpus expansion while inverse-probability weighting preserves the declared training estimand. A nonlinear untrained control is a more credible test of feature expansion than a linear projection.

Committing each 15-minute allocation makes the optimizer clock match the forecast clock. Per-seed effects preserve training variability that forecast averaging would hide. Oracle-relative allocation regret isolates decision quality under the repository's explicit modeled costs; it is not a claim about realized market impact.

## Consequences

Sequence and embedding schemas change incompatibly and must receive new schema identities. Existing PR #2 artifacts cannot be reused. The primary representation matrix remains 18 runs; six Fold 1 development runs select the common RDM coefficient, three Laplace appendix runs and two extra sparsity runs cap the declared matrix at 29.

Historical acquisition, training, evaluation, and TCA remain separately privileged and are not executed by this decision. The synthetic acceptance fixture proves software paths only. Difficulty adaptation, ensemble forecasts, Transformer capacity tests, arbitrary pass thresholds, and a random linear paper row remain available only as legacy or future functionality outside the active paper matrix.

## Verification

`docs/PAPER_DESIGN.md` is the normative protocol. `design-freeze-v1.json` records frozen choices. Focused tests cover the feature split, conditioning and padding invariance, target moments, equal-example loss, common-lambda selection, normalized representation metrics, horizon availability, residual reconstruction, bounded shape sampling, timestamp provenance, causal corporate actions, 15-minute committed decisions, oracle regret, complete cases, and fold-safe inference. Full repository gates and the serious synthetic multi-session fixture must pass before merge.
