# ADR 0006: Add a controlled sparse predictive-representation paper framework

- Status: Superseded by [ADR 0008](0008-redirect-paper-to-representation-accessibility.md)
- Date: 2026-09-04
- Owners: ExecSim maintainers

Implementation note: [ADR 0007](0007-harden-historical-paper-pipeline-contracts.md) refined this design. [ADR 0008](0008-redirect-paper-to-representation-accessibility.md) supersedes both records for the active paper protocol while preserving the forecast-only execution boundary and rejected model-family expansions.

## Context

ExecSim already separates point-in-time volume forecasts from deterministic constrained execution. The next research question is whether a sparse non-negative predictive representation makes intraday dynamics easier to predict and whether any forecast improvement survives a fixed MPC and transaction-cost-analysis boundary. The experiment must distinguish a representation claim from a trading claim, hold model capacity and supervised budgets as constant as practical, and remain reproducible without making a real-data result part of repository acceptance.

The study needs a larger licensed corpus, deep-learning and LightGBM dependencies, representation artifacts, and date-clustered inference. These capabilities must not become mandatory imports for ordinary simulation. Network access, real historical fitting, and a full paper run also require separate user authorization.

## Decision

Adopt a paper-specific research layer with these locked choices:

- Acquire immutable Alpaca SIP one-minute responses in resumable monthly chunks. Freeze 100 ordinary S&P 500 constituents using formation-period eligibility and median daily dollar volume as of 2021-01-04. Use stable instrument identifiers and never replace later survivors.
- Keep contemporaneous raw bars for replay. Apply point-in-time split adjustment only to cross-session ML features, with inverse price and direct volume restatement that preserves dollar notional.
- Store one 26-by-18, 15-minute tensor per complete regular session. Build causal windows from indexes, use eight completed context tokens, and predict horizons 1, 2, 4, and 8 directly. Fit 0.5/99.5 percentile clipping and median/IQR scaling on each training fold only.
- Compare one shared token encoder under matched dense identity and sparse RepReLU links. The future target uses the same encoder, receives gradients, and has no exponential-moving-average encoder or stop-gradient boundary.
- Use P1, a horizon-conditioned multilayer perceptron, for downstream results. Use a linear P0 and a four-layer tiny Transformer P2 only for the predictor-capacity test.
- Match linked latent distributions with sliced two-Wasserstein RDMReg. The dense target is standard Gaussian. The sparse target is a shifted unit-variance Laplace distribution with location `-log(2)/sqrt(2)`, rectified to 25% expected activity, and divided by its 0.5 root-mean-square value. Calibrate the RDMReg weight from 32 deterministic training mini-batches so its initial pre-link gradient norm is approximately 10% of the predictive-loss gradient norm.
- Adapt only the selected sparse checkpoint using training-only causal baseline-difficulty ranks. Apply weights to predictive loss, not RDMReg, for 10% additional steps at one-tenth the final learning rate.
- Compare EWMA and native LightGBM scale/shape forecasters on raw features, a deterministic 640-dimensional Gaussian placebo, dense embeddings, sparse embeddings, and adapted sparse embeddings. Use P1 for downstream comparisons. Select LightGBM settings on validation forecast metrics, never test or TCA results.
- Export only the last observed linked latent and the four predicted latents. Never export an actual future latent. Use safetensors for representation weights and native LightGBM text models with checksummed manifests.
- Evaluate all forecasting variants through the existing `VolumeForecast` to deterministic MPC boundary. Change only the forecast. Keep replay, constraints, quantity, side assignment, cost assumptions, and optimizer mathematics fixed.
- Use three global chronological folds and seeds 13, 29, and 47. Average paired cases equally by date and use a five-date moving-block bootstrap whose blocks never cross folds.
- Keep PyTorch, safetensors, and LightGBM in exact-version optional extras. Keep network acquisition, historical training, and the full paper run disabled by default with separate configuration and command-line authorization.

## Rationale

The design isolates representation geometry: dense and sparse branches share the encoder, predictor family, data, folds, seeds, and downstream learner. Direct multi-horizon prediction avoids an autoregressive error path that would confound geometry. A native scale/shape LightGBM baseline is strong for tabular volume forecasting while the random projection detects gains caused only by added dimensionality. Date-clustered inference respects the primary paired unit better than treating symbol-as-of rows as independent observations.

The existing optimizer boundary makes economic attribution interpretable. A representation can change forecasted volume but cannot choose trades, relax capacity, inspect future bars, or introduce a hypothetical market response.

## Consequences

Ordinary ExecSim installs and imports without the paper extras. Paper artifacts have stricter identities covering source, fold, cutoff, normalization, architecture, geometry, predictor, seed, checkpoint, and dependency versions. Tests use CPU synthetic fixtures only; they establish plumbing and mathematical invariants, not predictive quality.

The repository may truthfully complete software acceptance while corpus acquisition, historical fitting, and empirical research questions remain `NOT RUN`. A paper claim requires the locked, separately authorized historical experiment. The closest-work review must be refreshed before submission.

## Alternatives considered

- An EMA or separate target encoder was rejected because it changes the selected shared-encoder LpJEPA branch and weakens the matched comparison.
- Stop-gradient future targets were rejected for the same reason.
- Random interior masking and time reversal were rejected because the research question is forward intraday prediction from information available at an as-of time.
- RevIN was rejected because per-sequence normalization can remove absolute activity information needed for remaining-volume forecasting.
- Symbol identifiers in the encoder were rejected to reduce stock-identity memorization; LightGBM may use symbol categorically.
- End-to-end learned execution and reinforcement learning were rejected because they bypass the auditable optimizer boundary.
- A broad model zoo, foundation model, distributed framework, mandatory CUDA stack, experiment tracker, and notebook authority were rejected because they do not isolate the hypothesis and add operational variance.
- IEX fallback was rejected because it silently changes the corpus when SIP entitlement is unavailable.
- Test- or TCA-driven model selection was rejected because it invalidates the locked out-of-sample comparison.

## Verification

`tests/test_paper_data.py`, `tests/test_paper_sequences.py`, `tests/test_paper_representations.py`, and `tests/test_paper_pipeline.py` cover the software and synthetic mathematical contract. `docs/PAPER_DESIGN.md` defines the complete current specification. Historical outcomes remain unverified until an authorized paper run produces checksummed artifacts.
