# Predictive representation accessibility paper specification

This document is the normative protocol for `sparse-jepa-v2`. It defines software and future historical experiments; it reports no empirical result. V1 remains immutably blocked with 62 eligible names under its exact-minute formation rule. V2 amends only task-resolution data quality. Target acquisition, historical fitting, evaluation, and TCA remain separately authorized operations and are `NOT RUN`.

## Research hierarchy and boundary

The paper evaluates one chain in order:

1. **RQ1, representation accessibility:** how well can fixed-capacity probes recover future linked latents from a frozen context representation?
2. **RQ2, information retention and forecast value:** does the frozen representation retain a fixed observable future-volume signal, and does adding it improve a strong causal LightGBM volume forecast?
3. **Application analysis, execution decision value:** do forecast differences change modeled allocation cost under an otherwise identical deterministic optimizer and replay?

Support transitions and independently defined market regimes characterize the sparse representation. Capacity, latent dimension, data scale, and market complexity are future research axes, not current experiments.

The invariant boundary is:

```text
point-in-time market data -> representation/forecast -> VolumeForecast
  -> deterministic constrained allocation -> exogenous replay/TCA
```

Learned components cannot choose a trade, relax a constraint, observe an actual future token at inference, or modify replay and cost mathematics. `docs/MATHEMATICAL_MODEL.md` remains authoritative for execution mathematics.

## Data, folds, and time identities

The provider contract remains Alpaca SIP with raw adjustment and `America/New_York` time. The formation period is 2021-01-04 through 2021-12-31. V2 retrieves provider-native `1Day` bars and requires a valid daily observation on at least 95% of expected XNYS trading days. A daily row must have the expected instrument and local session date, one unique observation, finite OHLCV, trade count, and VWAP, positive OHLC and VWAP, nonnegative volume and trade count, `high >= max(open, close, low)`, and `low <= min(open, close, high)`. Early closes are ordinary expected daily sessions and are not failures when their daily observation is valid.

The complete 505-candidate table is rebuilt from the 2021-01-04 S&P 500 membership source; it is not derived by changing v1 output. Eligibility applies to the source's S&P 500 share-class rows. The compatibility field `security_type=ordinary_common_stock` is a protocol label and is not an independently sourced legal-security classification. Stable sourced instrument identity, a median daily price floor of $5, positive median daily dollar volume, and stable-ID tie-breaking remain unchanged. If at least 100 names qualify, v2 freezes exactly the top 100 by formation-period median daily dollar volume. The ranking is available only after the formation interval and no later delisting, rename, merger, or data gap triggers survivor replacement. Sourced ticker histories are mandatory and aliases are never inferred.

The target period remains 2022-01-03 through 2025-12-31. SPY is an input instrument, never an execution-universe member, and uses the same resolution-specific rules. Target-period token or minute gaps exclude individual cases rather than frozen instruments. V2 must complete and hash the formation report, candidate data, token diagnostics, v1/v2 bias diagnostic, universe, and regenerated resource/acquisition plan, then stop at `AWAITING V2 FORMATION APPROVAL` before any target acquisition.

The completed formation gate qualifies 497 of 505 candidates and freezes the declared top 100. The authoritative counts, member table, SPY audit, activity-bias diagnostic, and bounded resource evidence are in `V2_FORMATION_QUALITY_REPORT.md`. This formation result is data-quality evidence, not a model result.

Every research row and artifact distinguishes three times:

- `training_cutoff`: last date permitted to fit model parameters, transforms, categories, and selection decisions;
- `market_information_as_of`: latest market or action information available for the case;
- `feature_history_end`: latest prior session used by causal ADV, preceding-close, and seasonal features.

Held-out ADV20, preceding close, seasonal history, and corporate-action state advance using history strictly before the case. Normalizers, category vocabularies, representation weights, LightGBM weights, and selected parameters remain frozen at `training_cutoff`. An action may affect a case only when `known_at <= market_information_as_of` and `effective_at <= observation_at`. Announced-but-not-effective, effective-but-unknown, known-and-effective, and post-training known-and-effective cases are tested. The study records the provider data vintage as a limitation.

The locked folds are:

| Fold | Train | Validation | Test |
|---|---|---|---|
| fold-1 | 2022-01-03 to 2023-12-29 | 2024-01-02 to 2024-03-28 | 2024-04-01 to 2024-06-28 |
| fold-2 | 2022-01-03 to 2024-06-28 | 2024-07-01 to 2024-09-30 | 2024-10-01 to 2024-12-31 |
| fold-3 | 2022-01-03 to 2024-12-31 | 2025-01-02 to 2025-06-30 | 2025-07-01 to 2025-12-31 |

Membership is derived from `(fold_id, session_date)` and fails closed. Artifact identity uses `(fold_id, instrument_id, session_date)` because expanding folds legitimately reuse earlier dates.

## Resolution-specific session and feature contract

V2 preserves four independent quality identities for each instrument-session: daily observation quality, exact full-session minute quality, 15-minute-token representation quality, and exact TCA-window minute quality. The manifest fields are `daily_valid`, `minute_exact_full_session`, `token_valid_full_session`, `tca_window_exact`, `early_close`, `provider_gap_count`, `observed_minute_count`, `valid_token_count`, `invalid_token_reason`, `token_observed_bar_counts`, and `token_provider_gap_counts`. No downstream stage may infer one identity from another.

Primary representation sessions are standard XNYS sessions partitioned into 26 fixed local-time intervals: 09:30-09:44, 09:45-09:59, through 15:45-15:59. V2 aggregates only observed provider minute bars in each interval and never inserts, interpolates, or zero-fills a missing bar. Each token requires at least two chronological rows with valid finite raw fields. Open is the first observed open; high and low are extrema; close is the last observed close; volume and trade count are sums; and VWAP is the observed-volume-weighted mean. Realized volatility is

```text
sqrt(sum((log(close_i) - log(close_(i-1)))^2 for consecutive observed closes))
```

This is an observed-grid realized-variation convention: a return across an absent provider minute is retained as one gap-spanning return and is not represented as a regular one-minute return or elapsed-time normalized. The previous valid token close supplies the cross-token return. All 26 tokens and every required cross-token input must be valid. Early closes remain outside the primary 26-token experiment. A 385-row session can therefore be representation-valid even though it is not exact at minute resolution.

The corpus-wide builder must apply `resolution-aware-v2` to target validation, stock sessions, SPY sessions, and every causal seasonal-history aggregation. Sequence records persist the protocol, total provider-gap count, and all 26 observed-bar counts. The record symbol is the unique symbol observed for that instrument-session and must match the sourced ticker interval for the same stable instrument and date; the formation symbol is never reused as an unconditional target-period label.

Formation-year token completeness uses the 251 predeclared standard-session denominator. Before the token scan, the diagnostic bands are frozen as high at least 95%, medium at least 80% but below 95%, and low below 80%. These labels describe data availability only and cannot select the universe or tune a model.

Full-session minute exactness independently requires every expected XNYS regular-session timestamp exactly once, in order, with one instrument/date, exact timezone, finite schema values, positive prices, nonnegative counts and volume, and the OHLC inequalities. The primary TCA contract applies the same strict rule only to the simulator's consumed interval: start-inclusive 10:30 through end-exclusive 15:30, or exactly 300 timestamps ending at 15:29. TCA never imputes prices, capacity, or fills. Consequently, the representation sample may be larger than the TCA complete-case sample.

The stored token tensor retains these 18 causal observations:

1. close-to-close log return;
2. open-to-close log return;
3. log high-low range;
4. realized volatility;
5. share-volume surprise;
6. dollar-volume surprise;
7. trade-count surprise;
8. cumulative-volume ratio;
9. ADV bucket ratio;
10. change in volume surprise;
11. VWAP to previous close;
12. log ADV20;
13. time sine;
14. time cosine;
15. elapsed fraction;
16. SPY return;
17. SPY volume surprise;
18. SPY realized volatility.

The encoder receives the 13 dynamic fields at positions 1-7, 9-11, and 16-18. The predictor separately receives the five current conditioning fields: cumulative-volume ratio, log ADV20, time sine, time cosine, and elapsed fraction. Symbol is absent from the representation. The sequence store also persists raw target volume, VWAP, the causal expected bucket-volume baseline, token availability, source hash, and all three time identities.

Contexts use eight completed tokens with explicit left padding. The first origin is 10:30, the minimum observed context is four tokens, and forward horizons are 1, 2, 4, and 8 tokens. Targets beyond close are masked. Training samples exactly two deterministic origins per session and epoch. Validation, test, representation evaluation, and embedding export use their complete deterministic grids. Normalization fits TRAIN only using 0.5% and 99.5% clipping and median/IQR scaling; padding stays zero.

## Primary representation geometry

The shared encoder is:

```text
Linear(13, 128) -> LayerNorm(128) -> GELU -> Linear(128, 128)
```

The same encoder processes context and future target tokens. Target gradients remain enabled. There is no EMA encoder, stop-gradient, dropout, batch normalization, attention, or sparse-only initialization. Dense and sparse branches use identical seeded initialization distributions.

Dense uses identity and `GeneralizedGaussian(p=2, mu=0, sigma=1)`. Sparse uses RepReLU and the primary target `p=2`, `mu=Phi^-1(0.25)=-0.6744897501960817`, and `sigma=1`. This produces 75% probability at zero after rectification. For scale `a=p^(1/p)sigma`, rectified moments use analytical incomplete-gamma tails and `target_rms=sqrt(E[ReLU(X)^2])`. Targets divide by this derived RMS. Parameters, both moments, zero fraction, and RMS are configuration and artifact identity fields.

The appendix contains exactly one matched rectified-Laplace control at 75% zeros and two additional Gaussian sparsity points needed to complete the 50%, 75%, and 87.5% support sweep. They are not main rows.

## JEPA predictor, loss, and RDMReg

The training predictor flattens `8*128` linked context values and appends eight mask values, five current conditioning values, and a learned 16-value horizon embedding. It applies `Linear(1053,256) -> GELU -> LayerNorm(256) -> Linear(256,128)` and then the geometry link. It predicts each horizon directly.

Linked context is multiplied by `context_mask[...,None]` before every predictor. For example `i`, predictive loss first averages latent MSE across its valid horizons, then averages equally across examples. RDMReg receives only actual valid linked context and target rows, never padding or predictions. It executes in FP32, including under BF16 training, with 512 training projections. Its 2,048-projection diagnostic uses a deterministic uniform sample of at most 2,048 actual valid latents so validation memory cannot scale with the corpus.

Select one common RDM coefficient from `{0.1,1,10}`. Fold 1, seed 13 trains dense and sparse candidates with identical budgets. Eligible candidates pass collapse gates. Selection minimizes mean validation error across geometries on the fixed observable probe. Store this parameter-selection receipt separately from the design freeze, then freeze the coefficient for all 18 primary runs. Test and TCA cannot select it.

Training uses AdamW, learning rate `3e-4`, weight decay `1e-4`, 5% warmup, cosine decay, clipping at 1, at most 40 epochs, and patience 6. Seeds are 13, 29, and 47. CUDA/BF16 is used when supported; CPU/FP32 is a debugging path. Historical loading is bounded, lazy, and Windows-spawn-safe. Non-finite state fails immediately. Periodic safetensors and separate checksummed trusted continuation state support deterministic resume. Best selection uses validation only and requires collapse gates.

## Frozen accessibility and observable probes

Train each representation once and freeze it. Evaluate three horizon-specific probe capacities: affine or ridge P0, 64-unit MLP P1, and 256-unit MLP P2. Probe outputs have no geometry link or rectification. The main capacity analysis does not retrain JEPA and does not use a Transformer.

Fit latent scale from TRAIN only. For `S=trace(Cov_train(z))`, report `mean(||z_hat-z||^2/S)` and the equivalent per-dimension value. Compare with zero, broadcast TRAIN mean, and current-latent persistence. RQ1 includes only origins with all four horizons, the derived 10:30 through 14:00 interval.

The fixed observable probe target at horizon `h` is `log(1+actual bucket volume)-log(1+causal seasonal expected bucket volume)`. Fit horizon-specific ridge probes from frozen representation values using TRAIN only. This is also the common-RDM selection observable.

## Embeddings, placebo, and LightGBM

Each export contains current observed linked latent, predicted `h1`, `h2`, `h4`, and `h8` latents, then four horizon-availability flags. Unavailable predictions are zero. Width is 644. Actual future target latents are prohibited.

The placebo is a frozen, untrained nonlinear network with the JEPA encoder and predictor architecture and seed discipline. It consumes no targets and exports the same 644 columns. The former linear Gaussian projection is not a paper row.

Raw LightGBM receives the full causal `8*18` context, mask, category, as-of/future-bucket metadata, calendar state, liquidity group, and causal baseline. The TRAIN symbol vocabulary is persisted and reused unchanged.

The scale target and inverse are:

```text
y = log(1 + remaining_volume) - log(1 + baseline_remaining_volume)
remaining_hat = max(0, (1 + baseline_remaining_volume) * exp(y_hat) - 1)
```

Every origin enters scale training. Shape uses one long-form model with target `log(conditional_share+1e-6)`. TRAIN selects exactly one deterministic origin per session from each available token band: 4-9, 10-15, 16-20, and 21-25. Persist inclusion probability. Weight the case by inverse probability and divide that weight over its valid future rows. Validation and test use complete grids. Stable softmax covers only valid future buckets.

The eight-candidate grid remains leaves `{15,31}`, minimum child rows `{50,200}`, L2 `{1,10}`, learning rate 0.03, 2,000 rounds, early stopping 100, feature and bagging fractions 0.8, and bagging frequency 1. Select scale by validation log-volume MAE and shape by validation curve error. Main rows are EWMA, raw, untrained neural, dense JEPA, and sparse JEPA. Difficulty adaptation is future-only. Primary forecast metrics are log remaining-volume MAE and cumulative conditional-curve error.

## Regime and support characterization

Regimes are exploratory and never derived from embeddings. A row is unusual when current volume surprise exceeds its TRAIN 90th percentile, current realized volatility exceeds its TRAIN 90th percentile, or the cumulative-curve error of the causal historical baseline exceeds its TRAIN 90th percentile. The identical three inputs and OR statistic apply in TRAIN, validation, and test; learned-model error never defines a regime.

Sparse-only support results include mean, median, and p95 active dimensions; activation-frequency quantiles; consecutive support Jaccard; support-transition rate; per-regime Jaccard; transition matrices where identified; and chance Jaccard from marginal activation rates. Do not report a redundant nonzero-magnitude view as a separate representation.

## Forecast provider and execution clock

At a 15-minute boundary, the provider builds causal features, predicts residual scale and valid shape, reconstructs volumes, and disaggregates with a TRAIN-only within-token profile. Between boundaries it removes elapsed minute weights from the cached curve and renormalizes without rerunning the model or restarting a partial token.

The optimizer also acts every 15 minutes. It solves the continuous constrained remaining-horizon allocation once and commits the next 15 planned minute quantities. Replay applies actual hard participation, inventory, price, and costs each minute without re-solving the committed segment.

The primary sample is 30 liquidity-spaced stocks, all valid test dates, 10:30 to 15:30, quantity 3% causal ADV20, deterministic balanced sides, 10% planned and hard participation, and zero risk and tracking penalties. Half-spread is arrival price times `5e-5`; temporary impact at full participation is arrival price times `1e-3`. A 10-stock appendix uses 1% and 5% ADV.

The evaluation-only oracle uses realized future volume in the same continuous constrained allocation. Primary execution metrics are `normalized_allocation_regret=(C_model-C_oracle)/max(C_oracle,epsilon)` and absolute modeled impact cost `C_model`. Implementation shortfall is secondary. These are modeled costs, not observed market response.

Evaluate every JEPA seed as a primary replicate and average paired metric effects, not forecasts. A three-seed ensemble is appendix-only.

## Matching, inference, freeze, and reporting

Intersect exact case IDs before comparison, including fold, date, instrument, as-of or target, order size, side, window, constraints, and cost identities. Duplicate method-case rows fail. Report dropped counts, form paired differences on the intersection, then average within date.

Use a fold-stratified five-trading-day moving-block bootstrap with 10,000 repetitions and 95% intervals. Each fold retains its intended date contribution and blocks never cross folds. One- and ten-day blocks are sensitivity analyses. If formal confirmatory claims are made, apply Holm correction to five predeclared contrasts. No arbitrary binary success threshold applies.

`configs/paper/sparse_jepa/design-freeze-v1.json` preserves its original bytes. The separately named `safe-default-receipt-v1.json` and terminal evidence preserve the later implementation and blocked acquisition state without requiring Git history at load time. `configs/paper/sparse_jepa_v2/design-freeze-v2.json` stores the v2 choices and binds the six v2 YAML files, normative documents, and unchanged scientific matrix. The only conceptual changes are daily formation completeness, token-resolved representation completeness, exact consumed-window TCA completeness, and nonzero interpretation of absent provider minutes. After future common-RDM and per-fold LightGBM validation selection, a v2 parameter-freeze receipt records selected values, Git commit, config hash, timestamp, and checksummed upstream manifests before any locked-test stage can run. Schemas retain dormant `encoder_capacity`, `predictor_capacity`, `latent_dimension`, `target_sparsity`, `training_data_scale`, and `market_complexity_stratum` fields for future work without varying them now.

The maximum representation matrix is 29: 18 primary, six common-lambda development, three Laplace appendix, and two extra-sparsity runs. Historical main tables cover dataset/folds/exclusions, accessibility/baselines, forecasting, and oracle-relative execution. Main figures cover the frozen capacity curve, model-level forecast performance, forecast error by as-of, and allocation regret with paired intervals. Support, regimes, block-length sensitivity, sparsity, Laplace, and order-size sensitivity remain appendix artifacts. Synthetic fixtures cannot populate historical tables.

Before training, manifest-derived resource planning bounds JEPA steps, long-shape rows, embedding rows and bytes, and the 29-run ceiling. The run aborts when a configured safe bound is exceeded. `paper plan` can measure bounded local sequence, RDMReg, embedding, and TCA throughput; measured CPU or GPU telemetry is evidence for that device only and is never converted into an unverified historical runtime promise.

## Authorization and evidence

Network, historical fitting, and full evaluation each require their matching configuration switch and CLI flag. `execsim ml paper run` resumes compatible artifacts and executes authorized stages only.

Acceptance uses deterministic fixtures with at least four synthetic stocks plus SPY, 40 sessions over two folds, causal histories, corporate-action timing cases, one malformed exclusion, accessibility baselines, bounded shape sampling, the nonlinear placebo, residual LightGBM, provider truncation, committed MPC, oracle regret, complete cases, bootstrap sensitivities, and historical-schema reporting. Tiny fixture fits prove software paths only. Historical effectiveness remains `EMPIRICAL RESULT NOT AVAILABLE`.
