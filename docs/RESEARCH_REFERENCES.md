# Research and implementation references

These sources motivate non-obvious model and implementation choices. ExecSim documents deviations and does not treat a citation as empirical validation.

## Optimal execution

- Almgren, R., and Chriss, N. [Optimal execution of portfolio transactions](https://www.math.nyu.edu/~chriss/optliq_f.pdf). The analytical reference and inventory-risk structure follow this classical discrete-time framework, while ExecSim's general QP adds explicit forecast capacity and tracking constraints.
- The OSQP project. [OSQP solver documentation](https://osqp.org/docs/). ExecSim records status, residuals, iterations, and timing and fails closed on unacceptable solutions.

## Benchmarks and market data

- Bank for International Settlements. [Execution algorithms and market functioning](https://www.bis.org/publ/mktc13.pdf). This provides institutional context for schedule-based execution algorithms and volume-oriented benchmarks.
- Alpaca. [Historical stock data documentation](https://docs.alpaca.markets/docs/historical-stock-data-1). ExecSim stores normalized vendor output locally and keeps simulation vendor-independent.
- pandas. [Time series and date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html). ExecSim requires timezone-aware timestamps at simulation and dataset boundaries.

## Numerical and ML infrastructure

- NumPy. [Random sampling documentation](https://numpy.org/doc/stable/reference/random/index.html). Synthetic scenarios and bootstrap intervals use explicit generators and seeds.
- scikit-learn. [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Preprocessing is fitted only on training partitions, and test partitions remain locked until selection is complete.
- Apache Arrow. [Dataset documentation](https://arrow.apache.org/docs/python/dataset.html). ML sources and partitioned artifacts use columnar Parquet interfaces.

## Predictive representations

- Assran et al. [Self-supervised learning from images with a joint-embedding predictive architecture](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html). ExecSim adopts the predictive-representation separation, not the image architecture or empirical claims.
- Balestriero and LeCun. [LeJEPA: Provable and scalable self-supervised learning without the heuristics](https://arxiv.org/abs/2511.08544). The shared trainable encoder and distribution-matching branch motivate the controlled dense reference.
- Kuang et al. [Rectified LpJEPA: Joint-embedding predictive architectures with sparse and maximum-entropy representations](https://arxiv.org/abs/2602.01456). RepReLU, rectified generalized-Gaussian targets, and sliced RDMReg motivate the sparse geometry. ExecSim applies them to a new forward intraday forecasting protocol and makes no transfer claim from image results.
- Kuang et al. [LpWM: A case for sparse representations in world models](https://arxiv.org/abs/2608.22764). The predictor-capacity question motivates P0/P1/P2, but ExecSim does not adopt action conditioning, planning from latents, or end-to-end control.

### Closest-work boundary

| Work | Borrowed idea | ExecSim distinction |
|---|---|---|
| I-JEPA | Predict in representation space | Shared trainable token encoder on causal intraday states; no image architecture, EMA teacher, or stop-gradient |
| LeJEPA | Distribution matching without heuristic teacher updates | Matched dense Gaussian reference with sliced RDMReg and future-target gradients enabled |
| Rectified LpJEPA | RepReLU and rectified generalized-Gaussian support | Primary comparison uses rectified Gaussian to separate sparse/nonnegative geometry from the Laplace-tail appendix |
| LpWM | Accessibility at different predictor capacities | Frozen horizon-specific affine, 64-unit, and 256-unit probes; no actions, latent planner, RL, or end-to-end control |
| Intraday volume forecasting | Remaining-volume and conditional-shape prediction | Information-matched raw LightGBM plus frozen representation hybrids, with a causal seasonal residual target |
| Optimal execution | Forecast-sensitive constrained allocation | Representation models only produce `VolumeForecast`; deterministic optimization, minute replay, and assumed costs remain fixed |

The paper does not treat lower self-prediction error as sufficient. It separately measures covariance-normalized latent accessibility, retention of a fixed future-volume observable, incremental supervised forecast value, and oracle-relative modeled allocation regret.

## Documentation and repository practice

- [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/)
- [Google developer documentation style guide](https://developers.google.com/style)
- [GitHub Docs style guide](https://docs.github.com/en/contributing/style-guide-and-content-model/style-guide)

The repository-specific application of these guides is defined in `docs/standards/implementation.md`.
