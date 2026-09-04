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

## Documentation and repository practice

- [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/)
- [Google developer documentation style guide](https://developers.google.com/style)
- [GitHub Docs style guide](https://docs.github.com/en/contributing/style-guide-and-content-model/style-guide)

The repository-specific application of these guides is defined in `docs/standards/implementation.md`.
