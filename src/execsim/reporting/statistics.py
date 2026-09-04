from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def bootstrap_mean_interval(
    values: Sequence[float], *, confidence: float = 0.95, samples: int = 2_000, seed: int = 17
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("Bootstrap confidence must be in (0, 1) and samples must be positive.")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def aggregate_results(results: pd.DataFrame, *, seed: int = 17) -> pd.DataFrame:
    required = {"strategy", "implementation_shortfall_bps", "completion_rate"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Result table missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for strategy, group in results.groupby("strategy", sort=True):
        costs = pd.to_numeric(group["implementation_shortfall_bps"], errors="coerce").dropna()
        lower, upper = bootstrap_mean_interval(costs.to_numpy(), seed=seed)
        rows.append(
            {
                "strategy": strategy,
                "run_count": len(group),
                "mean_shortfall_bps": float(costs.mean()) if len(costs) else np.nan,
                "median_shortfall_bps": float(costs.median()) if len(costs) else np.nan,
                "std_shortfall_bps": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
                "q10_shortfall_bps": float(costs.quantile(0.10)) if len(costs) else np.nan,
                "q90_shortfall_bps": float(costs.quantile(0.90)) if len(costs) else np.nan,
                "bootstrap_mean_ci_lower_bps": lower,
                "bootstrap_mean_ci_upper_bps": upper,
                "mean_completion_rate": float(group["completion_rate"].mean()),
                "mean_modeled_cost": float(group["total_modeled_execution_cost"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy", kind="stable").reset_index(drop=True)


def paired_strategy_differences(
    results: pd.DataFrame,
    *,
    baseline: str = "twap",
    pair_columns: Sequence[str] = (
        "symbol",
        "trade_date",
        "side",
        "quantity",
        "start_time",
        "end_time",
        "hard_participation_rate",
        "risk_aversion",
        "temporary_impact",
    ),
    seed: int = 17,
) -> pd.DataFrame:
    available_pairs = [column for column in pair_columns if column in results.columns]
    if not available_pairs:
        raise ValueError("Paired comparison requires at least one sample identity column.")
    baseline_rows = results.loc[results["strategy"] == baseline]
    if baseline_rows.empty:
        raise ValueError(f"Baseline strategy is absent: {baseline}")
    rows: list[dict[str, object]] = []
    for strategy in sorted(set(results["strategy"]) - {baseline}):
        candidate = results.loc[results["strategy"] == strategy]
        paired = candidate.merge(
            baseline_rows,
            on=available_pairs,
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        differences = (
            paired["implementation_shortfall_bps_candidate"]
            - paired["implementation_shortfall_bps_baseline"]
        ).dropna()
        lower, upper = bootstrap_mean_interval(differences.to_numpy(), seed=seed)
        rows.append(
            {
                "strategy": strategy,
                "baseline": baseline,
                "paired_count": len(differences),
                "mean_difference_bps": float(differences.mean()) if len(differences) else np.nan,
                "median_difference_bps": float(differences.median())
                if len(differences)
                else np.nan,
                "win_rate": float((differences < 0).mean()) if len(differences) else np.nan,
                "bootstrap_mean_ci_lower_bps": lower,
                "bootstrap_mean_ci_upper_bps": upper,
            }
        )
    return pd.DataFrame(rows)
