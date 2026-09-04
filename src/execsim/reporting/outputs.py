from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from execsim.reporting.statistics import aggregate_results, paired_strategy_differences


def write_research_outputs(
    *,
    results: pd.DataFrame,
    decision_trace: pd.DataFrame,
    spec: dict[str, object],
    output_dir: Path,
    seed: int,
    write_files: bool = True,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    aggregate = aggregate_results(results, seed=seed)
    paired = (
        paired_strategy_differences(results, seed=seed)
        if "twap" in set(results["strategy"])
        else pd.DataFrame()
    )
    paths: dict[str, Path] = {}
    if not write_files:
        return paths, aggregate, paired
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["run_results"] = output_dir / "run_results.parquet"
    paths["aggregate"] = output_dir / "strategy_aggregate.csv"
    paths["paired"] = output_dir / "paired_differences.csv"
    paths["config"] = output_dir / "config_snapshot.json"
    paths["provenance"] = output_dir / "provenance.json"
    paths["report"] = output_dir / "REPORT.md"
    results.to_parquet(paths["run_results"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    paired.to_csv(paths["paired"], index=False)
    if not decision_trace.empty:
        paths["decision_trace"] = output_dir / "decision_trace.parquet"
        decision_trace.to_parquet(paths["decision_trace"], index=False)
    config_text = json.dumps(spec, indent=2, sort_keys=True)
    paths["config"].write_text(config_text + "\n", encoding="utf-8")
    provenance = {
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "result_rows": len(results),
        "strategies": sorted(results["strategy"].unique()),
        "seed": seed,
        "evaluation_only_rows": int(results["evaluation_only"].sum()),
        "claim_boundary": "demonstration; not an empirical performance claim",
        "nondeterministic_fields": ["optimizer_time_seconds", "solve_time_seconds"],
    }
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(_markdown_report(results, aggregate, paired), encoding="utf-8")
    paths.update(_write_figures(results, output_dir))
    return paths, aggregate, paired


def _markdown_report(results: pd.DataFrame, aggregate: pd.DataFrame, paired: pd.DataFrame) -> str:
    best = aggregate.sort_values("mean_shortfall_bps", kind="stable").iloc[0]
    lines = [
        "# ExecSim strategy comparison",
        "",
        (
            "This report is a reproducible model demonstration. The configured costs are "
            "assumptions, and the sample does not establish predictive strategy superiority."
        ),
        "",
        "## Scope",
        "",
        f"- Runs: {len(results)}",
        f"- Strategies: {', '.join(sorted(results['strategy'].unique()))}",
        f"- Unique symbol-dates: {results[['symbol', 'trade_date']].drop_duplicates().shape[0]}",
        "",
        "## Descriptive result",
        "",
        (
            "The lowest mean modeled implementation shortfall in this grid was "
            f"`{best['strategy']}` at {best['mean_shortfall_bps']:.4f} basis points. "
            "Treat this as a property of the selected bars and assumptions, not a general "
            "ranking."
        ),
        "",
        "## Aggregate metrics",
        "",
        aggregate.to_markdown(index=False),
    ]
    if not paired.empty:
        lines.extend(["", "## Paired differences versus TWAP", "", paired.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "Minute bars do not represent quotes, queue position, within-bar paths, or "
                "counterfactual market reaction. Oracle rows, when present, are "
                "evaluation-only and use future information."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_figures(results: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    paths: dict[str, Path] = {}
    grouped = results.groupby("strategy", sort=True)
    figures: list[tuple[str, Any]] = []

    fig, axis = plt.subplots(figsize=(8, 4.5))
    data = [group["implementation_shortfall_bps"].dropna() for _, group in grouped]
    labels = [name for name, _ in grouped]
    axis.boxplot(data, tick_labels=labels)
    axis.set_ylabel("Implementation shortfall (bps)")
    axis.set_title("Modeled shortfall by strategy")
    axis.tick_params(axis="x", rotation=30)
    figures.append(("shortfall_by_strategy", fig))

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for name, group in results.groupby("strategy", sort=True):
        curve = group.groupby("quantity")["total_modeled_execution_cost"].mean()
        axis.plot(curve.index, curve.values, marker="o", label=name)
    axis.set_xlabel("Parent-order quantity (shares)")
    axis.set_ylabel("Mean modeled cost (currency)")
    axis.set_title("Modeled cost sensitivity to order size")
    axis.legend()
    figures.append(("cost_vs_quantity", fig))

    for name, figure in figures:
        path = output_dir / f"{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths[name] = path
    return paths
