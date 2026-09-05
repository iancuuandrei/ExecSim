"""Static figure contract for evaluated paper results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def render_line_figure(data: pd.DataFrame, *, x: str, y: str, output: Path) -> Path:
    """Render a deterministic static line figure through the optional reporting extra."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'reporting' extra to render paper figures.") from exc
    if x not in data or y not in data or data.empty:
        raise ValueError("Figure data must contain non-empty x and y columns.")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(data[x], data[y], color="#214761", linewidth=1.5)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
