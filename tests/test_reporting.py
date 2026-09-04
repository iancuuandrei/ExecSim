from __future__ import annotations

import pandas as pd
import pytest

from execsim.reporting import aggregate_results, paired_strategy_differences


def _results() -> pd.DataFrame:
    rows = []
    for sample, twap, vwap in (("a", 2.0, 1.0), ("b", 4.0, 5.0), ("c", 6.0, 3.0)):
        for strategy, cost in (("twap", twap), ("vwap", vwap)):
            rows.append(
                {
                    "sample": sample,
                    "strategy": strategy,
                    "implementation_shortfall_bps": cost,
                    "completion_rate": 1.0,
                    "total_modeled_execution_cost": cost * 10,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_statistics_and_bootstrap_are_reproducible() -> None:
    first = aggregate_results(_results(), seed=5)
    second = aggregate_results(_results(), seed=5)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["strategy"]) == {"twap", "vwap"}
    assert (first["run_count"] == 3).all()


def test_paired_comparison_uses_identical_samples_and_reports_win_rate() -> None:
    paired = paired_strategy_differences(
        _results(), baseline="twap", pair_columns=("sample",), seed=5
    )

    assert paired.loc[0, "paired_count"] == 3
    assert paired.loc[0, "mean_difference_bps"] == pytest.approx(-1.0)
    assert paired.loc[0, "win_rate"] == pytest.approx(2 / 3)
