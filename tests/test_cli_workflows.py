from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from execsim.cli import main


def test_scenario_command_writes_deterministic_parquet(tmp_path: Path) -> None:
    output = tmp_path / "scenario.parquet"

    status = main(
        [
            "scenario",
            "--volume-shape",
            "midday_drought",
            "--price-path",
            "volatility_shock",
            "--seed",
            "19",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    bars = pd.read_parquet(output)
    assert len(bars) == 390
    assert bars["scenario_seed"].unique().tolist() == [19]


def test_simulate_command_runs_selected_policy(tmp_path: Path, capsys) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    timestamps = pd.date_range("2026-01-05 10:00", periods=3, freq="min", tz="America/New_York")
    pd.DataFrame(
        {
            "symbol": "TEST",
            "timestamp": timestamps,
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "vwap": [100.0, 100.0, 100.0],
            "volume": [10_000, 10_000, 10_000],
        }
    ).to_parquet(processed / "TEST.parquet", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project_name": "cli-test",
                "symbols": ["TEST"],
                "start_date": "2026-01-05",
                "end_date": "2026-01-05",
                "timezone": "America/New_York",
                "data_provider": "alpaca",
                "alpaca_feed": "sip",
                "alpaca_adjustment": "raw",
                "data_root": str(tmp_path),
                "raw_data_dir": str(tmp_path / "raw"),
                "processed_data_dir": str(processed),
                "manifest_path": str(tmp_path / "manifest.csv"),
                "reports_dir": str(tmp_path / "reports"),
                "default_bar_timeframe": "1min",
                "log_level": "INFO",
                "demo_twap": {
                    "symbol": "TEST",
                    "trade_date": "2026-01-05",
                    "quantity": 300,
                    "start_time": "10:00",
                    "end_time": "10:03",
                    "max_bar_participation_rate": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )

    status = main(["simulate", "--strategy", "twap", "--config", str(config), "--json"])

    assert status == 0
    output = capsys.readouterr().out
    assert '"strategy": "twap"' in output
    assert '"filled_qty": 300' in output
