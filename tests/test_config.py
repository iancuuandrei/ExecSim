from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from execsim.config import DEFAULT_CONFIG_PATH, load_config, load_project_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads() -> None:
    config = load_config()

    assert DEFAULT_CONFIG_PATH.exists()
    assert config.project_name == "execution-cost-sim"
    assert config.symbols == ("AAPL", "MSFT", "NVDA")
    assert config.start_date.isoformat() == "2026-03-16"
    assert config.end_date.isoformat() == "2026-04-14"
    assert config.timezone == "America/New_York"
    assert config.data_provider == "alpaca"
    assert config.alpaca_feed == "sip"
    assert config.alpaca_adjustment == "raw"
    assert config.default_bar_timeframe == "1min"
    assert config.demo_twap.symbol == "AAPL"
    assert config.demo_twap.trade_date.isoformat() == "2026-03-16"
    assert config.demo_twap.side == "buy"
    assert config.demo_twap.quantity == 5000
    assert config.demo_twap.start_time.strftime("%H:%M") == "10:00"
    assert config.demo_twap.end_time.strftime("%H:%M") == "10:30"
    assert config.demo_twap.max_bar_participation_rate == 0.05


def test_explicit_config_path_loads() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert config.data_root == "data"
    assert config.raw_data_dir == "data/raw/alpaca/minute_bars"
    assert config.processed_data_dir == "data/processed/alpaca/minute_bars"
    assert config.manifest_path == "data/processed/alpaca/minute_bars/manifest.csv"
    assert config.reports_dir == "reports"


def test_load_project_dotenv_uses_python_dotenv() -> None:
    dotenv_path = REPO_ROOT / ".env"

    with patch("execsim.config.load_dotenv", return_value=True) as mock_load_dotenv:
        loaded = load_project_dotenv(dotenv_path)

    assert loaded is True
    mock_load_dotenv.assert_called_once_with(dotenv_path=dotenv_path, override=True)
