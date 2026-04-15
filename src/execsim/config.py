from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


@dataclass(slots=True)
class ExecSimConfig:
    project_name: str
    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    timezone: str
    data_provider: str
    alpaca_feed: str
    alpaca_adjustment: str
    data_root: str
    raw_data_dir: str
    processed_data_dir: str
    manifest_path: str
    reports_dir: str
    default_bar_timeframe: str
    log_level: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ExecSimConfig":
        config = cls(
            project_name=_read_required_string(mapping, "project_name"),
            symbols=_read_required_symbols(mapping, "symbols"),
            start_date=_read_required_date(mapping, "start_date"),
            end_date=_read_required_date(mapping, "end_date"),
            timezone=_read_required_string(mapping, "timezone"),
            data_provider=_read_required_string(mapping, "data_provider"),
            alpaca_feed=_read_required_string(mapping, "alpaca_feed"),
            alpaca_adjustment=_read_required_string(mapping, "alpaca_adjustment"),
            data_root=_read_required_string(mapping, "data_root"),
            raw_data_dir=_read_required_string(mapping, "raw_data_dir"),
            processed_data_dir=_read_required_string(mapping, "processed_data_dir"),
            manifest_path=_read_required_string(mapping, "manifest_path"),
            reports_dir=_read_required_string(mapping, "reports_dir"),
            default_bar_timeframe=_read_required_string(mapping, "default_bar_timeframe"),
            log_level=_read_required_string(mapping, "log_level"),
        )
        config._validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "symbols": list(self.symbols),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timezone": self.timezone,
            "data_provider": self.data_provider,
            "alpaca_feed": self.alpaca_feed,
            "alpaca_adjustment": self.alpaca_adjustment,
            "data_root": self.data_root,
            "raw_data_dir": self.raw_data_dir,
            "processed_data_dir": self.processed_data_dir,
            "manifest_path": self.manifest_path,
            "reports_dir": self.reports_dir,
            "default_bar_timeframe": self.default_bar_timeframe,
            "log_level": self.log_level,
        }

    @property
    def market_timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def resolved_data_root(self) -> Path:
        return _resolve_project_path(self.data_root)

    @property
    def resolved_raw_data_dir(self) -> Path:
        return _resolve_project_path(self.raw_data_dir)

    @property
    def resolved_processed_data_dir(self) -> Path:
        return _resolve_project_path(self.processed_data_dir)

    @property
    def resolved_manifest_path(self) -> Path:
        return _resolve_project_path(self.manifest_path)

    def raw_symbol_path(self, symbol: str) -> Path:
        return self.resolved_raw_data_dir / f"{symbol.upper()}.parquet"

    def processed_symbol_path(self, symbol: str) -> Path:
        return self.resolved_processed_data_dir / f"{symbol.upper()}.parquet"

    def _validate(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("Config start_date must be on or before end_date.")

        if self.default_bar_timeframe != "1min":
            raise ValueError("This iteration only supports default_bar_timeframe=1min.")

        if self.data_provider.lower() != "alpaca":
            raise ValueError("This iteration only supports data_provider=alpaca.")


def load_config(path: str | Path | None = None) -> ExecSimConfig:
    config_path = _resolve_config_path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, Mapping):
        raise TypeError(f"Config at {config_path} must contain a mapping.")

    return ExecSimConfig.from_mapping(raw_config)


def load_project_dotenv(path: str | Path | None = None) -> bool:
    dotenv_path = DEFAULT_DOTENV_PATH if path is None else Path(path)
    return load_dotenv(dotenv_path=dotenv_path, override=True)


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_CONFIG_PATH

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    return PROJECT_ROOT / candidate


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    return PROJECT_ROOT / candidate


def _read_required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required config value: {key}")
    return value


def _read_required_symbols(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)

    if isinstance(value, str):
        raw_symbols = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        raw_symbols = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    else:
        raw_symbols = []

    if not raw_symbols:
        raise ValueError(f"Missing required config value: {key}")

    return tuple(symbol.upper() for symbol in raw_symbols)


def _read_required_date(mapping: Mapping[str, Any], key: str) -> date:
    value = mapping.get(key)

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date for config value: {key}") from exc

    raise ValueError(f"Missing required config value: {key}")
