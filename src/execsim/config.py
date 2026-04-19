from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"


@dataclass(slots=True)
class TwapSimulationDefaults:
    symbol: str
    trade_date: date
    side: str
    quantity: int
    start_time: time
    end_time: time
    max_bar_participation_rate: float

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        self.side = self.side.strip().lower()

        if not self.symbol:
            raise ValueError("demo_twap.symbol must be a non-empty string.")
        if self.side not in {"buy", "sell"}:
            raise ValueError("demo_twap.side must be 'buy' or 'sell'.")
        if self.quantity <= 0:
            raise ValueError("demo_twap.quantity must be positive.")
        if self.start_time >= self.end_time:
            raise ValueError("demo_twap.start_time must be before end_time.")
        if (
            self.max_bar_participation_rate < 0
            or self.max_bar_participation_rate > 1
        ):
            raise ValueError(
                "demo_twap.max_bar_participation_rate must be between 0 and 1."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date.isoformat(),
            "side": self.side,
            "quantity": self.quantity,
            "start_time": self.start_time.isoformat(timespec="minutes"),
            "end_time": self.end_time.isoformat(timespec="minutes"),
            "max_bar_participation_rate": self.max_bar_participation_rate,
        }


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
    demo_twap: TwapSimulationDefaults

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ExecSimConfig":
        symbols = _read_required_symbols(mapping, "symbols")
        start_date = _read_required_date(mapping, "start_date")
        end_date = _read_required_date(mapping, "end_date")
        config = cls(
            project_name=_read_required_string(mapping, "project_name"),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
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
            demo_twap=_read_twap_simulation_defaults(
                mapping=mapping,
                symbols=symbols,
                default_trade_date=start_date,
            ),
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
            "demo_twap": self.demo_twap.to_dict(),
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

        if self.demo_twap.symbol not in self.symbols:
            raise ValueError("demo_twap.symbol must be included in symbols.")

        if not self.start_date <= self.demo_twap.trade_date <= self.end_date:
            raise ValueError("demo_twap.trade_date must be within the configured date range.")


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


def _read_twap_simulation_defaults(
    mapping: Mapping[str, Any],
    symbols: tuple[str, ...],
    default_trade_date: date,
) -> TwapSimulationDefaults:
    raw_defaults = mapping.get("demo_twap", {})
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, Mapping):
        raise ValueError("demo_twap must be a mapping when provided.")

    return TwapSimulationDefaults(
        symbol=_read_optional_string(raw_defaults, "symbol", symbols[0]),
        trade_date=_read_optional_date(raw_defaults, "trade_date", default_trade_date),
        side=_read_optional_string(raw_defaults, "side", "buy"),
        quantity=_read_optional_positive_int(raw_defaults, "quantity", 1000),
        start_time=_read_optional_time(raw_defaults, "start_time", time(10, 0)),
        end_time=_read_optional_time(raw_defaults, "end_time", time(11, 0)),
        max_bar_participation_rate=_read_optional_participation_rate(
            raw_defaults,
            "max_bar_participation_rate",
            0.05,
        ),
    )


def _read_optional_string(
    mapping: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid string config value: {key}")
    return value


def _read_optional_date(
    mapping: Mapping[str, Any],
    key: str,
    default: date,
) -> date:
    value = mapping.get(key, default)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date for config value: {key}") from exc
    raise ValueError(f"Invalid date config value: {key}")


def _read_optional_time(
    mapping: Mapping[str, Any],
    key: str,
    default: time,
) -> time:
    value = mapping.get(key, default)
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        try:
            parsed_time = time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO time for config value: {key}") from exc
        if parsed_time.tzinfo is not None:
            raise ValueError(f"Config time value {key} must be timezone-naive.")
        return parsed_time
    raise ValueError(f"Invalid time config value: {key}")


def _read_optional_positive_int(
    mapping: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid integer config value: {key}")
    if value <= 0:
        raise ValueError(f"Config value {key} must be positive.")
    return value


def _read_optional_participation_rate(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Invalid participation-rate config value: {key}")

    rate = float(value)
    if rate < 0 or rate > 1:
        raise ValueError(f"Config value {key} must be between 0 and 1.")
    return rate
