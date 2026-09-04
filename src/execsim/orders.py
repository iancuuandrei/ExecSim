from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from numbers import Integral
from typing import Literal

OrderSide = Literal["buy", "sell"]
VALID_ORDER_SIDES = ("buy", "sell")


@dataclass(frozen=True, slots=True)
class ExecutionWindow:
    trade_date: date
    start_time: time
    end_time: time

    def __post_init__(self) -> None:
        _validate_date(self.trade_date, "trade_date")
        _validate_time(self.start_time, "start_time")
        _validate_time(self.end_time, "end_time")
        if self.start_time >= self.end_time:
            raise ValueError("Execution window start_time must be before end_time.")


@dataclass(frozen=True, slots=True)
class ParentOrder:
    symbol: str
    side: OrderSide
    quantity: int
    trade_date: date
    start_time: time
    end_time: time

    def __post_init__(self) -> None:
        symbol = _normalize_symbol(self.symbol)
        side = _normalize_side(self.side)
        quantity = _normalize_quantity(self.quantity)

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)

        ExecutionWindow(
            trade_date=self.trade_date,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    @property
    def execution_window(self) -> ExecutionWindow:
        return ExecutionWindow(
            trade_date=self.trade_date,
            start_time=self.start_time,
            end_time=self.end_time,
        )


def _normalize_symbol(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Parent order symbol must be a non-empty string.")
    return value.strip().upper()


def _normalize_side(value: str) -> OrderSide:
    if not isinstance(value, str):
        raise TypeError("Parent order side must be a string.")

    side = value.strip().lower()
    if side not in VALID_ORDER_SIDES:
        raise ValueError("Parent order side must be 'buy' or 'sell'.")

    return side  # type: ignore[return-value]


def _normalize_quantity(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Parent order quantity must be an integer share quantity.")

    quantity = int(value)
    if quantity <= 0:
        raise ValueError("Parent order quantity must be positive.")

    return quantity


def _validate_date(value: date, field_name: str) -> None:
    if not isinstance(value, date):
        raise TypeError(f"{field_name} must be a datetime.date.")


def _validate_time(value: time, field_name: str) -> None:
    if not isinstance(value, time):
        raise TypeError(f"{field_name} must be a datetime.time.")

    if value.tzinfo is not None:
        raise ValueError(f"{field_name} must be a local, timezone-naive time.")
