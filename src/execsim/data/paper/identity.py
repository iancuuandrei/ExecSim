"""Stable instrument and provider-symbol history contracts."""

from __future__ import annotations

from datetime import date
from itertools import pairwise

from execsim.data.paper.schemas import InstrumentSymbolInterval


def validate_symbol_history(intervals: tuple[InstrumentSymbolInterval, ...]) -> None:
    """Reject overlapping sourced aliases; requested-period code rejects trading-day gaps."""
    grouped: dict[str, list[InstrumentSymbolInterval]] = {}
    for interval in intervals:
        grouped.setdefault(interval.instrument_id, []).append(interval)
    for instrument_id, records in grouped.items():
        ordered = sorted(records, key=lambda item: (item.start, item.end, item.symbol))
        for prior, current in pairwise(ordered):
            if current.start <= prior.end:
                raise ValueError(f"Overlapping ticker history for {instrument_id}.")


def resolve_provider_symbol(
    intervals: tuple[InstrumentSymbolInterval, ...], instrument_id: str, on_date: date
) -> str:
    """Resolve an explicitly sourced ticker or report the acquisition stage blocked."""
    matches = [
        item
        for item in intervals
        if item.instrument_id == instrument_id and item.start <= on_date <= item.end
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"BLOCKED: no unique sourced provider symbol for {instrument_id} on {on_date}."
        )
    return matches[0].symbol.upper()
