from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from execsim.orders import ParentOrder


@dataclass(frozen=True, slots=True)
class TwapStrategy:
    def generate_schedule(self, parent_order: ParentOrder, bars: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in bars.columns:
            raise ValueError("TWAP scheduling requires a timestamp column.")
        if bars.empty:
            raise ValueError("TWAP scheduling requires at least one bar.")

        n_bars = len(bars)
        base_quantity = parent_order.quantity // n_bars
        remainder = parent_order.quantity % n_bars
        scheduled_quantities = [
            base_quantity + (1 if index < remainder else 0)
            for index in range(n_bars)
        ]

        return pd.DataFrame(
            {
                "timestamp": bars["timestamp"].reset_index(drop=True),
                "scheduled_qty": scheduled_quantities,
            }
        )
