from __future__ import annotations

from typing import Protocol

import pandas as pd

from execsim.orders import ParentOrder


class SchedulingStrategy(Protocol):
    def generate_schedule(self, parent_order: ParentOrder, bars: pd.DataFrame) -> pd.DataFrame:
        """Return one scheduled child quantity per input bar."""
