from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from execsim.config import ExecSimConfig


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def load_raw_symbol_bars(config: ExecSimConfig, symbol: str) -> pd.DataFrame:
    path = config.raw_symbol_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Raw parquet not found for {symbol}: {path}")
    return load_parquet(path)


def load_processed_symbol_bars(config: ExecSimConfig, symbol: str) -> pd.DataFrame:
    path = config.processed_symbol_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet not found for {symbol}: {path}")
    return load_parquet(path)


def iter_processed_symbol_paths(config: ExecSimConfig) -> Iterator[tuple[str, Path]]:
    for symbol in config.symbols:
        yield symbol, config.processed_symbol_path(symbol)
