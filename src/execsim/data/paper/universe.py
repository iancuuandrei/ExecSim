"""Formation-period selection for the frozen 100-stock paper universe."""

from __future__ import annotations

import pandas as pd

from execsim.data.paper.manifests import stable_hash, write_json_atomic
from execsim.data.paper.schemas import PaperUniverseMember


def select_frozen_universe(
    candidates: pd.DataFrame, *, size: int = 100
) -> tuple[PaperUniverseMember, ...]:
    """Select eligible ordinary constituents by median daily dollar volume."""
    required = {
        "instrument_id",
        "symbol",
        "security_type",
        "in_sp500_on_formation_date",
        "median_price",
        "session_completeness",
        "median_daily_dollar_volume",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Universe candidates missing columns: {sorted(missing)}")
    eligible = candidates.loc[
        (candidates["security_type"] == "ordinary_common_stock")
        & candidates["in_sp500_on_formation_date"].astype(bool)
        & (candidates["median_price"] >= 5.0)
        & (candidates["session_completeness"] >= 0.95)
        & candidates["instrument_id"].astype(str).str.len().gt(0)
    ].copy()
    eligible = eligible.sort_values(
        ["median_daily_dollar_volume", "instrument_id"], ascending=[False, True], kind="stable"
    ).head(size)
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} eligible universe members; {size} required.")
    members = []
    for rank, row in enumerate(eligible.itertuples(index=False), start=1):
        members.append(
            PaperUniverseMember(
                rank=rank,
                instrument_id=str(row.instrument_id),
                symbol=str(row.symbol).upper(),
                median_price=float(row.median_price),
                session_completeness=float(row.session_completeness),
                median_daily_dollar_volume=float(row.median_daily_dollar_volume),
                liquidity_group=min(5, (rank - 1) * 5 // size + 1),
            )
        )
    return tuple(members)


def write_universe_manifest(
    members: tuple[PaperUniverseMember, ...],
    *,
    output: str,
    source_hashes: tuple[str, ...],
    symbol_history: tuple[dict[str, str], ...] = (),
    paper_config_hash: str = "",
) -> str:
    """Write the completed frozen universe with selection and source identities."""
    if len(members) != 100 or tuple(member.rank for member in members) != tuple(range(1, 101)):
        raise ValueError("Completed paper universe must contain ranks 1 through 100.")
    rows = [
        {
            "rank": member.rank,
            "instrument_id": member.instrument_id,
            "symbol": member.symbol,
            "median_price": member.median_price,
            "session_completeness": member.session_completeness,
            "median_daily_dollar_volume": member.median_daily_dollar_volume,
            "liquidity_group": member.liquidity_group,
        }
        for member in members
    ]
    stable = {
        "schema_version": "paper-universe-v1",
        "as_of": "2021-01-04",
        "formation_period": ["2021-01-04", "2021-12-31"],
        "target_size": 100,
        "source_hashes": list(source_hashes),
        "members": rows,
        "symbol_history": symbol_history,
        "paper_config_hash": paper_config_hash,
    }
    payload = {
        **stable,
        "status": "complete",
        "universe_id": "universe-" + stable_hash(stable)[:16],
    }
    return str(write_json_atomic(output, payload))
