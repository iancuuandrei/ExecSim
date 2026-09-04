from __future__ import annotations

from execsim.policies.adaptive import AdaptiveMPCPolicy
from execsim.policies.oracle import DEPLOYABLE_POLICY_NAMES, OracleVwapPolicy
from execsim.policies.static import (
    AlmgrenChrissPolicy,
    ConstrainedOptimalPolicy,
    HistoricalVwapPolicy,
    PovPolicy,
    TwapPolicy,
)


def create_policy(
    name: str,
    *,
    risk_aversion: float = 0.0,
    half_spread: float = 0.0,
    temporary_impact: float = 0.1,
    volatility: float = 0.01,
    pov_target_rate: float = 0.05,
    tracking_penalty: float = 0.0,
    analytical_temporary_coefficient: float | None = None,
    allow_evaluation_only: bool = False,
) -> object:
    if name == "twap":
        return TwapPolicy()
    if name == "vwap":
        return HistoricalVwapPolicy()
    if name == "pov":
        return PovPolicy(pov_target_rate)
    if name == "almgren-chriss":
        coefficient = analytical_temporary_coefficient
        if coefficient is None:
            coefficient = max(temporary_impact / 100_000.0, 1e-12)
        return AlmgrenChrissPolicy(coefficient, volatility, risk_aversion)
    if name == "optimal":
        return ConstrainedOptimalPolicy(
            risk_aversion=risk_aversion,
            tracking_penalty=tracking_penalty,
            half_spread=half_spread,
            temporary_impact=max(temporary_impact, 1e-12),
            volatility=volatility,
        )
    if name == "mpc":
        return AdaptiveMPCPolicy(
            risk_aversion=risk_aversion,
            tracking_penalty=tracking_penalty,
            half_spread=half_spread,
            temporary_impact=max(temporary_impact, 1e-12),
            volatility=volatility,
        )
    if name == "oracle-vwap" and allow_evaluation_only:
        return OracleVwapPolicy()
    allowed = (*DEPLOYABLE_POLICY_NAMES, "oracle-vwap (explicit opt-in)")
    raise ValueError(f"Unknown or disallowed policy {name!r}; allowed={allowed}")
