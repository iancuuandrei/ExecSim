from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, time
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from execsim.costs import CostParameter, LinearTemporaryImpactModel, ParameterProvenance
from execsim.forecasting import HistoricalProfileForecaster, RealizedVolumeOracleForecaster
from execsim.orders import OrderSide, ParentOrder
from execsim.policies import (
    ExecutionConstraints,
    create_policy,
)
from execsim.reporting.outputs import write_research_outputs
from execsim.simulator import simulate_policy


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    symbols: tuple[str, ...]
    trade_dates: tuple[date, ...]
    quantities: tuple[int, ...]
    sides: tuple[str, ...] = ("buy",)
    strategies: tuple[str, ...] = ("twap", "vwap", "pov", "almgren-chriss", "optimal", "mpc")
    start_time: time = time(10, 0)
    end_time: time = time(11, 0)
    planned_participation_rate: float = 0.10
    hard_participation_rate: float = 0.10
    pov_target_rate: float = 0.05
    half_spread: float = 0.01
    temporary_impacts: tuple[float, ...] = (0.10,)
    volatility: float = 0.01
    risk_aversions: tuple[float, ...] = (0.0, 0.01)
    profile_estimator: str = "mean"
    profile_lookback_sessions: int = 20
    seed: int = 17
    include_oracle: bool = False

    def __post_init__(self) -> None:
        if not self.symbols or not self.trade_dates or not self.quantities or not self.strategies:
            raise ValueError("Experiment dimensions must be non-empty.")
        if any(quantity <= 0 for quantity in self.quantities):
            raise ValueError("Experiment quantities must be positive.")
        if any(side not in {"buy", "sell"} for side in self.sides):
            raise ValueError("Experiment sides must be buy or sell.")
        if self.start_time >= self.end_time:
            raise ValueError("Experiment start_time must precede end_time.")
        allowed = {"twap", "vwap", "pov", "almgren-chriss", "optimal", "mpc", "oracle-vwap"}
        unknown = set(self.strategies).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown experiment strategies: {sorted(unknown)}")
        if "oracle-vwap" in self.strategies and not self.include_oracle:
            raise ValueError("oracle-vwap requires include_oracle=true.")

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["trade_dates"] = [value.isoformat() for value in self.trade_dates]
        payload["start_time"] = self.start_time.isoformat(timespec="minutes")
        payload["end_time"] = self.end_time.isoformat(timespec="minutes")
        return payload


@dataclass(frozen=True, slots=True)
class ExperimentOutputs:
    run_id: str
    output_dir: Path
    results: pd.DataFrame
    decision_trace: pd.DataFrame
    aggregate: pd.DataFrame
    paired: pd.DataFrame
    paths: dict[str, Path]


@dataclass(slots=True)
class ExperimentRunner:
    spec: ExperimentSpec
    bars_by_symbol: dict[str, pd.DataFrame]
    reports_root: Path = Path("reports/runs")

    def run(self, *, write_outputs: bool = True) -> ExperimentOutputs:
        normalized = {
            symbol.upper(): self._validate_symbol_bars(symbol, bars)
            for symbol, bars in self.bars_by_symbol.items()
        }
        run_id = _stable_run_id(self.spec)
        rows: list[dict[str, object]] = []
        traces: list[pd.DataFrame] = []
        sequence = product(
            sorted(self.spec.symbols),
            sorted(self.spec.trade_dates),
            sorted(self.spec.sides),
            sorted(self.spec.quantities),
            sorted(self.spec.temporary_impacts),
            sorted(self.spec.risk_aversions),
            self.spec.strategies,
        )
        for run_number, (symbol, trade_date, side, quantity, impact, risk, strategy) in enumerate(
            sequence, start=1
        ):
            symbol = symbol.upper()
            all_bars = normalized[symbol]
            day_bars = all_bars.loc[all_bars["timestamp"].dt.date == trade_date].copy()
            if day_bars.empty:
                raise ValueError(f"No bars for experiment unit {symbol} {trade_date}.")
            provider: Any = None
            if strategy in {"vwap", "optimal", "mpc"}:
                provider = HistoricalProfileForecaster(
                    all_bars,
                    estimator=self.spec.profile_estimator,  # type: ignore[arg-type]
                    lookback_sessions=self.spec.profile_lookback_sessions,
                )
            elif strategy == "oracle-vwap":
                provider = RealizedVolumeOracleForecaster(day_bars)
            order = ParentOrder(
                symbol,
                cast(OrderSide, side),
                quantity,
                trade_date,
                self.spec.start_time,
                self.spec.end_time,
            )
            policy = create_policy(
                strategy,
                risk_aversion=risk,
                half_spread=self.spec.half_spread,
                temporary_impact=impact,
                volatility=self.spec.volatility,
                pov_target_rate=self.spec.pov_target_rate,
                allow_evaluation_only=self.spec.include_oracle,
            )
            constraints = ExecutionConstraints(
                self.spec.planned_participation_rate, self.spec.hard_participation_rate
            )
            cost_model = LinearTemporaryImpactModel(
                half_spread=CostParameter(
                    self.spec.half_spread, ParameterProvenance.ASSUMED, "research assumption"
                ),
                temporary_impact=CostParameter(
                    impact, ParameterProvenance.ASSUMED, "currency/share at 100% participation"
                ),
            )
            result = simulate_policy(
                parent_order=order,
                bars=day_bars,
                policy=policy,
                constraints=constraints,
                cost_model=cost_model,
                forecast_provider=provider,
            )
            row = asdict(result.summary)
            row.update(
                {
                    "run_number": run_number,
                    "run_id": run_id,
                    "trade_date": trade_date.isoformat(),
                    "quantity": quantity,
                    "start_time": self.spec.start_time.isoformat(timespec="minutes"),
                    "end_time": self.spec.end_time.isoformat(timespec="minutes"),
                    "planned_participation_rate": self.spec.planned_participation_rate,
                    "hard_participation_rate": self.spec.hard_participation_rate,
                    "risk_aversion": risk,
                    "temporary_impact": impact,
                    "half_spread": self.spec.half_spread,
                    "session_total_volume": float(day_bars["volume"].sum()),
                    "session_realized_volatility": _session_volatility(day_bars),
                    "evaluation_only": strategy.startswith("oracle"),
                }
            )
            rows.append(row)
            if not result.decision_trace.empty:
                trace = result.decision_trace.copy()
                trace.insert(0, "run_number", run_number)
                trace.insert(0, "run_id", run_id)
                trace.insert(2, "symbol", symbol)
                trace.insert(3, "trade_date", trade_date.isoformat())
                trace.insert(4, "strategy", strategy)
                traces.append(trace)
        results = pd.DataFrame(rows).sort_values("run_number", kind="stable").reset_index(drop=True)
        results = _add_regimes(results)
        decision_trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
        output_dir = self.reports_root / run_id
        paths, aggregate, paired = write_research_outputs(
            results=results,
            decision_trace=decision_trace,
            spec=self.spec.to_jsonable(),
            output_dir=output_dir,
            seed=self.spec.seed,
            write_files=write_outputs,
        )
        return ExperimentOutputs(
            run_id, output_dir, results, decision_trace, aggregate, paired, paths
        )

    @staticmethod
    def _validate_symbol_bars(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        prepared = bars.copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
        if prepared["timestamp"].dt.tz is None:
            raise ValueError(f"Bars for {symbol} must be timezone-aware.")
        return prepared.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _stable_run_id(spec: ExperimentSpec) -> str:
    encoded = json.dumps(spec.to_jsonable(), sort_keys=True, separators=(",", ":")).encode()
    return f"run-{hashlib.sha256(encoded).hexdigest()[:12]}"


def _session_volatility(bars: pd.DataFrame) -> float:
    prices = pd.to_numeric(bars["vwap"], errors="coerce").fillna(bars["close"])
    returns = np.log(prices).diff().dropna()
    return float(returns.std(ddof=0)) if len(returns) else 0.0


def _add_regimes(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    volume_median = output["session_total_volume"].median()
    volatility_median = output["session_realized_volatility"].median()
    output["liquidity_regime"] = np.where(
        output["session_total_volume"] >= volume_median, "high", "low"
    )
    output["volatility_regime"] = np.where(
        output["session_realized_volatility"] >= volatility_median, "high", "low"
    )
    return output


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None
