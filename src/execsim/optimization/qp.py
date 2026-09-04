from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from execsim.optimization.integer import project_to_integer_capacities


@dataclass(frozen=True, slots=True)
class OptimalExecutionProblem:
    quantity: int
    forecast_volumes: NDArray[np.float64]
    forecast_volatilities: NDArray[np.float64]
    half_spreads: NDArray[np.float64]
    temporary_impacts: NDArray[np.float64]
    max_participation_rate: float
    risk_aversion: float = 0.0
    delta_t: float = 1.0
    tracking_penalty: float = 0.0
    forecast_weights: NDArray[np.float64] | None = None
    epsilon_volume: float = 1.0
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-7
    max_iterations: int = 20_000

    def __post_init__(self) -> None:
        vectors = (
            self.forecast_volumes,
            self.forecast_volatilities,
            self.half_spreads,
            self.temporary_impacts,
        )
        arrays = tuple(np.asarray(vector, dtype=float) for vector in vectors)
        if any(array.ndim != 1 for array in arrays) or not arrays[0].size:
            raise ValueError("Optimizer inputs must be non-empty vectors.")
        if len({array.size for array in arrays}) != 1:
            raise ValueError("Optimizer input vectors must have equal length.")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Optimizer inputs must be finite.")
        if np.any(arrays[0] < 0) or any(np.any(array < 0) for array in arrays[1:]):
            raise ValueError("Volumes, volatilities, spreads, and impacts must be non-negative.")
        if np.any(arrays[3] <= 0):
            raise ValueError("temporary_impacts must be strictly positive for a stable QP.")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("quantity must be a positive integer.")
        scalar_nonnegative = (self.risk_aversion, self.tracking_penalty)
        if any(not math.isfinite(value) or value < 0 for value in scalar_nonnegative):
            raise ValueError("Risk and tracking penalties must be finite and non-negative.")
        if not math.isfinite(self.max_participation_rate) or not (
            0 <= self.max_participation_rate <= 1
        ):
            raise ValueError("max_participation_rate must be in [0, 1].")
        if not math.isfinite(self.delta_t) or self.delta_t <= 0:
            raise ValueError("delta_t must be finite and positive.")
        if not math.isfinite(self.epsilon_volume) or self.epsilon_volume <= 0:
            raise ValueError("epsilon_volume must be finite and positive.")
        if self.forecast_weights is not None:
            weights = np.asarray(self.forecast_weights, dtype=float)
            if (
                weights.shape != arrays[0].shape
                or np.any(weights < 0)
                or not np.all(np.isfinite(weights))
            ):
                raise ValueError("forecast_weights must be a finite non-negative matching vector.")
            if not np.isclose(weights.sum(), 1.0, atol=1e-8):
                raise ValueError("forecast_weights must sum to one.")
        object.__setattr__(self, "forecast_volumes", arrays[0])
        object.__setattr__(self, "forecast_volatilities", arrays[1])
        object.__setattr__(self, "half_spreads", arrays[2])
        object.__setattr__(self, "temporary_impacts", arrays[3])


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    status: str
    status_value: int
    iterations: int
    primal_residual: float
    dual_residual: float
    solve_time_seconds: float
    objective_value: float
    minimum_eigenvalue: float
    warm_started: bool
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    continuous_quantities: NDArray[np.float64]
    integer_quantities: NDArray[np.int64]
    capacities: NDArray[np.int64]
    feasible_quantity: int
    predicted_capacity_shortfall: int
    diagnostics: SolverDiagnostics


def build_qp_matrices(
    problem: OptimalExecutionProblem,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], int]:
    volumes = problem.forecast_volumes
    n = len(volumes)
    capacities = np.floor(problem.max_participation_rate * volumes).astype(np.int64)
    feasible = min(problem.quantity, int(capacities.sum()))
    lower = _lower_triangular(n)
    risk_diag = np.diag(problem.forecast_volatilities**2 * problem.delta_t)
    temporary = np.diag(problem.temporary_impacts / np.maximum(volumes, problem.epsilon_volume))
    matrix = 2.0 * temporary
    linear = problem.half_spreads.astype(float).copy()

    if problem.risk_aversion:
        matrix += 2.0 * problem.risk_aversion * (lower.T @ risk_diag @ lower)
        linear -= (
            2.0 * problem.risk_aversion * problem.quantity * (lower.T @ risk_diag @ np.ones(n))
        )
    if problem.tracking_penalty:
        if problem.forecast_weights is None:
            raise ValueError("forecast_weights are required when tracking_penalty is positive.")
        target = max(feasible, 1)
        matrix += 2.0 * problem.tracking_penalty * np.eye(n) / target**2
        linear -= 2.0 * problem.tracking_penalty * problem.forecast_weights / target

    matrix = (matrix + matrix.T) / 2.0
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("QP objective matrix is not symmetric.")
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    if minimum_eigenvalue < -1e-10:
        raise ValueError(f"QP objective matrix is not positive semidefinite: {minimum_eigenvalue}")
    return matrix, linear, capacities, feasible


@lru_cache(maxsize=32)
def _lower_triangular(size: int) -> NDArray[np.float64]:
    """Cache the horizon-only cumulative-trade transformation."""
    matrix = np.tril(np.ones((size, size), dtype=np.float64))
    matrix.flags.writeable = False
    return matrix


def solve_optimal_execution(
    problem: OptimalExecutionProblem,
    *,
    warm_start: NDArray[np.float64] | None = None,
) -> OptimizationResult:
    try:
        import osqp
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install the 'optimization' extra to use optimal policies.") from exc

    matrix, linear, capacities, feasible = build_qp_matrices(problem)
    n = len(capacities)
    if feasible == 0:
        diagnostics = SolverDiagnostics(
            status="no_capacity",
            status_value=0,
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            solve_time_seconds=0.0,
            objective_value=0.0,
            minimum_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),
            warm_started=False,
            absolute_tolerance=problem.absolute_tolerance,
            relative_tolerance=problem.relative_tolerance,
        )
        zeros = np.zeros(n, dtype=float)
        return OptimizationResult(
            continuous_quantities=zeros,
            integer_quantities=zeros.astype(np.int64),
            capacities=capacities,
            feasible_quantity=0,
            predicted_capacity_shortfall=problem.quantity,
            diagnostics=diagnostics,
        )

    constraints = sparse.vstack([sparse.eye(n), np.ones((1, n))], format="csc")
    lower_bounds = np.concatenate([np.zeros(n), [float(feasible)]])
    upper_bounds = np.concatenate([capacities.astype(float), [float(feasible)]])
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(matrix),
        q=linear,
        A=constraints,
        l=lower_bounds,
        u=upper_bounds,
        verbose=False,
        eps_abs=problem.absolute_tolerance,
        eps_rel=problem.relative_tolerance,
        max_iter=problem.max_iterations,
        adaptive_rho=True,
        polishing=True,
        scaled_termination=True,
        check_termination=1,
    )
    did_warm_start = warm_start is not None
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=float)
        if warm.shape != (n,) or not np.all(np.isfinite(warm)):
            raise ValueError("warm_start must be a finite vector matching the horizon.")
        solver.warm_start(x=np.clip(warm, 0.0, capacities.astype(float)))

    started = perf_counter()
    try:
        solution = solver.solve(raise_error=True)
    except Exception as exc:
        raise RuntimeError(
            "Optimal execution solver failed before producing an acceptable solution: "
            f"max_iterations={problem.max_iterations}, "
            f"absolute_tolerance={problem.absolute_tolerance}, "
            f"relative_tolerance={problem.relative_tolerance}."
        ) from exc
    elapsed = perf_counter() - started
    info = solution.info
    status = str(info.status).lower()
    if status != "solved" or solution.x is None:
        raise RuntimeError(
            "Optimal execution solve was unreliable: "
            f"status={info.status}, primal_residual={info.prim_res}, dual_residual={info.dual_res}"
        )
    continuous = np.asarray(solution.x, dtype=float)
    if not np.isclose(continuous.sum(), feasible, atol=max(1e-5, problem.absolute_tolerance * 10)):
        raise RuntimeError("Optimal execution solution violates the completion constraint.")
    integer = project_to_integer_capacities(continuous, capacities, feasible, tolerance=1e-5)
    diagnostics = SolverDiagnostics(
        status=status,
        status_value=int(info.status_val),
        iterations=int(info.iter),
        primal_residual=float(info.prim_res),
        dual_residual=float(info.dual_res),
        solve_time_seconds=elapsed,
        objective_value=float(info.obj_val),
        minimum_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),
        warm_started=did_warm_start,
        absolute_tolerance=problem.absolute_tolerance,
        relative_tolerance=problem.relative_tolerance,
    )
    return OptimizationResult(
        continuous_quantities=continuous,
        integer_quantities=integer,
        capacities=capacities,
        feasible_quantity=feasible,
        predicted_capacity_shortfall=problem.quantity - feasible,
        diagnostics=diagnostics,
    )
