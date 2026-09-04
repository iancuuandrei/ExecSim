"""Convex execution optimization and analytical references."""

from execsim.optimization.analytical import almgren_chriss_continuous_schedule
from execsim.optimization.integer import project_to_integer_capacities
from execsim.optimization.qp import (
    OptimalExecutionProblem,
    OptimizationResult,
    SolverDiagnostics,
    build_qp_matrices,
    solve_optimal_execution,
)

__all__ = [
    "OptimalExecutionProblem",
    "OptimizationResult",
    "SolverDiagnostics",
    "almgren_chriss_continuous_schedule",
    "build_qp_matrices",
    "project_to_integer_capacities",
    "solve_optimal_execution",
]
