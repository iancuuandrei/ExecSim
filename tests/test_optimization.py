from __future__ import annotations

import numpy as np
import pytest

from execsim.optimization import (
    OptimalExecutionProblem,
    almgren_chriss_continuous_schedule,
    build_qp_matrices,
    project_to_integer_capacities,
    solve_optimal_execution,
)


def _constant_problem(risk_aversion: float) -> OptimalExecutionProblem:
    n = 8
    return OptimalExecutionProblem(
        quantity=80,
        forecast_volumes=np.full(n, 1_000.0),
        forecast_volatilities=np.full(n, 0.1),
        half_spreads=np.zeros(n),
        temporary_impacts=np.ones(n),
        max_participation_rate=1.0,
        risk_aversion=risk_aversion,
    )


def test_qp_matrix_is_symmetric_psd_and_risk_neutral_is_equal_rate() -> None:
    problem = _constant_problem(0.0)
    matrix, _, _, _ = build_qp_matrices(problem)
    result = solve_optimal_execution(problem)

    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() >= 0
    assert result.integer_quantities.tolist() == [10] * 8


def test_qp_matches_analytical_reference_and_more_risk_frontloads() -> None:
    risk_aversion = 0.05
    problem = _constant_problem(risk_aversion)
    numerical = solve_optimal_execution(problem).continuous_quantities
    analytical = almgren_chriss_continuous_schedule(
        80,
        8,
        temporary_quadratic_coefficient=1.0 / 1_000.0,
        volatility=0.1,
        risk_aversion=risk_aversion,
    )
    high_risk = solve_optimal_execution(_constant_problem(0.2)).continuous_quantities

    assert numerical == pytest.approx(analytical, rel=1e-5, abs=1e-5)
    assert high_risk[0] > numerical[0] > 10


def test_infeasible_completion_and_integer_projection_are_explicit() -> None:
    problem = OptimalExecutionProblem(
        quantity=20,
        forecast_volumes=np.array([10.0, 10.0, 10.0]),
        forecast_volatilities=np.zeros(3),
        half_spreads=np.zeros(3),
        temporary_impacts=np.ones(3),
        max_participation_rate=0.1,
    )
    result = solve_optimal_execution(problem)

    assert result.capacities.tolist() == [1, 1, 1]
    assert result.feasible_quantity == 3
    assert result.predicted_capacity_shortfall == 17
    assert result.integer_quantities.tolist() == [1, 1, 1]


def test_largest_remainder_projection_preserves_caps_and_ties_by_index() -> None:
    rounded = project_to_integer_capacities([1.5, 1.5, 1.0], [2, 2, 1], 4)

    assert rounded.tolist() == [2, 1, 1]
    assert rounded.sum() == 4


def test_warm_and_cold_solutions_agree() -> None:
    problem = _constant_problem(0.05)
    cold = solve_optimal_execution(problem)
    warm = solve_optimal_execution(problem, warm_start=cold.continuous_quantities)

    assert warm.continuous_quantities == pytest.approx(cold.continuous_quantities, abs=1e-6)
    assert warm.diagnostics.warm_started is True
