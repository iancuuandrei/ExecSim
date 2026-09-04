from __future__ import annotations

import numpy as np
import pytest

from execsim.optimization import (
    OptimalExecutionProblem,
    OptimalExecutionWorkspace,
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


def test_structured_risk_matrix_matches_dense_reference() -> None:
    problem = _constant_problem(0.05)
    matrix, linear, _, _ = build_qp_matrices(problem)
    n = len(problem.forecast_volumes)
    lower = np.tril(np.ones((n, n)))
    risk_diagonal = np.diag(problem.forecast_volatilities**2 * problem.delta_t)
    temporary = np.diag(problem.temporary_impacts / np.maximum(problem.forecast_volumes, 1.0))
    expected_matrix = 2.0 * temporary + 2.0 * problem.risk_aversion * (
        lower.T @ risk_diagonal @ lower
    )
    expected_linear = problem.half_spreads - (
        2.0 * problem.risk_aversion * problem.quantity * (lower.T @ risk_diagonal @ np.ones(n))
    )

    assert matrix == pytest.approx(expected_matrix, abs=1e-15)
    assert linear == pytest.approx(expected_linear, abs=1e-15)


def test_workspace_reuses_horizon_without_changing_solution() -> None:
    problem = _constant_problem(0.05)
    reference = solve_optimal_execution(problem)
    workspace = OptimalExecutionWorkspace(8, validation_level="structural")
    first = workspace.solve(problem)
    second = workspace.solve(problem, warm_start=first.continuous_quantities)

    assert first.integer_quantities.tolist() == reference.integer_quantities.tolist()
    assert second.continuous_quantities == pytest.approx(reference.continuous_quantities, abs=1e-6)
    assert first.diagnostics.workspace_reused is False
    assert second.diagnostics.workspace_reused is True
    assert second.diagnostics.solver_setup_time_seconds == 0.0
    assert second.diagnostics.solver_update_time_seconds > 0.0


def test_structural_validation_does_not_call_eigendecomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_eigenvalue_check(*args: object, **kwargs: object) -> None:
        raise AssertionError("structural validation must not compute eigenvalues")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail_eigenvalue_check)
    result = OptimalExecutionWorkspace(8, validation_level="structural").solve(
        _constant_problem(0.05)
    )

    assert result.diagnostics.validation_level == "structural"
    assert result.diagnostics.minimum_eigenvalue > 0.0


def test_workspace_matches_standalone_solver_over_shrinking_horizons() -> None:
    workspace = OptimalExecutionWorkspace(8, validation_level="full")
    remaining = 80
    warm_start: np.ndarray | None = None

    for horizon in range(8, 0, -1):
        problem = OptimalExecutionProblem(
            quantity=remaining,
            forecast_volumes=np.linspace(900.0, 1_100.0, horizon),
            forecast_volatilities=np.linspace(0.08, 0.12, horizon),
            half_spreads=np.full(horizon, 0.01),
            temporary_impacts=np.full(horizon, 0.5),
            max_participation_rate=0.5,
            risk_aversion=0.03,
        )
        standalone = solve_optimal_execution(problem, warm_start=warm_start)
        reused = workspace.solve(problem, warm_start=warm_start)

        assert reused.continuous_quantities == pytest.approx(
            standalone.continuous_quantities, abs=1e-8
        )
        assert reused.integer_quantities.tolist() == standalone.integer_quantities.tolist()
        action = int(reused.integer_quantities[0])
        remaining -= action
        warm_start = reused.continuous_quantities[1:].copy() if horizon > 1 else None
