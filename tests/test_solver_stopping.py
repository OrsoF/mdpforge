from importlib import import_module

import numpy as np
import pytest

from mdpforge import Benchmark
from tests._contracts import assert_optimal_solution, matrix_model


@pytest.fixture(params=["mdpforge_vi", "mdpforge_mpi"])
def solver_class(request):
    return import_module(f"mdpforge.solvers.{request.param}").Solver


@pytest.fixture
def recurrent_model(monkeypatch):
    monkeypatch.setattr(
        np.random, "randint", lambda *args, **kwargs: np.zeros(2, dtype=int)
    )
    return matrix_model(
        "stopping_recurrent", [np.eye(2), np.eye(2)], [[1.0, 0.0], [2.0, 0.0]]
    )


@pytest.mark.parametrize("max_iterations", [0, -1, True, 1.5, np.inf, None])
def test_solver_rejects_invalid_iteration_limit(
    solver_class, recurrent_model, max_iterations
):
    with pytest.raises(ValueError, match="max_iterations"):
        solver_class(recurrent_model, 0.9, max_iterations=max_iterations)


def test_solver_iteration_limit_and_rerun(solver_class, recurrent_model):
    solver = solver_class(recurrent_model, 0.9, final_precision=1e-4, max_iterations=1)
    with pytest.raises(RuntimeError, match="max_iterations after 1 iterations"):
        solver.run()

    assert solver.stop_reason == "max_iterations"
    assert solver.iterations == 1
    assert np.isfinite(solver.runtime) and solver.runtime >= 0

    solver = solver_class(
        recurrent_model, 0.9, final_precision=1e-4, max_iterations=1_000
    )
    solver.run()

    assert solver.stop_reason == "converged"
    assert 1 < solver.iterations <= solver.max_iterations
    assert_optimal_solution(solver, recurrent_model, 0.9, 1e-4)
    solver.run()
    assert_optimal_solution(solver, recurrent_model, 0.9, 1e-4)


def test_solver_accepts_convergence_on_last_iteration(solver_class, recurrent_model):
    recurrent_model.reward_matrix[:] = 0
    solver = solver_class(recurrent_model, 0.9, max_iterations=1)
    solver.run()

    assert solver.stop_reason == "converged"
    assert solver.iterations == 1
    assert_optimal_solution(solver, recurrent_model, 0.9, 1e-3)


@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_solver_rejects_non_finite_iterates(solver_class, recurrent_model, non_finite):
    recurrent_model.reward_matrix[0, :] = non_finite
    solver = solver_class(recurrent_model, 0.9, max_iterations=1)

    with np.errstate(invalid="ignore", over="ignore"):
        with pytest.raises(RuntimeError, match="non_finite"):
            solver.run()

    assert solver.stop_reason == "non_finite"
    assert solver.iterations == 1
    assert np.isfinite(solver.runtime) and solver.runtime >= 0


def test_mpi_rejects_non_finite_values_in_non_greedy_actions(recurrent_model):
    from mdpforge.solvers.mdpforge_mpi import Solver

    recurrent_model.reward_matrix[:, 1] = -np.inf
    solver = Solver(recurrent_model, 0.9, max_iterations=1)
    with pytest.raises(RuntimeError, match="non_finite"):
        solver.run()

    assert solver.stop_reason == "non_finite"
    assert solver.iterations == 1


def test_solver_reports_numerically_unattainable_precision(
    solver_class, recurrent_model
):
    # Scaling the smallest positive precision by 1 - gamma rounds to zero.
    precision = np.nextafter(0.0, 1.0)
    solver = solver_class(
        recurrent_model, 0.5, final_precision=precision, max_iterations=100
    )
    with pytest.raises(RuntimeError, match="stagnation"):
        solver.run()

    assert solver.stop_reason == "stagnation"
    assert 1 <= solver.iterations < solver.max_iterations


def test_benchmark_forwards_iteration_limit_and_records_failure(
    solver_class, recurrent_model
):
    bench = Benchmark(default_solvers=False).add_mdp(recurrent_model)
    bench.add_solver("bounded", solve_function=solver_class, max_iterations=1)
    result = bench.run(discount=0.9, precision=1e-4, repeats=1)

    row = result.rows[0]
    assert row["status"] == "error"
    assert "max_iterations" in row["error"]
    assert np.isfinite(row["runtime"])
