from importlib import import_module

import pytest

from tests._contracts import assert_optimal_solution, run_solver_preserving_input


@pytest.mark.parametrize("discount", [0.9, 0.99])
@pytest.mark.parametrize("name", ["aggregated_vi", "aggregated_qvi", "aggregated_mpi"])
def test_aggregated_solvers_recover_chain_values(chain, discount, name):
    solver_class = import_module(f"mdpforge.solvers.{name}").Solver
    solver = run_solver_preserving_input(solver_class, chain, discount, 1e-8)
    assert_optimal_solution(solver, chain, discount, 1e-8)
