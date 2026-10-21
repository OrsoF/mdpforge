import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.benchmark import Benchmark
from mdpforge.core.solver import GenericSolver
from tests._contracts import assert_optimal_solution


class ChainSolver(GenericSolver):
    def _solve(self):
        # Known solution of the deterministic chain fixture.
        return [2 * self.discount, 2, 0]


@pytest.mark.parametrize("discount", [0.5, 0.9, 0.99])
def test_solver_only_needs_algorithm(chain, discount):
    solver = ChainSolver(chain, discount)
    assert solver.model is chain
    assert solver.discount == discount
    assert solver.final_precision == 1e-3
    assert solver.name == "ChainSolver"

    solver.run()
    assert isinstance(solver.value, np.ndarray)
    assert_allclose(solver.value, [2 * discount, 2, 0], rtol=0, atol=0)
    assert solver.policy is None
    assert np.isfinite(solver.runtime) and solver.runtime >= 0
    assert solver.bellman_residual() == pytest.approx(0)
    assert_optimal_solution(solver, chain, discount, 1e-3)


@pytest.mark.parametrize("discount", [-0.1, 0, 1, 1.1, np.nan])
def test_generic_solver_rejects_invalid_discount(chain, discount):
    with pytest.raises(AssertionError, match="discount"):
        ChainSolver(chain, discount)


@pytest.mark.parametrize("precision", [-1, 0, np.nan, np.inf])
def test_generic_solver_rejects_invalid_precision(chain, precision):
    with pytest.raises(AssertionError, match="final_precision"):
        ChainSolver(chain, 0.9, final_precision=precision)


def test_generic_solver_requires_algorithm(chain):
    with pytest.raises(NotImplementedError, match="implement _solve"):
        GenericSolver(chain, 0.9).run()


def test_custom_options_and_policy_work_with_benchmark(chain):
    class PolicySolver(ChainSolver):
        def __init__(self, model, discount, final_precision=1e-3, *, policy_action):
            super().__init__(model, discount, final_precision)
            self.policy_action = policy_action

        def _solve(self):
            self.policy = np.full(self.model.state_dim, self.policy_action, dtype=int)
            return super()._solve()

    solver = PolicySolver(chain, 0.9, final_precision=1e-4, policy_action=0)
    solver.run()
    assert solver.final_precision == 1e-4
    assert_allclose(solver.policy, [0, 0, 0], rtol=0, atol=0)

    bench = Benchmark().add_mdp(chain)
    bench.add_solver("PolicySolver", solve_function=PolicySolver, policy_action=0)
    results = bench.run(0.9, precision=1e-4, repeats=1, seed=0)
    assert len(results.rows) == 1
    assert results.rows[0]["solver"] == "PolicySolver"
    assert results.rows[0]["status"] == "success", results.rows[0]["error"]
    assert results.rows[0]["error_bound"] <= 1e-4


def test_vi_uses_initial_value_without_modifying_it(chain):
    from mdpforge.solvers import mdpforge_vi

    initial_value = np.array([1.8, 2.0, 0.0])
    initial_value.flags.writeable = False
    solver = mdpforge_vi.Solver(
        chain, 0.9, final_precision=1e-4, initial_value=initial_value
    )
    solver.run()

    assert_allclose(solver.value, [1.8, 2.0, 0.0], atol=1e-12, rtol=0)
    assert_allclose(initial_value, [1.8, 2.0, 0.0], atol=0, rtol=0)
    assert solver.bellman_residual() <= 1e-5


@pytest.mark.parametrize(
    "initial_value",
    [np.zeros(2), np.zeros((3, 1)), [0, np.nan, 0], [0, np.inf, 0]],
)
def test_vi_rejects_invalid_initial_value(chain, initial_value):
    from mdpforge.solvers.mdpforge_vi import Solver

    with pytest.raises(ValueError, match="initial_value"):
        Solver(chain, 0.9, initial_value=initial_value)
