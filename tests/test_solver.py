import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.benchmark import Benchmark
from mdpforge.core.solver import GenericSolver


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

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver(PolicySolver, policy_action=0)
    results = bench.run(0.9, precision=1e-4, repeats=1, seed=0)
    assert len(results) == 1
    assert results[0]["solver"] == "PolicySolver"
    assert results[0]["status"] == "success", results[0]["error"]
    assert results[0]["error_bound"] <= 1e-4
