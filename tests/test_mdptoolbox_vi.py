"""Public API delegation and numerical contracts for MDPToolbox VI."""

from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from mdpforge import Benchmark
from mdpforge.core.operators import optimal_bellman_operator


@pytest.mark.parametrize("discount", [0.5, 0.99, 0.999])
def test_native_vi_accuracy_and_preservation(discount):
    pytest.importorskip("mdptoolbox")
    from mdpforge.solvers.mdptoolbox_vi import Solver

    transition = np.array([[0.8, 0.2], [0.1, 0.9]])
    reward = np.array([[1.0, 0.0], [2.0, 0.0]])
    model = SimpleNamespace(
        name="native_vi_regression",
        state_dim=2,
        action_dim=2,
        transition_matrix=[csr_matrix(transition) for _ in range(2)],
        reward_matrix=reward,
    )
    reference = deepcopy(model)
    # Action zero is optimal; independently solve its discounted linear system.
    expected = np.linalg.solve(np.eye(2) - discount * transition, reward[:, 0])

    solver = Solver(model, discount, final_precision=1e-5)
    solver.run()
    assert_allclose(solver.value, expected, atol=1e-5, rtol=0)
    residual = (
        optimal_bellman_operator(reference, solver.value, discount) - solver.value
    )
    assert np.abs(residual).max() <= 1e-5 * (1 - discount)
    assert_array_equal(solver.policy, np.zeros(2, dtype=int))
    assert_array_equal(model.reward_matrix, reference.reward_matrix)
    for matrix, original in zip(model.transition_matrix, reference.transition_matrix):
        for field in ("data", "indices", "indptr"):
            assert_array_equal(getattr(matrix, field), getattr(original, field))


def test_native_vi_rejects_invalid_probabilities():
    pytest.importorskip("mdptoolbox")
    from mdptoolbox.error import StochasticError

    from mdpforge.solvers.mdptoolbox_vi import Solver

    model = SimpleNamespace(
        name="invalid_sparse_vi",
        state_dim=2,
        action_dim=1,
        transition_matrix=[csr_matrix([[0.25, 0.5], [0.25, 0.75]])],
        reward_matrix=np.ones((2, 1)),
    )
    with pytest.raises(StochasticError, match="not stochastic"):
        Solver(model, 0.99).run()


@pytest.mark.parametrize(
    "name, backend_name",
    [("mdptoolbox_vi", "ValueIteration"), ("mdptoolbox_vigs", "ValueIterationGS")],
)
def test_vi_forwards_public_backend_options(chain, monkeypatch, name, backend_name):
    pytest.importorskip("mdptoolbox")
    adapter = import_module(f"mdpforge.solvers.{name}")
    calls = []

    class NativeVI:
        def __init__(self, transitions, rewards, **options):
            calls.append((transitions, rewards, options))
            self.V = (1.8, 2.0, 0.0)
            self.policy = (0, 0, 0)

        def run(self):
            pass

    monkeypatch.setattr(adapter, backend_name, NativeVI)
    solver = adapter.Solver(chain, 0.9, final_precision=1e-4, max_iterations=17)
    solver.run()
    assert len(calls) == 1
    transitions, rewards, options = calls[0]
    assert rewards is chain.reward_matrix
    if name == "mdptoolbox_vi":
        assert transitions is chain.transition_matrix
    else:
        assert_array_equal(
            transitions,
            np.stack([matrix.toarray() for matrix in chain.transition_matrix]),
        )
        assert options.pop("initial_value") == 0
    assert options == {
        "discount": 0.9,
        "epsilon": 1e-4 * (1 - 0.9),
        "max_iter": 17,
    }
    assert_array_equal(solver.value, [1.8, 2.0, 0.0])
    assert_array_equal(solver.policy, [0, 0, 0])


@pytest.mark.parametrize("case", ["constant", "mixing", "roundoff"])
def test_native_vi_errors_are_preserved_and_recorded(case):
    pytest.importorskip("mdptoolbox")
    from mdptoolbox.error import StochasticError
    from mdptoolbox.mdp import ValueIteration

    from mdpforge.solvers.mdptoolbox_vi import Solver

    transition = np.array([[0.8, 0.2], [0.1, 0.9]])
    reward = np.array([[1.0, 0.0], [2.0, 0.0]])
    if case == "constant":
        reward[:] = 1.0
    elif case == "mixing":
        transition[:] = [0.25, 0.75]
    else:
        transition[0, 0] += 1e-12
    model = SimpleNamespace(
        name=f"native_vi_{case}",
        state_dim=2,
        action_dim=2,
        transition_matrix=[csr_matrix(transition) for _ in range(2)],
        reward_matrix=reward,
    )
    with pytest.raises((OverflowError, ValueError, StochasticError)) as native_error:
        ValueIteration(
            model.transition_matrix,
            model.reward_matrix,
            discount=0.9,
            epsilon=1e-5 * (1 - 0.9),
            max_iter=100_000_000,
        )
    with pytest.raises(type(native_error.value)) as wrapper_error:
        Solver(model, 0.9, final_precision=1e-5).run()
    assert str(wrapper_error.value) == str(native_error.value)
    bench = Benchmark(default_solvers=False).add_mdp(model)
    bench.add_solver("mdptoolbox_vi")
    result = bench.run(discount=0.9, precision=1e-5, repeats=1)
    assert result.rows[0]["status"] == "error"
    assert type(native_error.value).__name__ in result.rows[0]["error"]
