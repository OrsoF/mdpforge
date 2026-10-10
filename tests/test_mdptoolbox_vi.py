"""Regression cases for sparse and degenerate mdptoolbox VI inputs."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from mdpforge.core.operators import optimal_bellman_operator


@pytest.mark.parametrize("discount", [0.5, 0.99, 0.999])
@pytest.mark.parametrize("case", ["constant", "mixing", "roundoff"])
def test_sparse_vi_accuracy_and_preservation(discount, case, monkeypatch):
    pytest.importorskip("mdptoolbox")
    from mdpforge.solvers.mdptoolbox_vi import Solver

    transition = np.array([[0.25, 0.75], [0.25, 0.75]])
    reward = np.array([[1.0, 0.0], [2.0, 0.0]])
    if case == "constant":
        reward[:] = 1.0
    elif case == "roundoff":
        transition[0, 0] += 1e-14
    model = SimpleNamespace(
        name="sparse_vi_regression",
        state_dim=2,
        action_dim=2,
        transition_matrix=[csr_matrix(transition) for _ in range(2)],
        reward_matrix=reward,
    )
    reference = deepcopy(model)
    # Action zero is optimal; independently solve its discounted linear system.
    expected = np.linalg.solve(np.eye(2) - discount * transition, reward[:, 0])

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Sparse VI must not materialize an S x S array")

    monkeypatch.setattr(csr_matrix, "toarray", forbid_dense)
    monkeypatch.setattr(csr_matrix, "todense", forbid_dense)
    # Comparing a sparse matrix to zero can allocate every implicit zero.
    monkeypatch.setattr(csr_matrix, "__ge__", forbid_dense)
    solver = Solver(model, discount, final_precision=1e-5)
    solver.run()
    assert_allclose(solver.value, expected, atol=1e-5, rtol=0)
    residual = optimal_bellman_operator(reference, solver.value, discount) - solver.value
    assert np.abs(residual).max() <= 1e-5 * (1 - discount)
    assert_array_equal(solver.policy, np.zeros(2, dtype=int))
    assert_array_equal(model.reward_matrix, reference.reward_matrix)
    for matrix, original in zip(model.transition_matrix, reference.transition_matrix):
        for field in ("data", "indices", "indptr"):
            assert_array_equal(getattr(matrix, field), getattr(original, field))


def test_sparse_vi_rejects_invalid_probabilities():
    pytest.importorskip("mdptoolbox")
    from mdpforge.solvers.mdptoolbox_vi import Solver

    model = SimpleNamespace(
        name="invalid_sparse_vi",
        state_dim=2,
        action_dim=1,
        transition_matrix=[csr_matrix([[0.25, 0.5], [0.25, 0.75]])],
        reward_matrix=np.ones((2, 1)),
    )
    with pytest.raises(ValueError, match="not stochastic"):
        Solver(model, 0.99).run()
