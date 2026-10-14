import sys
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge.core.conversion import (
    build_marmote_transition_list,
    compute_mdpsolver_args,
)
from mdpforge.core.mdp import MDP
from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import (
    bellman_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
)
from mdpforge.core.validation import validate_model


def test_matrix_mdp_accepts_dense_transitions(chain):
    transitions = np.array([matrix.toarray() for matrix in chain.transition_matrix])
    model = MDP.from_matrices("dense_mdp", transitions, chain.reward_matrix)

    assert all(isinstance(matrix, csr_matrix) for matrix in model.transition_matrix)
    assert_allclose(
        optimal_bellman_operator(model, np.array([2.0, 4.0, 8.0]), 0.5),
        [2.0, 6.0, 4.0],
    )


def test_bellman_and_policy_selection():
    transitions = np.array(
        [
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]],
        ],
        dtype=float,
    )
    model: MDPProtocol = SimpleNamespace(
        name="external_mdp",
        state_dim=3,
        action_dim=2,
        transition_matrix=[csr_matrix(p) for p in transitions],
        reward_matrix=np.array([[1, 0], [2, 3], [0, 4]], dtype=float),
    )
    validate_model(model)
    value = np.array([2, 4, 8], dtype=float)
    expected_q = np.array([[3, 1], [6, 4.5], [1, 8]])
    assert_allclose(bellman_operator(model, value, 0.5), expected_q)
    assert_allclose(optimal_bellman_operator(model, value, 0.5), [3, 6, 8])

    transition, reward = compute_transition_reward_policy(model, np.array([0, 1, 1]))
    assert_allclose(transition.toarray(), [[0, 1, 0], [0.5, 0.5, 0], [0, 0, 1]])
    assert_allclose(reward, [1, 3, 4])
    assert_allclose(reward + 0.5 * (transition @ value), [3, 4.5, 8])


def test_mdpsolver_conversion_accepts_external_model(chain):
    model: MDPProtocol = SimpleNamespace(
        name=chain.name,
        state_dim=chain.state_dim,
        action_dim=chain.action_dim,
        transition_matrix=chain.transition_matrix,
        reward_matrix=chain.reward_matrix,
    )
    transitions, rewards = compute_mdpsolver_args(model)
    assert transitions == [
        [0, 0, 1, 1.0],
        [1, 0, 2, 1.0],
        [2, 0, 2, 1.0],
        [0, 1, 0, 1.0],
        [1, 1, 1, 1.0],
        [2, 1, 2, 1.0],
    ]
    assert rewards == [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("backend", ["mdpsolver", "marmote"])
def test_transition_conversion_preserves_zeros_duplicates_and_inputs(
    canonical, backend, monkeypatch
):
    transition = csr_matrix(
        ([0.25, 0.75, 0.0, 1.0], [1, 1, 0, 0], [0, 3, 4]), shape=(2, 2)
    )
    if canonical:
        transition.sum_duplicates()
    original = transition.copy()
    rewards = np.array([[2.0], [3.0]])
    expected = [(0, 1, 1.0), (1, 0, 1.0)]
    if not canonical:
        expected.insert(0, (0, 1, 1.0))

    if backend == "mdpsolver":
        model = SimpleNamespace(
            state_dim=2,
            action_dim=1,
            transition_matrix=[transition],
            reward_matrix=rewards,
        )
        transitions, converted_rewards = compute_mdpsolver_args(model)
        assert transitions == [
            [row, 0, column, value] for row, column, value in expected
        ]
        assert converted_rewards == [[2.0], [3.0]]
        assert_allclose(rewards, [[2.0], [3.0]], rtol=0, atol=0)
    else:

        class SparseMatrix:
            def __init__(self, size):
                self.size = size
                self.entries = []

            def addToEntry(self, row, column, value):
                self.entries.append((row, column, value))

        core = SimpleNamespace(SparseMatrix=SparseMatrix)
        monkeypatch.setitem(sys.modules, "marmote", SimpleNamespace(core=core))
        monkeypatch.setitem(sys.modules, "marmote.core", core)
        matrices = build_marmote_transition_list(2, 1, [transition])
        assert len(matrices) == 1
        assert matrices[0].size == 2
        assert matrices[0].entries == expected

    assert_allclose(transition.data, original.data, rtol=0, atol=0)
    assert_allclose(transition.indices, original.indices, rtol=0, atol=0)
    assert_allclose(transition.indptr, original.indptr, rtol=0, atol=0)
