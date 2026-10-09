from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge.core.conversion import compute_mdpsolver_args
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
