from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_array

from mdpforge.core.operators import (
    bellman_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
)
from mdpforge.core.validation import validate_model


@pytest.mark.parametrize("sparse", [False, True])
def test_bellman_and_policy_selection(sparse):
    transitions = np.array(
        [
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]],
        ],
        dtype=float,
    )
    model = SimpleNamespace(
        state_dim=3,
        action_dim=2,
        transition_matrix=[csr_array(p) for p in transitions]
        if sparse
        else transitions,
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
