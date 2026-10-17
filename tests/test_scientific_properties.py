"""Model-independent identities that detect incorrect test oracles and operators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.operators import optimal_bellman_operator
from tests._contracts import enumerate_optimal_value, matrix_model


@pytest.mark.parametrize("discount", [0.3, 0.9, 0.99])
def test_exhaustive_oracle_is_a_bellman_fixed_point(mdp_case, discount):
    """Enumeration, linear algebra and the Bellman operator must agree."""
    value = enumerate_optimal_value(mdp_case, discount)
    assert_allclose(optimal_bellman_operator(mdp_case, value, discount), value, atol=1e-11)


@pytest.mark.parametrize("shift", [-3.0, 2.0])
def test_reward_translation_shifts_optimal_value(mdp_case, shift):
    discount = 0.9
    original = enumerate_optimal_value(mdp_case, discount)
    translated = matrix_model(
        "translated", mdp_case.transition_matrix, mdp_case.reward_matrix + shift
    )
    assert_allclose(
        enumerate_optimal_value(translated, discount),
        original + shift / (1 - discount),
        atol=1e-11,
    )


def test_positive_reward_rescaling_rescales_optimal_value(mdp_case):
    discount = 0.8
    scaled = matrix_model(
        "scaled", mdp_case.transition_matrix, 3.5 * mdp_case.reward_matrix
    )
    assert_allclose(
        enumerate_optimal_value(scaled, discount),
        3.5 * enumerate_optimal_value(mdp_case, discount),
        atol=1e-11,
    )


def test_state_and_action_relabeling_do_not_change_optimal_value(mdp_case):
    discount = 0.9
    permutation = np.arange(mdp_case.state_dim)[::-1]
    transitions = [
        matrix.toarray()[permutation][:, permutation]
        for matrix in reversed(mdp_case.transition_matrix)
    ]
    rewards = mdp_case.reward_matrix[permutation, ::-1]
    relabeled = matrix_model("relabelled", transitions, rewards)
    assert_allclose(
        enumerate_optimal_value(relabeled, discount),
        enumerate_optimal_value(mdp_case, discount)[permutation],
        atol=1e-11,
    )
