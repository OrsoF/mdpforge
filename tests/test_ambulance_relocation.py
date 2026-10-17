import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.validation import validate_model
from mdpforge.models.ambulance_relocation import Model


@pytest.mark.parametrize("state_dim, action_dim", [(50, 10), (73, 13)])
def test_ambulance_dynamics_costs_and_optimal_value(state_dim, action_dim):
    model = Model(state_dim=state_dim, action_dim=action_dim)
    model.create_model(cache="off")
    validate_model(model)

    expected_rewards = np.empty((state_dim, action_dim))
    for state, location in enumerate(model.state_grid):
        for action, target in enumerate(model.action_grid):
            response = np.dot(model.arrival_probs, np.abs(model.state_grid - target))
            expected_rewards[state, action] = -(response + 0.25 * abs(target - location))
    assert_array_equal(model.reward_matrix, expected_rewards)

    probabilities = np.where(model.arrival_probs > 1e-12, model.arrival_probs, 0)
    for transition in model.transition_matrix:
        assert_array_equal(
            transition.toarray(), np.broadcast_to(probabilities, (state_dim, state_dim))
        )

    # Action-independent transitions make the greedy immediate reward optimal.
    discount = 0.9
    immediate = expected_rewards.max(axis=1)
    continuation = probabilities @ immediate / (1 - discount * probabilities.sum())
    value = immediate + discount * continuation
    assert_allclose(
        optimal_bellman_operator(model, value, discount), value, rtol=0, atol=1e-12
    )

    first = model.transition_matrix[0]
    for transition in model.transition_matrix[1:]:
        for field in ("data", "indices", "indptr"):
            assert not np.shares_memory(getattr(first, field), getattr(transition, field))


def test_ambulance_preserves_transition_cutoff_without_renormalizing(monkeypatch):
    probabilities = np.zeros(50)
    probabilities[:4] = [0.25, 0.75 - 1.5e-12, 1e-12, 0.5e-12]
    monkeypatch.setattr(
        Model, "_arrival_distribution", lambda self, kind: probabilities.copy()
    )
    model = Model(state_dim=50)
    model.create_model(cache="off")
    expected = probabilities.copy()
    expected[2:] = 0
    for transition in model.transition_matrix:
        assert transition.nnz == 100
        assert_array_equal(transition.getrow(0).toarray().ravel(), expected)
