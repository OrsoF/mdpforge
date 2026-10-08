import numpy as np
import pytest
from scipy.sparse import csr_array

from mdpforge.core.validation import validate_model


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_accepts_valid_model(small_model, sparse):
    if not sparse:
        small_model._model_to_numpy()
    validate_model(small_model)


def test_rejects_wrong_reward_shape(small_model):
    small_model.reward_matrix = np.zeros(
        (small_model.state_dim, small_model.action_dim + 1)
    )
    with pytest.raises(ValueError, match="Reward matrix has shape"):
        validate_model(small_model)


def test_rejects_wrong_transition_shape(small_model):
    small_model.transition_matrix[0] = csr_array(np.eye(small_model.state_dim + 1))
    with pytest.raises(ValueError, match="Transition matrix for action 0 has shape"):
        validate_model(small_model)


def test_rejects_wrong_number_of_actions(small_model):
    small_model.transition_matrix = small_model.transition_matrix[:1]
    with pytest.raises(ValueError, match="contains 1 actions, expected 2"):
        validate_model(small_model)


@pytest.mark.parametrize("field", ["reward", "transition"])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf], ids=["nan", "infinity"])
def test_rejects_nonfinite_values(small_model, field, invalid_value):
    if field == "reward":
        small_model.reward_matrix[0, 0] = invalid_value
    else:
        transition = small_model.transition_matrix[0].toarray()
        transition[0, 0] = invalid_value
        small_model.transition_matrix[0] = csr_array(transition)
    with pytest.raises(ValueError, match="NaN or infinite values"):
        validate_model(small_model)


def test_rejects_nonstochastic_transition(small_model):
    small_model.transition_matrix[0] = small_model.transition_matrix[0] * 0.5
    with pytest.raises(ValueError, match="not stochastic"):
        validate_model(small_model)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_rejects_negative_probabilities_even_when_rows_sum_to_one(small_model, sparse):
    transition = small_model.transition_matrix[0].toarray()
    transition[0] = [1.1, -0.1, 0, 0]
    small_model.transition_matrix[0] = csr_array(transition) if sparse else transition
    with pytest.raises(ValueError, match="negative probabilities"):
        validate_model(small_model)
