import numpy as np
import pytest
from scipy.sparse import csr_array

from core.validation import validate_model


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_accepts_valid_model(forest, sparse):
    if not sparse:
        forest._model_to_numpy()
    validate_model(forest)


def test_rejects_wrong_reward_shape(forest):
    forest.reward_matrix = np.zeros((forest.state_dim, forest.action_dim + 1))
    with pytest.raises(ValueError, match="Reward matrix has shape"):
        validate_model(forest)


def test_rejects_wrong_transition_shape(forest):
    forest.transition_matrix[0] = csr_array(np.eye(forest.state_dim + 1))
    with pytest.raises(ValueError, match="Transition matrix for action 0 has shape"):
        validate_model(forest)


def test_rejects_wrong_number_of_actions(forest):
    forest.transition_matrix = forest.transition_matrix[:1]
    with pytest.raises(ValueError, match="contains 1 actions, expected 2"):
        validate_model(forest)


@pytest.mark.parametrize("field", ["reward", "transition"])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf], ids=["nan", "infinity"])
def test_rejects_nonfinite_values(forest, field, invalid_value):
    if field == "reward":
        forest.reward_matrix[0, 0] = invalid_value
    else:
        transition = forest.transition_matrix[0].toarray()
        transition[0, 0] = invalid_value
        forest.transition_matrix[0] = csr_array(transition)
    with pytest.raises(ValueError, match="NaN or infinite values"):
        validate_model(forest)


def test_rejects_nonstochastic_transition(forest):
    forest.transition_matrix[0] = forest.transition_matrix[0] * 0.5
    with pytest.raises(ValueError, match="not stochastic"):
        validate_model(forest)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_rejects_negative_probabilities_even_when_rows_sum_to_one(forest, sparse):
    transition = forest.transition_matrix[0].toarray()
    transition[0] = [1.1, -0.1, 0, 0]
    forest.transition_matrix[0] = csr_array(transition) if sparse else transition
    with pytest.raises(ValueError, match="negative probabilities"):
        validate_model(forest)
