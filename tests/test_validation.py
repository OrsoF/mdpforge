"""Validate classes of invalid finite MDPs, not individual named models."""

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_array, csr_matrix

from mdpforge.core.validation import validate_model


def bad_name(model):
    del model.name


def bad_rewards_shape(model):
    model.reward_matrix = np.zeros((model.state_dim, model.action_dim + 1))


def bad_transition_shape(model):
    model.transition_matrix[0] = csr_matrix(np.eye(model.state_dim + 1))


def bad_action_count(model):
    model.transition_matrix = model.transition_matrix[:-1]


def bad_transition_sum(model):
    model.transition_matrix[0] *= 0.5


def bad_negative_transition(model):
    transition = model.transition_matrix[0].toarray()
    transition[0] = 0
    transition[0, 0], transition[0, 1] = 1.1, -0.1
    model.transition_matrix[0] = csr_matrix(transition)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (bad_name, "missing required attributes: name"),
        (bad_rewards_shape, "Reward matrix has shape"),
        (bad_transition_shape, "Transition matrix for action 0 has shape"),
        (bad_action_count, "contains 1 actions, expected 2"),
        (bad_transition_sum, "not stochastic"),
        (bad_negative_transition, "negative probabilities"),
    ],
    ids=[
        "name",
        "reward-shape",
        "transition-shape",
        "action-count",
        "stochastic",
        "negative",
    ],
)
def test_model_validation_rejects_structural_violations(small_model, mutate, error):
    mutate(small_model)
    with pytest.raises(ValueError, match=error):
        validate_model(small_model)


@pytest.mark.parametrize("location", ["reward", "transition"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf], ids=["nan", "infinity"])
def test_model_validation_rejects_nonfinite_data(small_model, location, invalid):
    if location == "reward":
        small_model.reward_matrix[0, 0] = invalid
    else:
        transition = small_model.transition_matrix[0].toarray()
        transition[0, 0] = invalid
        small_model.transition_matrix[0] = csr_matrix(transition)
    with pytest.raises(ValueError, match="NaN or infinite values"):
        validate_model(small_model)


@pytest.mark.parametrize("matrix_type", [np.array, csc_matrix, csr_array])
def test_model_validation_requires_csr(small_model, matrix_type):
    small_model.transition_matrix[0] = matrix_type(np.eye(small_model.state_dim))
    with pytest.raises(ValueError, match="must be a CSR matrix"):
        validate_model(small_model)
