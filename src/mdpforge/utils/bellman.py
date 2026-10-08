"""Bellman operators, policy evaluation, value-function utilities, and diagnostics."""

from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.model import GenericModel
from mdpforge.utils.exact_value_function import get_exact_value


def norminf(value: np.ndarray) -> float:
    """
    Infinite norm of a vector.
    """
    return np.absolute(value).max()


def optimal_bellman_operator(
    model: GenericModel, value: np.ndarray, discount: float
) -> np.ndarray:
    """
    Returns T*(value)
    """
    Q = np.empty((model.action_dim, model.state_dim))
    for aa in range(model.action_dim):
        Q[aa] = model.reward_matrix[:, aa] + discount * model.transition_matrix[aa].dot(
            value
        )
    return Q.max(axis=0)


def q_optimal_bellman_operator(
    model: GenericModel, q_value: np.ndarray, discount: float
) -> np.ndarray:
    """
    Returns T*(q_value)
    """
    value = q_value.max(axis=1)
    q_value_new = np.zeros((model.state_dim, model.action_dim))
    for aa in range(model.action_dim):
        q_value_new[:, aa] = model.reward_matrix[
            :, aa
        ] + discount * model.transition_matrix[aa].dot(value)
    return q_value_new


def bellman_operator(model: GenericModel, value: np.ndarray, discount: float):
    """
    Returns R + discount * T @ V
    """
    q_value = np.zeros((model.state_dim, model.action_dim))
    for aa in range(model.action_dim):
        q_value[:, aa] = model.reward_matrix[
            :, aa
        ] + discount * model.transition_matrix[aa].dot(value)
    return q_value


def bellman_policy_operator(
    value: np.ndarray,
    discount: float,
    transition_policy: np.ndarray,
    reward_policy: np.ndarray,
) -> np.ndarray:
    """
    Returns R^pi + gamma . T^pi @ value
    """
    return reward_policy + discount * transition_policy.dot(value)


def iterative_policy_evaluation(
    transition_policy: np.ndarray,
    reward_policy: np.ndarray,
    discount: float,
    variation_tol: float,
    initial_value: np.ndarray,
) -> np.ndarray:
    """
    Apply Bellman^pi to initial_value until |initial_value - Bellman^pi initial_value| < tolerance
    """
    if initial_value is None:
        value = np.zeros((transition_policy.shape[0]))
    else:
        value = initial_value

    while True:
        next_value = bellman_policy_operator(
            value, discount, transition_policy, reward_policy
        )
        distance = norminf(next_value - value)
        value = next_value

        if distance < variation_tol:
            break

    return value


def compute_transition_reward_policy(model: GenericModel, policy: np.ndarray) -> Tuple:
    """
    Given T, R, policy, returns T^pi, R^pi.
    """
    # model._model_to_sparse()
    transition_policy = lil_matrix((model.state_dim, model.state_dim))
    reward_policy = np.zeros(model.state_dim)
    for aa in range(model.action_dim):
        ind = (policy == aa).nonzero()[0]
        if ind.size > 0:
            for ss in ind:
                transition_row = model.transition_matrix[aa][[ss], :]
                if hasattr(transition_row, "toarray"):
                    transition_row = transition_row.toarray()
                transition_policy[[ss], :] = transition_row
            reward_policy[ind] = model.reward_matrix[ind, aa]
    transition_policy = transition_policy.tocsr()
    # model._model_to_numpy()
    return transition_policy, reward_policy


def get_value_policy_value(
    model: GenericModel,
    discount: float,
    value: np.ndarray,
    precision: float = 1e-3,
) -> np.ndarray:
    """
    For any V, returns V^{pi_V}
    """
    model.test_model()
    assert value.shape[0] == model.state_dim, (
        "Shape of the value should be ({},) instead of {}.".format(
            model.state_dim, value.shape
        )
    )
    policy = bellman_operator(model, value, discount).argmax(axis=1)
    transition_policy, reward_policy = compute_transition_reward_policy(model, policy)
    value_policy = iterative_policy_evaluation(
        transition_policy, reward_policy, discount, precision, value
    )
    return value_policy


def optimal_bellman_residual(model: GenericModel, value: np.ndarray, discount: float):
    """Returns ||V - T^* V||_inf."""
    return norminf(optimal_bellman_operator(model, value, discount) - value)


def bellman_no_max(
    model: GenericModel, value: np.ndarray, discount: float
) -> np.ndarray:
    """
    For a given value V, returns (R + gamma * T @ V)
    """
    q_value = np.empty((model.state_dim, model.action_dim))
    for aa in range(model.action_dim):
        q_value[:, aa] = (
            model.reward_matrix[:, aa] + discount * model.transition_matrix[aa] @ value
        )
    return q_value


def get_optimal_policy(model: GenericModel, discount: float) -> np.ndarray:
    value = get_exact_value(model, discount)
    q_value = bellman_no_max(model, value, discount)
    return q_value.argmax(axis=1)
