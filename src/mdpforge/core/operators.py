from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, diags, issparse

if TYPE_CHECKING:
    from mdpforge.core.model import GenericModel


def norminf(value: np.ndarray) -> float:
    """Compute the infinity norm of a vector."""
    return np.max(np.abs(value))


def bellman_operator(
    model: "GenericModel", value: np.ndarray, discount: float
) -> np.ndarray:
    """
    Apply R + discount * T @ V.

    Returns a Q-value array with shape (state_dim, action_dim).
    """
    q_value = np.empty((model.state_dim, model.action_dim))
    for action in range(model.action_dim):
        q_value[:, action] = model.reward_matrix[
            :, action
        ] + discount * model.transition_matrix[action].dot(value)
    return q_value


def optimal_bellman_operator(
    model: "GenericModel", value: np.ndarray, discount: float
) -> np.ndarray:
    """Apply the optimal Bellman operator to a value function."""
    return np.max(bellman_operator(model, value, discount), axis=1)


def q_optimal_bellman_operator(
    model: "GenericModel", q_value: np.ndarray, discount: float
) -> np.ndarray:
    """Apply the optimal Bellman operator to a Q-value function."""
    value = q_value.max(axis=1)
    return bellman_operator(model, value, discount)


def bellman_policy_operator(
    value: np.ndarray,
    discount: float,
    transition_policy,
    reward_policy: np.ndarray,
) -> np.ndarray:
    """Apply the Bellman operator for a fixed policy."""
    return reward_policy + discount * transition_policy.dot(value)


def compute_transition_reward_policy(
    model: "GenericModel", policy: np.ndarray
) -> tuple:
    """Given T, R, and a policy, return T^pi and R^pi."""
    transition_policy = csr_matrix((model.state_dim, model.state_dim))

    for action in range(model.action_dim):
        mask = policy == action
        if not np.any(mask):
            continue
        transition = model.transition_matrix[action]
        if not issparse(transition):
            transition = csr_matrix(transition)
        transition_policy += diags(mask.astype(float)) @ transition

    states = np.arange(model.state_dim)
    reward_policy = np.asarray(model.reward_matrix[states, policy]).ravel()
    return transition_policy.tocsr(), reward_policy


def compact_optimal_bellman_operator(
    model: "GenericModel",
    value: np.ndarray,
    discount: float,
    shared_reward: bool = False,
) -> np.ndarray:
    """Apply T* without materializing the full action-by-state array."""
    if shared_reward:
        new_value = model.transition_matrix[0].dot(value)
        for aa in range(1, model.action_dim):
            np.maximum(
                new_value,
                model.transition_matrix[aa].dot(value),
                out=new_value,
            )
        new_value *= discount
        new_value += model.reward_matrix[:, 0]
        return new_value

    new_value = model.reward_matrix[:, 0] + discount * model.transition_matrix[0].dot(
        value
    )
    for aa in range(1, model.action_dim):
        candidate = model.reward_matrix[:, aa] + discount * model.transition_matrix[
            aa
        ].dot(value)
        np.maximum(new_value, candidate, out=new_value)
    return new_value
