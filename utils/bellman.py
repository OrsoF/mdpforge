"""Bellman operators, policy evaluation, value-function utilities, and diagnostics."""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel
from utils.exact_value_function import get_exact_value


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


def compact_optimal_bellman_operator(
    model: GenericModel,
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


def full_iterative_policy_evaluation(
    model: GenericModel,
    discount: float,
    policy: np.ndarray,
    variation_tol: float = 1e-2,
    initial_value: Optional[np.ndarray] = None,
):
    transition_policy, reward_policy = compute_transition_reward_policy(model, policy)
    return iterative_policy_evaluation(
        transition_policy,
        reward_policy,
        discount,
        variation_tol,
        initial_value,
    )


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


def compute_transition_reward_policy_numpy(model: GenericModel, policy: np.ndarray):
    transition_policy = np.zeros((model.state_dim, model.state_dim))
    reward_policy = np.zeros(model.state_dim)
    for aa in range(model.action_dim):
        ind: np.ndarray = (policy == aa).nonzero()[0]
        if ind.size > 0:
            transition_policy[ind, :] = model.transition_matrix[aa][ind, :]
            reward_policy[ind] = model.reward_matrix[ind, aa]
    return transition_policy, reward_policy


def value_span_on_regions(
    full_value: np.ndarray, states_in_regions: List[List[int]]
) -> List[float]:
    """
    Returns [max full_value_k - min full_value_k for k in range(K)]
    """
    return [np.ptp(full_value[region]) for region in states_in_regions]


def q_value_span_on_regions(
    full_q_value: np.ndarray, states_in_regions: List[List[int]]
) -> np.ndarray:
    """
    Returns an array of shape (region_number, action_dim)
    """
    region_number, action_dim = len(states_in_regions), full_q_value.shape[1]
    spans_of_q = np.zeros((region_number, action_dim))
    for region_index, region in enumerate(states_in_regions):
        spans_of_q[region_index, :] = full_q_value[region].max(axis=0) - full_q_value[
            region
        ].min(axis=0)
    return spans_of_q


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


def apply_obo_until_var_small(
    model: GenericModel,
    discount: float,
    variation_tol: float,
    initial_value: np.ndarray,
    shift_acceleration: bool = False,
) -> Tuple[np.ndarray, float]:
    value = initial_value

    while True:
        new_value = optimal_bellman_operator(model, value, discount)
        delta = new_value - value
        variation = norminf(delta)
        if variation < variation_tol:
            value = new_value
            break
        if shift_acceleration:
            shift = 0.5 * (delta.max() + delta.min())
            new_value += discount * shift / (1 - discount)
        value = new_value

    return value, variation


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


def policy_bellman_value_to_value(
    env: GenericModel,
    value: np.ndarray,
    policy: np.ndarray,
    discount: float,
) -> np.ndarray:
    """
    Returns T^pi(value) for given pi, value.
    """
    q_value = np.array(
        [
            env.reward_matrix[:, aa] + discount * (env.transition_matrix[aa].dot(value))
            for aa in range(env.action_dim)
        ]
    )

    return q_value.max(axis=0)


def policy_evaluation(
    env: GenericModel, policy: np.ndarray, discount: float, epsi: float = 1e-2
):
    """
    Evaluate the policy using the Bellman operator until convergence.
    """
    assert policy.shape == (
        env.state_dim,
        env.action_dim,
    ), "Policy shape is {} instead of {}".format(
        policy.shape, (env.state_dim, env.action_dim)
    )
    value = np.zeros((env.state_dim,))

    while True:
        value_old = value.copy()
        value = policy_bellman_value_to_value(env, value, policy, discount)

        if np.linalg.norm(value - value_old, ord=np.inf) < epsi:
            return value


def bellman_residual_policy(
    model: GenericModel, policy: np.ndarray, value: np.ndarray, discount: float
) -> float:
    """Compute the Bellman residual for a given policy and value function."""
    transition_policy, reward_policy = compute_transition_reward_policy(model, policy)
    bellman_value = reward_policy + discount * transition_policy.dot(value)
    return norminf(bellman_value - value)


def is_policy_optimal(
    model: GenericModel,
    policy: np.ndarray,
    discount: float,
    evaluation_tol: float = 1e-10,
    optimality_tol: float = 1e-8,
) -> bool:
    """
    Check whether a deterministic policy is greedy with respect to its own
    value function.

    For a finite discounted MDP, this implies optimality.
    """
    policy = np.asarray(policy)

    policy_value = full_iterative_policy_evaluation(
        model,
        discount,
        policy,
        variation_tol=evaluation_tol,
    )

    q_value = bellman_no_max(model, policy_value, discount)

    state_indices = np.arange(policy.shape[0])
    policy_q = q_value[state_indices, policy]
    best_q = np.max(q_value, axis=1)

    return bool(np.all(policy_q >= best_q - optimality_tol))
