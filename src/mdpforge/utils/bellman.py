"""Bellman iteration and residuals."""

from typing import Tuple

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import (
    bellman_policy_operator,
    norminf,
    optimal_bellman_operator,
)


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
    assert 0 < discount < 1, "discount must be strictly between 0 and 1"
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


def apply_obo_until_var_small(
    model: MDPProtocol,
    discount: float,
    variation_tol: float,
    initial_value: np.ndarray,
    shift_acceleration: bool = False,
) -> Tuple[np.ndarray, float]:
    assert 0 < discount < 1, "discount must be strictly between 0 and 1"
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


def optimal_bellman_residual(model: MDPProtocol, value: np.ndarray, discount: float):
    """Returns ||V - T^* V||_inf."""
    return norminf(optimal_bellman_operator(model, value, discount) - value)
