"""Bellman iteration and residuals."""

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import (
    bellman_policy_operator,
    norminf,
    optimal_bellman_operator,
)


def iterative_policy_evaluation(
    transition_policy: np.ndarray | csr_matrix,
    reward_policy: np.ndarray,
    discount: float,
    variation_tol: float,
    initial_value: np.ndarray | None,
    *,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Apply policy Bellman updates until the residual threshold or iteration cap.

    variation_tol: positive residual threshold, already scaled by the caller.
    initial_value: starting vector of shape (S,), or None for zeros.
    max_iterations: None for no cap, or an int > 0; return the last update at the cap.
    """
    assert 0 < discount < 1, "discount must be strictly between 0 and 1"
    if max_iterations is not None and (
        type(max_iterations) is not int or max_iterations < 1
    ):
        raise ValueError("max_iterations must be None or a positive integer")
    value = (
        np.zeros(transition_policy.shape[0]) if initial_value is None else initial_value
    )
    iterations = 0

    while True:
        next_value = bellman_policy_operator(
            value, discount, transition_policy, reward_policy
        )
        policy_residual = norminf(next_value - value)
        value = next_value
        iterations += 1

        if policy_residual < variation_tol or (
            max_iterations is not None and iterations >= max_iterations
        ):
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
