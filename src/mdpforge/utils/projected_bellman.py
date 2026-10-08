from typing import List, Optional

import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.utils.bellman import norminf


def projected_optimal_bellman_operator(
    model: GenericModel,
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    weights,
) -> np.ndarray:
    """
    Returns w @ max_a(agg_R + gamma * agg_T @ agg_V)
    """
    contracted_value = np.asarray(contracted_value).reshape(-1)
    contracted_q_value = np.empty((model.state_dim, model.action_dim))

    for aa in range(model.action_dim):
        contracted_q_value[:, aa] = aggregated_reward[
            :, aa
        ] + discount * aggregated_transition[aa].dot(contracted_value)

    return weights.dot(contracted_q_value.max(axis=1))


def apply_pobo_until_var_small(
    model: GenericModel,
    discount: float,
    aggregated_transition: List,
    aggregated_reward: np.ndarray,
    weights,
    variation_tol: float,
    initial_contracted_value: Optional[np.ndarray] = None,
    norm_weights: Optional[np.ndarray] = None,
    max_steps=np.inf,
    shift_acceleration: bool = False,
):
    steps_done = 0

    contracted_value = (
        initial_contracted_value
        if initial_contracted_value is not None
        else np.zeros((aggregated_transition[0].shape[0]))
    )

    while True:
        steps_done += 1
        new_contracted_value = projected_optimal_bellman_operator(
            model,
            discount,
            contracted_value,
            aggregated_transition,
            aggregated_reward,
            weights,
        )
        delta = new_contracted_value - contracted_value
        if norm_weights is None:
            variation = norminf(delta)
        else:
            variation = norminf(np.asarray(norm_weights).ravel() * delta)
        if variation < variation_tol:
            contracted_value = new_contracted_value
            break
        if shift_acceleration:
            shift = (delta.max() + delta.min()) / 2
            # Projected Bellman operators preserve uniform shifts up to discount.
            new_contracted_value += discount * shift / (1 - discount)
        contracted_value = new_contracted_value
        if steps_done >= max_steps:
            break

    return contracted_value, variation
