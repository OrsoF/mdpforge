from typing import List, Optional, Tuple

import numpy as np

from core.model import GenericModel
from utils.bellman import norminf


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


def projected_optimal_bellman_operator_residual(
    model: GenericModel,
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    weights,
):
    return norminf(
        contracted_value
        - projected_optimal_bellman_operator(
            model,
            discount,
            contracted_value,
            aggregated_transition,
            aggregated_reward,
            weights,
        )
    )


def aggregation_norm_weights(model: GenericModel, partition, method: str) -> np.ndarray:
    if method == "region_size":
        return 1 / np.bincount(
            partition.state_to_region,
            minlength=partition.n_regions,
        )
    if method == "average_reward":
        return np.array(
            [
                model.reward_matrix[region, :].max(axis=1).mean() + 1e-2
                for region in partition.states_in_region
            ]
        )
    if method == "average_average_reward":
        return np.array(
            [
                model.reward_matrix[region, :].mean() + 1e-2
                for region in partition.states_in_region
            ]
        )
    if method == "max_reward_region":
        return np.array(
            [
                model.reward_matrix[region, :].max()
                for region in partition.states_in_region
            ]
        )
    raise ValueError(f"Unknown norm weight method: {method}")


def projected_optimal_q_bellman_operator(
    model: GenericModel,
    discount: float,
    contracted_q_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
) -> np.ndarray:
    """
    Returns w @ max_a(agg_R + gamma * agg_T @ agg_V)
    """

    region_number = contracted_q_value.shape[0]

    contracted_value = contracted_q_value.max(axis=1)

    contracted_q_value = np.empty((region_number, model.action_dim))
    for aa in range(model.action_dim):
        contracted_q_value[:, aa] = aggregated_reward[
            :, aa
        ] + discount * aggregated_transition[aa].dot(contracted_value)

    return contracted_q_value


def apply_poqbo_until_var_small(
    model: GenericModel,
    discount: float,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    variation_tol: float,
    init_agg_qvalue: Optional[np.ndarray] = None,
    norm_weights: Optional[np.ndarray] = None,
    max_steps=np.inf,
    shift_acceleration: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Apply Pi T_Q until |V - Pi T_Q V| < variation_tol.

    ``shift_acceleration`` removes the nearly constant residual component whose
    discounted decay is especially slow when the discount is close to one.
    """
    steps_done = 0

    q_contracted_value = (
        np.zeros((aggregated_reward.shape[0]))
        if init_agg_qvalue is None
        else init_agg_qvalue
    )

    while True:
        steps_done += 1

        new_q_contracted_value = projected_optimal_q_bellman_operator(
            model,
            discount,
            q_contracted_value,
            aggregated_transition,
            aggregated_reward,
        )
        delta = new_q_contracted_value - q_contracted_value
        if norm_weights is None:
            variation = norminf(delta)
        else:
            norm_weights = np.asarray(norm_weights)
            if norm_weights.ndim == 1:
                norm_weights = norm_weights[:, None]
            variation = norminf(norm_weights * delta)
        if variation < variation_tol:
            q_contracted_value = new_q_contracted_value
            break
        if shift_acceleration:
            shift = (delta.max() + delta.min()) / 2
            # T(Q + c) = T(Q) + discount * c for a uniform scalar shift.
            new_q_contracted_value += discount * shift / (1 - discount)
        q_contracted_value = new_q_contracted_value
        if steps_done >= max_steps:
            break

    return q_contracted_value, variation


def projected_policy_bellman_operator(
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition_policy: np.ndarray,
    aggregated_reward_policy: np.ndarray,
) -> np.ndarray:
    """
    Returns (w R^pi phi) + discount * (w T^pi phi) V
    """
    return aggregated_reward_policy + discount * aggregated_transition_policy.dot(
        contracted_value
    )


def apply_ppbo_until_var_small(
    discount: float,
    agg_trans_pi: np.ndarray,
    agg_rew_pi: np.ndarray,
    variation_tol: float,
    init_agg_value: np.ndarray,
    max_steps: int,
    shift_acceleration: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Applies Pi T^pi until ||V - Pi T^pi V|| <= variation_tol.
    """
    contracted_value = init_agg_value
    steps_done = 0

    while True:
        steps_done += 1
        new_contracted_value = projected_policy_bellman_operator(
            discount,
            contracted_value,
            agg_trans_pi,
            agg_rew_pi,
        )
        delta = new_contracted_value - contracted_value
        variation = norminf(delta)
        if variation < variation_tol:
            contracted_value = new_contracted_value
            break
        if shift_acceleration:
            shift = (delta.max() + delta.min()) / 2
            new_contracted_value += discount * shift / (1 - discount)
        contracted_value = new_contracted_value
        if steps_done >= max_steps:
            break

    return contracted_value, variation


def check_dimensions(
    model: GenericModel,
    contracted_value: np.ndarray,
    aggregated_transition: list,
    aggregated_reward: np.ndarray,
):
    region_number = contracted_value.shape[0]
    assert len(aggregated_transition) == model.action_dim
    assert aggregated_transition[0].shape == (
        model.state_dim,
        region_number,
    ), "Wrong transition shape : {} != {}".format(
        aggregated_transition[0].shape, (model.state_dim, region_number)
    )
    assert aggregated_reward.shape == (model.state_dim, model.action_dim)
