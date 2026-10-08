"""Q-learning and partition-based temporal-difference update routines."""

from typing import Tuple

import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.core.partition import Partition

from .simulation import generate_sars, rng


def q_learning_update(
    model: GenericModel,
    q_value: np.ndarray,
    lr: float,
    state: int,
    exploration_prob: float,
    discount: float,
) -> Tuple[np.ndarray, int]:
    action = (
        np.argmax(q_value[state])
        if rng.random() < exploration_prob
        else rng.integers(model.action_dim)
    )
    next_state, reward = generate_sars(model, state, action)
    q_value[state, action] += lr * (
        reward + discount * q_value[next_state].max() - q_value[state, action]
    )
    return q_value, next_state


def q_value_block_update(
    q_value: np.ndarray,
    partition: Partition,
    current_state: int,
    action: int,
    lr: float,
    reward: float,
    next_state: int,
    discount: float,
) -> np.ndarray:
    """
    Update the q_value by block using the partition. To be improved using np array properties.
    """
    current_region_index = partition.get_region_index(current_state)

    delta = (
        reward + discount * q_value[next_state].max() - q_value[current_state, action]
    )
    q_value[partition.states_in_region[current_region_index], action] += lr * delta
    return q_value


def ql_block_update_step(
    model: GenericModel,
    partition: Partition,
    q_value: np.ndarray,
    lr: float,
    current_state: int,
    epsilon: float,
    discount: float,
):

    action = (
        rng.integers(model.action_dim)
        if rng.random() < epsilon
        else np.argmax(q_value[current_state])
    )
    next_state, reward = generate_sars(model, current_state, action)
    current_region = partition.get_region_index(current_state)
    delta = (
        reward + discount * q_value[next_state].max() - q_value[current_state, action]
    )
    q_value[partition.states_in_region[current_region], action] += lr * delta

    return q_value, next_state


def ql_block_update_along_trajectory(
    model: GenericModel,
    partition: Partition,
    q_value: np.ndarray,
    lr: float,
    epsilon: float,
    discount: float,
    start_state: int,
    episode_length: int,
):
    state = start_state
    for _ in range(episode_length):
        action = (
            rng.integers(model.action_dim)
            if rng.random() < epsilon
            else np.argmax(q_value[state])
        )
        next_state, reward = generate_sars(model, state, action)
        current_region = partition.get_region_index(state)
        delta = reward + discount * q_value[next_state].max() - q_value[state, action]
        q_value[partition.states_in_region[current_region], action] += lr * delta

        state = next_state

    return q_value
