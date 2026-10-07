"""MDP transition sampling and trajectory generation utilities."""

from typing import List, Tuple

import numpy as np

from core.model import GenericModel
from core.partition import Partition

rng = np.random.default_rng(0)


def generate_sars(model: GenericModel, state: int, action: int) -> tuple[int, float]:
    transition = model.transition_matrix[action]
    if hasattr(transition, "indptr"):
        start, end = transition.indptr[state], transition.indptr[state + 1]
        states = transition.indices[start:end]
        probs = transition.data[start:end]
        next_state = state if len(states) == 0 else rng.choice(states, p=probs)
    else:
        probs = transition[state]
        next_state = rng.choice(model.state_dim, p=probs)
    reward = model.reward_matrix[state, action]
    return int(next_state), float(reward)


def generate_trajectory(
    model: GenericModel,
    episode_length: int,
    exploration_prob: float,
    q_value: np.ndarray,
) -> List:
    """Generate a trajectory of (state, action, reward, next_state) tuples."""
    trajectory = []
    state = rng.integers(model.state_dim)
    # state = model.state_dim // 2
    for index in range(episode_length):
        action = (
            rng.integers(model.action_dim)
            if rng.random() < exploration_prob
            else np.argmax(q_value[state])
        )
        next_state, reward = generate_sars(model, state, action)
        trajectory.append((state, action, reward, next_state))
        state = next_state
    return trajectory


def convert_trajectory_to_region(trajectory: list, partition: Partition) -> List:
    """
    Convert a trajectory of (state, action, reward, next_state) tuples to (region, action, reward, next_region).
    """
    converted_trajectory = []
    for state, action, reward, next_state in trajectory:
        region = partition.get_region_index(state)
        next_region = partition.get_region_index(next_state)
        converted_trajectory.append((region, action, reward, next_region))
    return converted_trajectory


def generate_trajectory_partition(
    model: GenericModel,
    partition: Partition,
    episode_length: int,
    exploration_prob: float,
    q_value: np.ndarray,
):
    trajectory = []
    state = rng.integers(model.state_dim)
    state_region = partition.get_region_index(state)
    for index in range(episode_length):
        action = (
            rng.integers(model.action_dim)
            if rng.random() < exploration_prob
            else np.argmax(q_value[state])
        )
        next_state, reward = generate_sars(model, state, action)
        next_state_region = partition.get_region_index(next_state)

        trajectory.append((state_region, action, reward, next_state_region))
        if next_state == state:
            next_state = rng.integers(model.state_dim)
        state = next_state
        state_region = next_state_region
    return trajectory


def generate_sars_random_trajectory_epsilon_greedy(
    model: GenericModel, start_state: int, episode_length: int
) -> List[Tuple[int, float]]:
    """Generate a trajectory of (state, action, reward, next_state) tuples."""
    trajectory = []
    state = start_state
    for _ in range(episode_length):
        action = rng.integers(model.action_dim)  # Random action
        next_state, reward = generate_sars(model, state, action)
        trajectory.append((state, action, reward, next_state))
        state = next_state
    return trajectory
