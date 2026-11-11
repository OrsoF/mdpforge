"""Finite-return versions of existing goal-directed discounted models."""

import random
from contextlib import contextmanager

import numpy as np
from scipy.sparse import csr_matrix, diags, hstack, vstack

from mdpforge.core.mdp import MDP


@contextmanager
def _preserved_seed(seed):
    """Seed legacy generators without changing the caller's RNG state."""
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class EpisodicModel(MDP):
    """Add absorbing goals and a geometric timeout to a catalogue model."""

    supported_criteria = frozenset({"total"})
    source_model = None
    stop_probability = 0.01

    def __init__(self, state_dim, *, seed=None):
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        self.seed = seed
        with _preserved_seed(seed):
            self._source = self.source_model(state_dim=state_dim)
        source = self._source
        super().__init__(source.state_dim + 1, source.action_dim)
        self.requested_state_dim = state_dim
        goals = tuple(self._goal_states(source))
        if not goals or any(
            type(s) is not int or not 0 <= s < source.state_dim for s in goals
        ):
            raise ValueError("source model must have valid goal states")
        self.terminal_states = goals + (source.state_dim,)
        self.name = f"{source.name}_total_stop_{self.stop_probability}_seed_{seed}"

    def _goal_states(self, source):
        raise NotImplementedError

    def _build_model(self):
        with _preserved_seed(self.seed):
            self._source.create_model(cache="off")
        source = self._source
        n = source.state_dim
        goals = self.terminal_states[:-1]
        timeout_probability = np.full(n, self.stop_probability)
        timeout_probability[list(goals)] = 0.0
        survival = 1.0 - timeout_probability
        timeout_column = csr_matrix(timeout_probability[:, None])
        timeout_row = csr_matrix(([1.0], ([0], [n])), shape=(1, n + 1))
        self.transition_matrix = []
        for transition in source.transition_matrix:
            original = transition.tolil(copy=True)
            for goal in goals:
                original.rows[goal] = [goal]
                original.data[goal] = [1.0]
            top = hstack((diags(survival) @ original.tocsr(), timeout_column))
            self.transition_matrix.append(vstack((top, timeout_row)).tocsr())
        self.reward_matrix = np.zeros((n + 1, self.action_dim), dtype=float)
        self.reward_matrix[:n] = source.reward_matrix
        self.reward_matrix[list(goals)] = 0.0
        self._source = None
