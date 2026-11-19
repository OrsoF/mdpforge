"""Small graph-to-discounted-MDP adapter for topology benchmark recipes."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP


class TopologyMDP(MDP):
    """Convert graph neighborhoods into action-dependent stochastic transitions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        seed: int,
        slug: str,
        parameters: tuple,
        goal_fraction: float = 0.05,
        preferred_probability: float = 0.8,
    ) -> None:
        if isinstance(state_dim, bool) or int(state_dim) != state_dim or state_dim < 4:
            raise ValueError("state_dim must be an integer >= 4")
        if (
            isinstance(action_dim, bool)
            or int(action_dim) != action_dim
            or action_dim < 1
        ):
            raise ValueError("action_dim must be a positive integer")
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(goal_fraction) or not 0 < goal_fraction < 1:
            raise ValueError("goal_fraction must be in (0, 1)")
        if not np.isfinite(preferred_probability) or not 0 < preferred_probability <= 1:
            raise ValueError("preferred_probability must lie in (0, 1]")
        self.requested_state_dim = int(state_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.goal_fraction = float(goal_fraction)
        self.preferred_probability = float(preferred_probability)
        self.terminal_states = tuple(
            range(
                self.state_dim - max(1, int(self.state_dim * goal_fraction)),
                self.state_dim,
            )
        )
        suffix = "_".join(
            map(
                str,
                (
                    self.seed,
                    *parameters,
                    self.goal_fraction,
                    self.preferred_probability,
                ),
            )
        )
        self.name = f"{self.state_dim}_{self.action_dim}_{slug}_{suffix}"

    def _adjacency(self, rng: np.random.Generator) -> list[set[int]]:
        raise NotImplementedError

    def _build_model(self) -> None:
        rng = np.random.default_rng(self.seed)
        neighbors = self._adjacency(rng)
        if len(neighbors) != self.state_dim:
            raise RuntimeError("incorrect number of graph vertices")
        goals = set(self.terminal_states)
        rows, cols, values = (
            [[] for _ in range(self.action_dim)],
            [[] for _ in range(self.action_dim)],
            [[] for _ in range(self.action_dim)],
        )
        self.reward_matrix = -np.ones((self.state_dim, self.action_dim), dtype=float)
        self.reward_matrix[list(goals), :] = 0.0
        for state, successors in enumerate(neighbors):
            destinations = (
                [state]
                if state in goals
                else sorted(int(s) for s in successors if 0 <= int(s) < self.state_dim)
            )
            if not destinations:
                destinations = [state]
            k = len(destinations)
            for action in range(self.action_dim):
                if k == 1:
                    weights = [1.0]
                else:
                    # Controlled action changes the dominant outgoing edge while
                    # leaving every graph edge with positive probability.
                    other = (1.0 - self.preferred_probability) / k
                    weights = [other] * k
                    weights[action % k] += self.preferred_probability
                for target, probability in zip(destinations, weights):
                    if probability > 0:
                        rows[action].append(state)
                        cols[action].append(target)
                        values[action].append(probability)
        self.transition_matrix = [
            csr_matrix(
                (values[action], (rows[action], cols[action])),
                shape=(self.state_dim, self.state_dim),
            )
            for action in range(self.action_dim)
        ]
