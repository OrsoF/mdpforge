# Source: Sinclair, Banerjee & Yu (2019), Adaptive Discretization for Episodic RL.
# Ambulance relocation has continuous locations in [0, 1]; this is a tabular discretization.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


class Model(MDP):
    """Discretized one-dimensional ambulance relocation benchmark.

    State is current ambulance location bin.  Action is relocation target bin.
    After relocation, a request arrives from a mixture distribution; next state is
    the request location after service.  Reward is negative response + relocation
    cost, with higher values being better.
    """

    def __init__(self, state_dim: int = 2500, action_dim: int = 10):
        state_dim = int(np.sqrt(state_dim))
        self.state_dim = max(50, int(state_dim))
        self.action_dim = max(10, int(action_dim))
        self.name = f"{self.state_dim}_{self.action_dim}_ambulance_relocation"

    def _build_model(self):
        self.state_grid = np.linspace(0.0, 1.0, self.state_dim)
        self.action_grid = np.linspace(0.0, 1.0, self.action_dim)
        self.arrival_probs = self._arrival_distribution(kind="beta_mixture")
        self.expected_request = float(np.dot(self.arrival_probs, self.state_grid))
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim), dtype=float)
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s, x in enumerate(self.state_grid):
            for a, u in enumerate(self.action_grid):
                response_cost = np.dot(self.arrival_probs, np.abs(self.state_grid - u))
                relocation_cost = 0.25 * abs(u - x)
                self.reward_matrix[s, a] = -(response_cost + relocation_cost)
                for ns, p in enumerate(self.arrival_probs):
                    if p > 1e-12:
                        self.transition_matrix[a][s, ns] = p

        self.transition_matrix = [m.tocsr() for m in self.transition_matrix]

    def _arrival_distribution(self, kind="uniform"):
        x = self.state_grid
        if kind == "uniform":
            probs = np.ones_like(x)
        else:
            # Smooth two-hotspot Beta-like density without scipy.stats dependency.
            eps = 1e-9
            left = np.power(np.maximum(x, eps), 2.0 - 1.0) * np.power(
                np.maximum(1 - x, eps), 6.0 - 1.0
            )
            right = np.power(np.maximum(x, eps), 6.0 - 1.0) * np.power(
                np.maximum(1 - x, eps), 2.0 - 1.0
            )
            probs = 0.55 * left + 0.45 * right
        probs = np.maximum(probs, 0.0)
        return probs / probs.sum()
