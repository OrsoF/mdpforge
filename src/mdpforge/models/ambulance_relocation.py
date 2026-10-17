# Source: Sinclair, Banerjee & Yu (2019), Adaptive Discretization for Episodic RL.
# Ambulance relocation has continuous locations in [0, 1]; this is a tabular discretization.

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "sizes": {
        "small": {
            "requested_state_dim": 1000,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 5000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 15000,
            "parameters": {},
        },
    },
    "category": "service_operations",
    "description": "Ambulance relocation with response and movement costs.",
    "reference": (
        "Sinclair, Banerjee & Yu (2019), Adaptive Discretization for Episodic RL."
    ),
}


class Model(MDP):
    """Discretized one-dimensional ambulance relocation benchmark.

    State is current ambulance location bin.  Action is relocation target bin.
    After relocation, a request arrives from a mixture distribution; next state is
    the request location after service.  Reward is negative response + relocation
    cost, with higher values being better.
    """

    def __init__(self, state_dim: int = 2500, action_dim: int = 10):
        self.requested_state_dim = int(state_dim)
        self.state_dim = max(50, int(state_dim))
        self.action_dim = max(10, int(action_dim))
        self.name = f"{self.state_dim}_{self.action_dim}_ambulance_relocation"

    def _build_model(self):
        self.state_grid = np.linspace(0.0, 1.0, self.state_dim)
        self.action_grid = np.linspace(0.0, 1.0, self.action_dim)
        self.arrival_probs = self._arrival_distribution(kind="beta_mixture")
        self.expected_request = float(np.dot(self.arrival_probs, self.state_grid))
        response_cost = np.array(
            [
                np.dot(self.arrival_probs, np.abs(self.state_grid - u))
                for u in self.action_grid
            ]
        )
        relocation_cost = 0.25 * np.abs(
            self.action_grid[None, :] - self.state_grid[:, None]
        )
        self.reward_matrix = -(response_cost[None, :] + relocation_cost)

        # Every row and action has the same post-service request distribution.
        destinations = np.flatnonzero(self.arrival_probs > 1e-12)
        transition = csr_matrix(
            (
                np.tile(self.arrival_probs[destinations], self.state_dim),
                np.tile(destinations, self.state_dim),
                np.arange(self.state_dim + 1, dtype=np.int64) * destinations.size,
            ),
            shape=(self.state_dim, self.state_dim),
        )
        self.transition_matrix = [transition] + [
            transition.copy() for _ in range(self.action_dim - 1)
        ]

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
