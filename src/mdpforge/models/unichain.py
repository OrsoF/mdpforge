# Source: Puterman, M. L. (2014). Markov Decision Processes: Discrete Stochastic Dynamic Programming, p.353.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "tags": ["synthetic"],
    "sizes": {
        "small": {
            "state_dim": 100,
            "parameters": {"state_dim": 100},
            "source": "measured",
        },
        "medium": {
            "state_dim": 1000,
            "parameters": {"state_dim": 1000},
            "source": "measured",
        },
        "large": {
            "state_dim": 8000,
            "parameters": {"state_dim": 8000},
            "source": "inferred",
        },
    },
    "category": "synthetic",
    "description": "Two-action chain with a zero-cost boundary state.",
    "reference": (
        "Puterman, M. L. (2014). Markov Decision Processes: Discrete Stochastic "
        "Dynamic Programming, p.353."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = 2

        self.name = "{}_{}_unichain".format(
            self.state_dim,
            self.action_dim,
        )

    def _build_model(self):
        self.reward_matrix = -1 * np.ones((self.state_dim, self.action_dim))
        self.reward_matrix[0, :] = 0.0

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s1 in range(self.state_dim):
            self.transition_matrix[0][s1, max(s1 - 1, 0)] = 1.0
            self.transition_matrix[1][s1, min(s1 + 1, self.state_dim - 1)] = 1.0

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]
