# Source: Strehl, A. L. and Littman, M. L. (2008). An analysis of model-based interval estimation for Markov decision processes. JCSS 74(8):1309-1331.

import numpy as np
from scipy.sparse import dok_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "tags": ["control", "real-world"],
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
    "category": "control",
    "description": "River swimming with current-dependent movement rewards.",
    "reference": (
        "Strehl, A. L. and Littman, M. L. (2008). An analysis of model-based "
        "interval estimation for Markov decision processes. JCSS 74(8):1309-1331."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.state_dim = max(3, state_dim)
        self.action_dim = 2

        self.name = "{}_{}_sim".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.transition_matrix: list = [
            dok_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim), dtype=np.float16
        )

        for ss1 in range(self.state_dim):
            aa = 0
            self.transition_matrix[aa][ss1, max(0, ss1 - 1)] = 1.0

            aa = 1
            if ss1 == 0:
                self.transition_matrix[aa][ss1, ss1] = 0.4
                self.transition_matrix[aa][ss1, ss1 + 1] = 0.6
                self.reward_matrix[ss1, aa] = 1e-3
            elif ss1 == self.state_dim - 1:
                self.transition_matrix[aa][ss1, ss1 - 1] = 0.4
                self.transition_matrix[aa][ss1, ss1] = 0.6
            else:
                self.transition_matrix[aa][ss1, ss1] = 0.6
                self.transition_matrix[aa][ss1, ss1 + 1] = 0.35
                self.transition_matrix[aa][ss1, ss1 - 1] = 0.05
                self.reward_matrix[ss1, aa] = 1

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
