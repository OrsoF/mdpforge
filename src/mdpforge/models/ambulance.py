# Source: Mason, W. A. and Watts, D. J. (2012). Collaborative learning in networks. PNAS 109(3):764-769.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import beta
from tqdm import trange

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 9, "action_dim": 2}

METADATA = {
    "tags": ["resource-management", "real-world"],
    "sizes": {
        "small": {
            "state_dim": 100,
            "parameters": {"state_dim": 10000},
            "source": "measured",
        },
        "medium": {
            "state_dim": 200,
            "parameters": {"state_dim": 40000},
            "source": "measured",
        },
        "large": {
            "state_dim": 400,
            "parameters": {"state_dim": 160000},
            "source": "inferred",
        },
    },
    "category": "resource_management",
    "description": "One-dimensional ambulance positioning under random demand.",
    "reference": (
        "Mason, W. A. and Watts, D. J. (2012). Collaborative learning in networks. "
        "PNAS 109(3):764-769."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        if state_dim < 9:
            raise ValueError(
                "ambulance requires at least three grid points (state_dim >= 9)."
            )
        if action_dim < 2:
            raise ValueError("ambulance requires at least two actions.")

        self.state_dim = int(np.sqrt(state_dim))
        self.action_dim = action_dim

        self.name = "{}_{}_ambulance".format(self.state_dim, self.action_dim)

    def _reward_function(self, ss, aa):
        ss /= self.state_dim - 1
        aa /= self.action_dim - 1
        return 1 - abs(ss - aa)

    def _transition_function(self, ss: int):
        ss /= self.state_dim - 1
        return beta.pdf(ss, 5, 2)

    def _build_model(self):
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        transition_distribution = np.array(
            [self._transition_function(ss) for ss in range(self.state_dim)]
        )
        transition_distribution /= transition_distribution.sum()

        transition_matrix = csr_matrix(
            np.tile(transition_distribution, (self.state_dim, 1))
        )
        self.transition_matrix = [
            transition_matrix.copy() for _ in range(self.action_dim)
        ]

        for ss1 in trange(self.state_dim):
            for aa in range(self.action_dim):
                self.reward_matrix[ss1, aa] = self._reward_function(ss1, aa)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
