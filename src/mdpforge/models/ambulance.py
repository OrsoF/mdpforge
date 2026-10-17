# Source: Mason, W. A. and Watts, D. J. (2012). Collaborative learning in networks. PNAS 109(3):764-769.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import beta
from tqdm import trange

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 9, "action_dim": 2}

METADATA = {
    "sizes": {
        "small": {
            "requested_state_dim": 200,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 1500,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 3000,
            "parameters": {},
        },
    },
    "category": "service_operations",
    "description": "One-dimensional ambulance positioning under random demand.",
    "reference": (
        "Mason, W. A. and Watts, D. J. (2012). Collaborative learning in networks. "
        "PNAS 109(3):764-769."
    ),
}


class Model(MDP):
    CACHE_VERSION = 1

    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.requested_state_dim = int(state_dim)
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        state_dim = max(9, state_dim)
        if action_dim < 2:
            raise ValueError("ambulance requires at least two actions.")

        self.state_dim = state_dim
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
