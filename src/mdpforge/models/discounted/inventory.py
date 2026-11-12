# Source: Winston, W. L. and Goldberg, J. B. (2004). Operations Research: Applications and Algorithms, 4th ed.

import numpy as np
import scipy.stats as st
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "sizes": {
        "small": {
            "requested_state_dim": 2600,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 110001,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 332800,
            "parameters": {},
        },
    },
    "category": "asset_management",
    "description": "Stock replenishment under random demand.",
    "reference": (
        "Winston, W. L. and Goldberg, J. B. (2004). Operations Research: "
        "Applications and Algorithms, 4th ed."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 500, action_dim: int = 10):
        self.requested_state_dim = int(state_dim)
        self._stock_size = (state_dim - 1) // 2  # Size of the stock
        self.max_action = action_dim - 1  # Max Number of action - 1
        self.state_dim = 2 * self._stock_size + 1
        self.action_dim = self.max_action + 1
        self.BinomialeP = 0.4  #
        self.BinomialeQ = 2  # Size of the demand
        self._holding_cost = 4  # holding cost
        self._shortage_cost = 2  # shortageCost
        self._action_cost = 5  # action cost

        self.name = "{}_{}_inventory".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.state_dim = 2 * self._stock_size + 1
        self.action_dim = self.max_action + 1

        self._build_transition_matrix()
        self._build_reward_matrix()

    def _build_transition_matrix(self):
        demand = np.arange(self.BinomialeQ + 1)
        probabilities = st.binom.pmf(demand, self.BinomialeQ, self.BinomialeP)
        # Keep the original ascending-demand addition order at the lower bound.
        tails = np.array([np.cumsum(probabilities[y:])[-1] for y in demand])
        states = np.arange(self.state_dim)
        self.transition_matrix = []
        for action in range(self.action_dim):
            after_order = np.minimum(states + action, self.state_dim - 1)
            bound = np.minimum(self.BinomialeQ, after_order)
            targets = after_order[:, None] - demand
            valid = demand <= bound[:, None]
            values = np.broadcast_to(probabilities, targets.shape).copy()
            saturated = bound < self.BinomialeQ
            values[states[saturated], bound[saturated]] = tails[bound[saturated]]
            rows = np.broadcast_to(states[:, None], targets.shape)
            matrix = csr_matrix(
                (values[valid], (rows[valid], targets[valid])),
                shape=(self.state_dim, self.state_dim),
            )
            matrix.eliminate_zeros()
            self.transition_matrix.append(matrix)

    def _build_reward_matrix(self):
        stock = np.arange(self.state_dim) - self._stock_size
        costs = np.where(
            stock < 0, -self._shortage_cost * stock, self._holding_cost * stock
        )
        self.reward_matrix = np.asarray(
            costs[:, None] + np.arange(self.action_dim) * self._action_cost,
            dtype=float,
        )
