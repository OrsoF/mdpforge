# Source: inventory-control benchmark with one-period delivery lead time.

import itertools

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "resource_management",
    "description": "Inventory replenishment with a one-period delivery delay.",
    "reference": (
        "General reference: Puterman (1994). Markov Decision Processes: "
        "Discrete Stochastic Dynamic Programming."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        stock_capacity = int(np.ceil(np.sqrt(max(9, state_dim)))) - 1
        self.stock_capacity = max(2, stock_capacity)
        self.max_order = max(1, min(int(action_dim) - 1, self.stock_capacity))

        self.state_dim = (self.stock_capacity + 1) * (self.max_order + 1)
        self.action_dim = self.max_order + 1

        self.demand_probabilities = {0: 0.2, 1: 0.5, 2: 0.3}
        self.holding_cost = 0.2
        self.shortage_cost = 1.0
        self.order_cost = 0.4

        self.name = "{}_{}_inventory_leadtime_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.stock_capacity,
            self.max_order,
        )

    def _build_model(self):
        self.states = list(
            itertools.product(
                range(self.stock_capacity + 1),
                range(self.max_order + 1),
            )
        )
        self.state_to_index = {state: index for index, state in enumerate(self.states)}

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for state_index, (stock, pipeline) in enumerate(self.states):
            available = min(self.stock_capacity, stock + pipeline)
            for order in range(self.action_dim):
                expected_cost = self.order_cost * order
                for demand, probability in self.demand_probabilities.items():
                    sold = min(available, demand)
                    remaining = available - sold
                    shortage = max(0, demand - available)
                    expected_cost += probability * (
                        self.holding_cost * remaining + self.shortage_cost * shortage
                    )
                    next_state = (remaining, order)
                    self.transition_matrix[order][
                        state_index,
                        self.state_to_index[next_state],
                    ] += probability

                self.reward_matrix[state_index, order] = -expected_cost

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]
