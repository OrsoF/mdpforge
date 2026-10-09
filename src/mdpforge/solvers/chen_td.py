"""
This code allows to slice a state space progressively
along an evolving value function.
"""

from time import time

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import (
    compact_optimal_bellman_operator,
    norminf,
)
from mdpforge.core.partition import Partition


class Solver:
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        verbose: bool = False,
        iter_agg: int = 40,
        iter_bellman: int = 100,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        # Class arguments
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.verbose = verbose
        self.iter_agg = iter_agg
        self.iter_bellman = iter_bellman

        self.name = "Chen"
        self.policy = None
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )

        # Variables
        self.partition = Partition(self.model)
        self.contracted_value: np.ndarray

        self.alpha_function = lambda t: 0.001 / (t + 1)

    def run(self):
        start_time = time()
        tolerance = self.epsilon * (1 - self.discount)
        self.value = np.zeros((self.model.state_dim))

        n = 1
        while True:
            n += 1
            for _ in range(self.iter_bellman):
                new_value = compact_optimal_bellman_operator(
                    self.model, self.value, self.discount, self.shared_reward
                )
                if norminf(new_value - self.value) < tolerance:
                    self.runtime = time() - start_time
                    return
                self.value = new_value

            self.partition = Partition(self.model)
            self.partition.refine_by_width(self.value, self.epsilon / (n + 1))
            self.contracted_value = self.partition.weights.dot(self.value)
            alpha = self.alpha_function(n * (self.iter_agg + self.iter_bellman))

            for _ in range(self.iter_agg):
                self.extended_value = self.partition.phi.dot(self.contracted_value)
                samples = np.fromiter(
                    (
                        region[np.random.randint(len(region))]
                        for region in self.partition.states_in_region
                    ),
                    dtype=int,
                    count=self.partition.n_regions,
                )
                q = np.empty((self.partition.n_regions, self.model.action_dim))
                for aa in range(self.model.action_dim):
                    q[:, aa] = self.model.reward_matrix[samples, aa] + self.discount * (
                        self.model.transition_matrix[aa][samples].dot(
                            self.extended_value
                        )
                    )
                self.contracted_value = (
                    1 - alpha
                ) * self.contracted_value + alpha * q.max(axis=1)

            self.value = self.partition.phi.dot(self.contracted_value)
