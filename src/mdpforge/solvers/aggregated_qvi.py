"""
Implementation of Q-Value Iteration.
Here, the Q-value is updated by blocks
following an iteratively refined partition.
"""

from time import time

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import (
    compact_optimal_bellman_operator,
    norminf,
    q_optimal_bellman_operator,
)
from mdpforge.core.partition import Partition
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.bellman import (
    apply_obo_until_var_small,
    optimal_bellman_residual,
)
from mdpforge.utils.projected_bellman import (
    apply_poqbo_until_var_small,
    projected_optimal_q_bellman_operator,
)


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        verbose: bool = False,
        bellman_updates: int = 50,
        projected_steps: int = 100,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        # Class arguments
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.projected_steps = projected_steps
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )
        self.name = "PDQVI"
        self.policy = None

        self.partition = Partition(self.model)

        self.epsilon_pbr = self.epsilon * (1 - self.discount) / 2
        self.epsilon_span = self.epsilon * (1 - self.discount) / 2

    def run(self):
        start_time = time()
        contracted_q_value = np.zeros(
            (len(self.partition.states_in_region), self.model.action_dim)
        )

        while True:
            if self.partition.n_regions > 0.9 * self.model.state_dim:
                self.value = self.partition.phi.dot(contracted_q_value).max(axis=1)
                self.value, _ = apply_obo_until_var_small(
                    self.model,
                    self.discount,
                    self.epsilon * (1 - self.discount),
                    self.value,
                    shift_acceleration=True,
                )
                self.q_value = q_optimal_bellman_operator(
                    self.model, self.value[:, None], self.discount
                )
                self._finish(self.q_value, start_time)
                return

            self.partition.compute_agg_trans_reward_q()

            contracted_q_value, pbr_value = apply_poqbo_until_var_small(
                self.model,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.epsilon_pbr,
                contracted_q_value,
                max_steps=self.projected_steps,
                shift_acceleration=True,
            )

            q_value = self.partition.phi.dot(contracted_q_value)
            bellman_of_q_value = self._bellman_steps(q_value)
            n_regions = self.partition.n_regions
            self.partition.refine_by_width(bellman_of_q_value, self.epsilon_span)
            refined = self.partition.n_regions > n_regions

            if refined:
                contracted_q_value = self.partition.weights @ bellman_of_q_value

                self.partition.compute_agg_trans_reward_q()
                projected_q_bellman_value = projected_optimal_q_bellman_operator(
                    self.model,
                    self.discount,
                    contracted_q_value,
                    self.partition.aggregate_transition_matrix,
                    self.partition.aggregate_reward_matrix,
                )
                pbr_value = norminf(projected_q_bellman_value - contracted_q_value)

            if not refined and pbr_value <= self.epsilon_pbr:
                q_value = self.partition.phi.dot(contracted_q_value)
                if optimal_bellman_residual(
                    self.model, q_value.max(axis=1), self.discount
                ) <= self.epsilon * (1 - self.discount):
                    self.contracted_q_value = contracted_q_value
                    self._finish(q_value, start_time)
                    return

    def _bellman_steps(self, q_value: np.ndarray) -> np.ndarray:
        value = q_value.max(axis=1)
        for _ in range(self.bellman_updates):
            value = compact_optimal_bellman_operator(
                self.model, value, self.discount, self.shared_reward
            )
        return q_optimal_bellman_operator(self.model, value[:, None], self.discount)

    def _finish(self, q_value: np.ndarray, start_time: float):
        self.q_value = q_value
        self.value = q_value.max(axis=1)
        self.policy = q_value.argmax(axis=1)
        self.runtime = time() - start_time
