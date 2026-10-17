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
from mdpforge.utils.solver_options import (
    bounded_iterations,
    positive_integer,
    positive_real,
)

METADATA = {
    "display_name": "Aggregated QVI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Q-value iteration with progressive state aggregation, synchronous "
        "Bellman backups and a full-state VI fallback."
    ),
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        bellman_iterations: int = 50,
        max_projected_iterations: int = 100,
        *,
        max_iterations: int = 100_000,
        max_fallback_iterations: int = 100_000,
        switch_to_vi_ratio: float = 0.9,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        bellman_iterations: int > 0; Bellman updates per batch; default 50.
        max_projected_iterations: int > 0; projected iteration budget per batch; default
            100.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        max_fallback_iterations: int > 0; full-state iteration budget; default 100_000.
        switch_to_vi_ratio: float in (0, 1]; region fraction triggering full-state VI;
            default 0.9.
        """
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        # Class arguments
        positive_real(final_precision, "final_precision")
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        self.max_fallback_iterations = positive_integer(
            max_fallback_iterations, "max_fallback_iterations"
        )
        self.switch_to_vi_ratio = positive_real(
            switch_to_vi_ratio, "switch_to_vi_ratio"
        )
        if self.switch_to_vi_ratio > 1:
            raise ValueError("switch_to_vi_ratio must belong to (0, 1]")
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.bellman_iterations = positive_integer(
            bellman_iterations, "bellman_iterations"
        )
        self.max_projected_iterations = positive_integer(
            max_projected_iterations, "max_projected_iterations"
        )
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

        for _ in bounded_iterations(self.max_iterations, "aggregation"):
            if (
                self.partition.n_regions
                > self.switch_to_vi_ratio * self.model.state_dim
            ):
                self.value = self.partition.phi.dot(contracted_q_value).max(axis=1)
                self.value, _ = apply_obo_until_var_small(
                    self.model,
                    self.discount,
                    self.epsilon * (1 - self.discount),
                    self.value,
                    shift_acceleration=True,
                    max_iterations=self.max_fallback_iterations,
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
                max_steps=self.max_projected_iterations,
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
        for _ in range(self.bellman_iterations):
            value = compact_optimal_bellman_operator(
                self.model, value, self.discount, self.shared_reward
            )
        return q_optimal_bellman_operator(self.model, value[:, None], self.discount)

    def _finish(self, q_value: np.ndarray, start_time: float):
        self.q_value = q_value
        self.value = q_value.max(axis=1)
        self.policy = q_value.argmax(axis=1)
        self.runtime = time() - start_time
