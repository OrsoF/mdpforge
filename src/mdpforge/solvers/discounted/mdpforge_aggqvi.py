"""
Implementation of Q-Value Iteration.
Here, the Q-value is updated by blocks
following an iteratively refined partition.
"""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import (
    compact_optimal_bellman_operator,
    norminf,
    q_optimal_bellman_operator,
)
from mdpforge.core.partition import Partition
from mdpforge.core.solve_result import SolveResult
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
    validate_aggregation_options,
    validate_discount,
)

METADATA = {
    "display_name": "Aggregated QVI · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        bellman_iterations: int = 50,
        max_projected_iter: int = 100,
        *,
        max_agg_iter: int = 100_000,
        max_eval_iter: int = 100_000,
        switch_to_vi_ratio: float = 0.9,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        bellman_iterations: int > 0; Bellman updates per batch; default 50.
        max_projected_iter: int > 0; projected iteration budget per batch; default
            100.
        max_agg_iter: int > 0; outer iteration budget; default 100_000.
        max_eval_iter: int > 0; full-state iteration budget; default 100_000.
        switch_to_vi_ratio: float in (0, 1]; region fraction triggering full-state VI;
            default 0.9.
        """
        validate_discount(discount)
        validate_aggregation_options(
            final_precision=final_precision,
            bellman_iterations=bellman_iterations,
            max_projected_iter=max_projected_iter,
            max_agg_iter=max_agg_iter,
            max_eval_iter=max_eval_iter,
            switch_to_vi_ratio=switch_to_vi_ratio,
        )
        self.max_agg_iter = max_agg_iter
        self.max_eval_iter = max_eval_iter
        self.switch_to_vi_ratio = float(switch_to_vi_ratio)
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.bellman_iterations = bellman_iterations
        self.max_projected_iter = max_projected_iter
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )
        self.name = __name__.rsplit(".", 1)[-1]

        self.partition = Partition(self.model)

        self.epsilon_pbr = self.epsilon * (1 - self.discount) / 2
        self.epsilon_span = self.epsilon * (1 - self.discount) / 2

    def run(self) -> SolveResult:
        contracted_q_value = np.zeros(
            (len(self.partition.states_in_region), self.model.action_dim)
        )

        for _ in bounded_iterations(self.max_agg_iter, "aggregation"):
            if (
                self.partition.n_regions
                > self.switch_to_vi_ratio * self.model.state_dim
            ):
                value = self.partition.phi.dot(contracted_q_value).max(axis=1)
                value, _ = apply_obo_until_var_small(
                    self.model,
                    self.discount,
                    self.epsilon * (1 - self.discount),
                    value,
                    shift_acceleration=True,
                    max_iterations=self.max_eval_iter,
                )
                self.q_value = q_optimal_bellman_operator(
                    self.model, value[:, None], self.discount
                )
                return self._finish(self.q_value)

            self.partition.compute_agg_trans_reward_q()

            contracted_q_value, pbr_value = apply_poqbo_until_var_small(
                self.model,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.epsilon_pbr,
                contracted_q_value,
                max_steps=self.max_projected_iter,
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
                    return self._finish(q_value)

    def _bellman_steps(self, q_value: np.ndarray) -> np.ndarray:
        value = q_value.max(axis=1)
        for _ in range(self.bellman_iterations):
            value = compact_optimal_bellman_operator(
                self.model, value, self.discount, self.shared_reward
            )
        return q_optimal_bellman_operator(self.model, value[:, None], self.discount)

    def _finish(self, q_value: np.ndarray) -> SolveResult:
        self.q_value = q_value
        value = q_value.max(axis=1)
        policy = q_value.argmax(axis=1)
        return SolveResult(value, policy, validation_method="residual")
