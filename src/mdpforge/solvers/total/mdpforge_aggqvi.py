"""Total-reward QVI with progressive state aggregation and a VI fallback."""

import numpy as np

from mdpforge.core.operators import bellman_operator, q_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._aggregation import (
    check_aggregation_options,
    full_value_iteration,
    refine_partition,
    terminal_partition,
)
from mdpforge.solvers.total._common import check_options
from mdpforge.utils.projected_bellman import projected_optimal_q_bellman_operator

METADATA = {
    "display_name": "Total Aggregated QVI · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Total-reward Q-value iteration with progressive state aggregation "
        "and a full-state VI fallback."
    ),
}


class Solver:
    """Refine state-action values with a separate zero-valued terminal region."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        bellman_iterations=25,
        max_projected_iterations=10,
        *,
        max_iterations=100_000,
        max_fallback_iterations=100_000,
        switch_to_vi_ratio=0.9,
    ):
        """Configure undiscounted Q-value aggregation.

        final_precision: finite float > 0; full Bellman-residual tolerance.
        bellman_iterations: int > 0; full-state Q backups per batch.
        max_projected_iterations: int > 0; projected Q backups per batch.
        max_iterations: int > 0; aggregation batch budget.
        max_fallback_iterations: int > 0; full-state VI budget.
        switch_to_vi_ratio: float in (0, 1]; region fraction triggering VI.
        """
        terminal = check_options(model, final_precision, max_iterations)
        check_aggregation_options(
            bellman_iterations,
            max_projected_iterations,
            max_fallback_iterations,
            switch_to_vi_ratio,
        )
        self.model = model
        self.final_precision = final_precision
        self.bellman_iterations = bellman_iterations
        self.max_projected_iterations = max_projected_iterations
        self.max_iterations = max_iterations
        self.max_fallback_iterations = max_fallback_iterations
        self.switch_to_vi_ratio = switch_to_vi_ratio
        self.partition = terminal_partition(model, terminal)
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Start at zero and stop when the full Bellman residual is small."""
        partition = self.partition
        contracted_q = np.zeros((partition.n_regions, self.model.action_dim))
        for _ in range(self.max_iterations):
            if partition.n_regions > self.switch_to_vi_ratio * self.model.state_dim:
                return full_value_iteration(
                    self.model,
                    (partition.phi @ contracted_q).max(axis=1),
                    self.final_precision,
                    self.max_fallback_iterations,
                )

            transition, reward = partition.compute_agg_trans_reward_q()
            for _ in range(self.max_projected_iterations):
                next_q = projected_optimal_q_bellman_operator(
                    self.model, 1.0, contracted_q, transition, reward
                )
                if not np.isfinite(next_q).all():
                    raise RuntimeError("AGGQVI stopped: non-finite values")
                variation = np.max(np.abs(next_q - contracted_q))
                contracted_q = next_q
                if variation <= self.final_precision / 2:
                    break

            q_value = partition.phi @ contracted_q
            for _ in range(self.bellman_iterations):
                q_value = q_optimal_bellman_operator(self.model, q_value, 1.0)
            value = q_value.max(axis=1)
            greedy_q = bellman_operator(self.model, value, 1.0)
            if not np.isfinite(greedy_q).all():
                raise RuntimeError("AGGQVI stopped: non-finite values")
            if np.max(np.abs(greedy_q.max(axis=1) - value)) <= self.final_precision:
                return SolveResult(
                    value, greedy_q.argmax(axis=1), validation_method="residual"
                )

            if not refine_partition(partition, q_value, self.final_precision):
                return full_value_iteration(
                    self.model,
                    value,
                    self.final_precision,
                    self.max_fallback_iterations,
                )
            contracted_q = partition.weights @ q_value
        raise RuntimeError("AGGQVI stopped: max_iterations reached")
