"""Total-reward VI with progressive state aggregation and full-state fallback."""

import numpy as np

from mdpforge.core.operators import bellman_operator, optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._aggregation import (
    check_aggregation_options,
    full_value_iteration,
    refine_partition,
    terminal_partition,
)
from mdpforge.solvers.total._common import check_options
from mdpforge.utils.projected_bellman import projected_optimal_bellman_operator

METADATA = {
    "display_name": "Total Aggregated VI · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Total-reward value iteration with progressive state aggregation "
        "and a full-state VI fallback."
    ),
}


class Solver:
    """Refine aggregate values while keeping terminal values fixed at zero."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        bellman_iterations=25,
        max_projected_iterations=10,
        refinement="width",
        n_tiles=None,
        *,
        max_iterations=100_000,
        max_fallback_iterations=100_000,
        switch_to_vi_ratio=0.9,
    ):
        """Configure undiscounted aggregation.

        final_precision: finite float > 0; full Bellman-residual tolerance.
        bellman_iterations: int > 0; full-state backups per batch.
        refinement: "width" or "tiles"; partition splitting rule.
        n_tiles: None for automatic sizing, or int > 0; tiles per region.
        max_projected_iterations: int > 0; projected backups per batch.
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
            refinement,
            n_tiles,
        )
        self.model = model
        self.final_precision = final_precision
        self.bellman_iterations = bellman_iterations
        self.refinement = refinement
        self.n_tiles = n_tiles
        self.max_projected_iterations = max_projected_iterations
        self.max_iterations = max_iterations
        self.max_fallback_iterations = max_fallback_iterations
        self.switch_to_vi_ratio = switch_to_vi_ratio
        self.partition = terminal_partition(model, terminal)
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Start at zero and stop when the full Bellman residual is small."""
        partition = self.partition
        contracted_value = np.zeros(partition.n_regions)
        for _ in range(self.max_iterations):
            if partition.n_regions > self.switch_to_vi_ratio * self.model.state_dim:
                return full_value_iteration(
                    self.model,
                    partition.phi @ contracted_value,
                    self.final_precision,
                    self.max_fallback_iterations,
                )

            transition, reward = partition.compute_agg_trans_reward_v()
            for _ in range(self.max_projected_iterations):
                next_value = projected_optimal_bellman_operator(
                    self.model,
                    1.0,
                    contracted_value,
                    transition,
                    reward,
                    partition.weights,
                )
                if not np.isfinite(next_value).all():
                    raise RuntimeError("AGGVI stopped: non-finite values")
                variation = np.max(np.abs(next_value - contracted_value))
                contracted_value = next_value
                if variation <= self.final_precision / 2:
                    break

            value = partition.phi @ contracted_value
            for _ in range(self.bellman_iterations):
                value = optimal_bellman_operator(self.model, value, 1.0)
            q_value = bellman_operator(self.model, value, 1.0)
            if not np.isfinite(q_value).all():
                raise RuntimeError("AGGVI stopped: non-finite values")
            if np.max(np.abs(q_value.max(axis=1) - value)) <= self.final_precision:
                return SolveResult(
                    value, q_value.argmax(axis=1), validation_method="residual"
                )

            refined = refine_partition(
                partition, value, self.final_precision, self.refinement, self.n_tiles
            )
            if not refined:
                return full_value_iteration(
                    self.model,
                    value,
                    self.final_precision,
                    self.max_fallback_iterations,
                )
            contracted_value = partition.weights @ value
        raise RuntimeError("AGGVI stopped: max_iterations reached")
