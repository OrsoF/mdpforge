"""Total-reward MPI with progressively refined aggregate policy evaluation."""

import numpy as np

from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._aggregation import (
    refine_partition,
    terminal_partition,
)
from mdpforge.solvers.total._common import check_options, proper_initial_policy
from mdpforge.utils.projected_bellman import projected_policy_bellman_operator
from mdpforge.utils.solver_options import validate_aggregation_options

METADATA = {
    "display_name": "Total Aggregated MPI · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Total-reward modified policy iteration with progressively refined "
        "aggregate policy evaluation."
    ),
}


class Solver:
    """Alternate aggregate policy evaluation and full-state greedy improvement."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        bellman_iterations=25,
        max_projected_iter=10,
        refinement="width",
        n_tiles=None,
        *,
        max_policy_iter=100_000,
        max_eval_iter=100_000,
        switch_to_vi_ratio=0.9,
        max_agg_iter=100_000,
    ):
        """Configure undiscounted aggregate policy evaluation.

        final_precision: finite float > 0; full Bellman-residual tolerance.
        bellman_iterations: int > 0; full-state policy backups per batch.
        refinement: "width" or "tiles"; partition splitting rule.
        n_tiles: None for automatic sizing, or int > 0; tiles per region.
        max_projected_iter: int > 0; projected policy backups per batch.
        max_policy_iter: int > 0; policy improvement budget.
        max_eval_iter: int > 0; full-state policy evaluation budget.
        switch_to_vi_ratio: float in (0, 1]; region fraction triggering fallback.
        max_agg_iter: int > 0; aggregation batches per evaluation.
        """
        if type(max_policy_iter) is not int or max_policy_iter < 1:
            raise ValueError("max_policy_iter must be a positive integer")
        validate_aggregation_options(
            final_precision=final_precision,
            bellman_iterations=bellman_iterations,
            max_projected_iter=max_projected_iter,
            max_agg_iter=max_agg_iter,
            max_eval_iter=max_eval_iter,
            switch_to_vi_ratio=switch_to_vi_ratio,
            refinement=refinement,
            n_tiles=n_tiles,
        )
        self.terminal = check_options(model, final_precision, max_policy_iter)
        self.model = model
        self.final_precision = final_precision
        self.bellman_iterations = bellman_iterations
        self.refinement = refinement
        self.n_tiles = n_tiles
        self.max_projected_iter = max_projected_iter
        self.max_policy_iter = max_policy_iter
        self.max_eval_iter = max_eval_iter
        self.switch_to_vi_ratio = switch_to_vi_ratio
        self.max_agg_iter = max_agg_iter
        self.partition = terminal_partition(model, self.terminal)
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Initialize available terminal paths and check the full residual."""
        policy = proper_initial_policy(
            self.model, self.terminal, require_all_states=False
        )
        value = np.zeros(self.model.state_dim)
        for _ in range(self.max_policy_iter):
            transition, reward = compute_transition_reward_policy(self.model, policy)
            value = self._policy_evaluation(transition, reward, value)
            q_value = bellman_operator(self.model, value, 1.0)
            if not np.isfinite(q_value).all():
                raise RuntimeError("AGGMPI stopped: non-finite values")
            policy = q_value.argmax(axis=1)
            if np.max(np.abs(q_value.max(axis=1) - value)) <= self.final_precision:
                return SolveResult(value, policy, validation_method="residual")
        raise RuntimeError("AGGMPI stopped: max_policy_iter reached")

    def _policy_evaluation(self, transition, reward, value):
        """Refine projected policy values until the full policy residual is small."""
        partition = self.partition
        contracted_value = partition.weights @ value
        for _ in range(self.max_agg_iter):
            if partition.n_regions > self.switch_to_vi_ratio * self.model.state_dim:
                return self._full_policy_evaluation(
                    transition, reward, partition.phi @ contracted_value
                )

            agg_transition, agg_reward = partition.compute_agg_trans_reward_pi(
                transition, reward
            )
            for _ in range(self.max_projected_iter):
                next_value = projected_policy_bellman_operator(
                    1.0, contracted_value, agg_transition, agg_reward
                )
                if not np.isfinite(next_value).all():
                    raise RuntimeError("AGGMPI stopped: non-finite values")
                variation = np.max(np.abs(next_value - contracted_value))
                contracted_value = next_value
                if variation <= self.final_precision / 2:
                    break

            value = partition.phi @ contracted_value
            for _ in range(self.bellman_iterations):
                value = reward + transition.dot(value)
            next_value = reward + transition.dot(value)
            if not np.isfinite(next_value).all():
                raise RuntimeError("AGGMPI stopped: non-finite values")
            if np.max(np.abs(next_value - value)) <= self.final_precision / 2:
                return value

            refined = refine_partition(
                partition, value, self.final_precision, self.refinement, self.n_tiles
            )
            if not refined:
                return self._full_policy_evaluation(transition, reward, value)
            contracted_value = partition.weights @ value
        raise RuntimeError("AGGMPI stopped: max_agg_iter reached")

    def _full_policy_evaluation(self, transition, reward, value):
        """Finish policy evaluation without aggregation or discounted shifts."""
        for _ in range(self.max_eval_iter):
            next_value = reward + transition.dot(value)
            if not np.isfinite(next_value).all():
                raise RuntimeError("AGGMPI stopped: non-finite values")
            if np.max(np.abs(next_value - value)) <= self.final_precision / 2:
                return value
            value = next_value
        raise RuntimeError("AGGMPI stopped: max_eval_iter reached")
