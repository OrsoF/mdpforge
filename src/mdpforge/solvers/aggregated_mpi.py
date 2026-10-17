"""
Implementation of modified Policy Iteration
with a changed Policy Evaluation step.

The new Policy Evaluation successively slices
the state space to update by block the current
value function V^pi.
"""

import time

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import (
    bellman_operator,
    bellman_policy_operator,
    compact_optimal_bellman_operator,
    compute_transition_reward_policy,
    norminf,
)
from mdpforge.core.partition import Partition
from mdpforge.utils.bellman import (
    iterative_policy_evaluation,
)
from mdpforge.utils.projected_bellman import (
    apply_ppbo_until_var_small,
)
from mdpforge.utils.solver_options import (
    bounded_iterations,
    positive_integer,
    positive_real,
)

METADATA = {
    "display_name": "Aggregated MPI · mdpforge",
    "family": "mpi",
    "backend": "mdpforge",
    "description": (
        "Modified policy iteration with progressively refined aggregate "
        "policy evaluation and ground-state Bellman updates."
    ),
    "update_scheme": "synchronous",
}


class Solver:
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-1,
        bellman_iterations: int = 10,
        refinement: str = "width",
        n_tiles: int | None = None,
        max_projected_iterations: int = 100,
        *,
        max_iterations: int = 100_000,
        max_fallback_iterations: int = 100_000,
        switch_to_vi_ratio: float = 0.9,
        max_evaluation_iterations: int = 100_000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-1.
        bellman_iterations: int > 0; Bellman updates per batch; default 10.
        refinement: "width" or "tiles"; partition refinement method; default "width".
        n_tiles: None for automatic sizing, or an int > 0; tiles per refinement; default
            None.
        max_projected_iterations: int > 0; projected iteration budget per batch; default
            100.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        max_fallback_iterations: int > 0; full-state iteration budget; default 100_000.
        switch_to_vi_ratio: float in (0, 1]; region fraction triggering full-state VI;
            default 0.9.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            100_000.
        """
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
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
        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )
        self.model = model
        self.discount = discount

        self.epsilon_variation = final_precision
        self.epsilon_policy_evaluation = final_precision
        self.bellman_iterations = positive_integer(
            bellman_iterations, "bellman_iterations"
        )
        self.refinement = refinement

        self.n_tiles = (
            int(model.state_dim**0.5) + 1
            if n_tiles is None
            else positive_integer(n_tiles, "n_tiles")
        )
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )

        if self.refinement not in {"width", "tiles"}:
            raise ValueError("refinement must be 'width' or 'tiles'.")

        self.max_projected_iterations = positive_integer(
            max_projected_iterations, "max_projected_iterations"
        )

        self.name = "PDMPI" if self.refinement == "width" else "PDMPItiles"

        self.partition = Partition(self.model)

    def run(self):
        start_time = time.perf_counter()

        self.policy = np.zeros(
            self.model.state_dim,
            dtype=int,
        )

        self.value = np.zeros(
            self.model.state_dim,
            dtype=float,
        )

        for _ in bounded_iterations(self.max_iterations, "aggregation"):
            self.value = self._policy_evaluation(
                self.policy,
                self.epsilon_policy_evaluation,
                self.value,
            )

            q_value = bellman_operator(
                self.model,
                self.value,
                self.discount,
            )

            new_policy = q_value.argmax(axis=1)
            new_value = q_value.max(axis=1)

            condition_variation = norminf(
                new_value - self.value
            ) < self.epsilon_variation * (1 - self.discount)

            self.value = new_value

            if condition_variation:
                self.policy = new_policy
                self.runtime = time.perf_counter() - start_time
                break

            self.policy = new_policy

    def _compute_transition_reward_policy(
        self,
        policy: np.ndarray,
    ):
        """
        Construct the transition matrix P_pi and reward vector r_pi
        associated with the current deterministic policy.

        The transition matrix is assembled directly from CSR rows of the
        action-specific transition matrices.

        This avoids:
            - LIL matrix construction,
            - sparse-to-dense conversions,
            - repeated sparse row assignments.
        """
        policy = np.asarray(
            policy,
            dtype=int,
        )

        return compute_transition_reward_policy(self.model, policy)

    def _bellman_policy_value(
        self,
        contracted_value: np.ndarray,
        transition_policy,
        reward_policy: np.ndarray,
    ) -> np.ndarray:
        value = self.partition.phi @ contracted_value

        return bellman_policy_operator(
            value,
            self.discount,
            transition_policy,
            reward_policy,
        )

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsilon_policy_evaluation: float,
        initial_full_value: np.ndarray,
    ) -> np.ndarray:
        epsilon_pbr = (1 - self.discount) * epsilon_policy_evaluation / 2

        epsilon_span = (1 - self.discount) * epsilon_policy_evaluation / 2

        contracted_value = self.partition.weights @ initial_full_value

        transition_policy, reward_policy = self._compute_transition_reward_policy(
            policy
        )

        for _ in bounded_iterations(
            self.max_evaluation_iterations, "policy evaluation"
        ):
            if (
                self.partition.n_regions
                > self.switch_to_vi_ratio * self.model.state_dim
            ):
                value = self.partition.phi @ contracted_value

                value = iterative_policy_evaluation(
                    transition_policy,
                    reward_policy,
                    self.discount,
                    epsilon_policy_evaluation * (1 - self.discount),
                    value,
                    max_iterations=self.max_fallback_iterations,
                )

                return value

            self.partition.compute_agg_trans_reward_pi(
                transition_policy,
                reward_policy,
            )

            contracted_value, pbr_value = apply_ppbo_until_var_small(
                self.discount,
                self.partition.aggregate_transition_policy,
                self.partition.aggregate_reward_policy,
                epsilon_pbr,
                contracted_value,
                self.max_projected_iterations,
                shift_acceleration=True,
            )

            bellman_of_value = self._bellman_policy_value(
                contracted_value,
                transition_policy,
                reward_policy,
            )

            bellman_of_value = self._bellman_steps(bellman_of_value)

            span = max(self.partition.span(bellman_of_value))

            if span <= epsilon_span and pbr_value <= epsilon_pbr:
                self.value = self.partition.phi @ contracted_value

                return self.value

            if self.refinement == "width":
                parent = self.partition.refine_by_width(
                    bellman_of_value,
                    epsilon_span,
                )
            else:
                parent = self.partition.refine_by_tiles(
                    bellman_of_value,
                    self.n_tiles,
                )

            contracted_value = contracted_value[parent]

    def _bellman_steps(
        self,
        value: np.ndarray,
    ) -> np.ndarray:
        for _ in range(self.bellman_iterations):
            value = compact_optimal_bellman_operator(
                self.model, value, self.discount, self.shared_reward
            )

        return value
