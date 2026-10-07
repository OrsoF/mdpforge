"""
Implementation of modified Policy Iteration
with a changed Policy Evaluation step.

The new Policy Evaluation successively slices
the state space to update by block the current
value function V^pi.
"""

import time

import numpy as np

from core.model import SPARSE, GenericModel
from core.operators import compute_transition_reward_policy
from core.partition import Partition
from utils.bellman import (
    bellman_operator,
    bellman_policy_operator,
    compact_optimal_bellman_operator,
    iterative_policy_evaluation,
    norminf,
)
from utils.projected_bellman import (
    apply_ppbo_until_var_small,
)


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-1,
        verbose: bool = False,
        bellman_updates: int = 10,
        split_method: str = "width",
        n_tiles: int | None = None,
        projected_steps: int = 100,
    ):
        self.model = model
        self.discount = discount

        self.epsilon_final_policy_evaluation = final_precision
        self.epsilon_variation = final_precision
        self.epsilon_policy_evaluation = final_precision

        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.split_method = split_method

        self.n_tiles = (
            int(model.state_dim**0.5) + 1 if n_tiles is None else int(n_tiles)
        )
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )

        assert self.discount < 1.0, "Use aggregated_pim_total instead."

        if self.split_method not in {"width", "tiles"}:
            raise ValueError("split_method must be 'width' or 'tiles'.")

        self.projected_steps = projected_steps

        self.model._convert_model(SPARSE)

        self.name = "PDPIM" if self.split_method == "width" else "PDPIMtiles"

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

        while True:
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

            condition_policy = np.all(new_policy == self.policy)

            self.value = new_value

            if condition_variation or condition_policy:
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

        while True:
            if self.partition.n_regions > 0.9 * self.model.state_dim:
                value = self.partition.phi @ contracted_value

                value = iterative_policy_evaluation(
                    transition_policy,
                    reward_policy,
                    self.discount,
                    epsilon_policy_evaluation,
                    value,
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
                self.projected_steps,
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

            if self.split_method == "width":
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
        for _ in range(self.bellman_updates):
            value = compact_optimal_bellman_operator(
                self.model, value, self.discount, self.shared_reward
            )

        return value
