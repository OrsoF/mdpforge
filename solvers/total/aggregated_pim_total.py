"""
Implementation of modified Policy Iteration
with a changed Policy Evaluation step.

The new Policy Evaluation successively slices
the state space to update by block the current
value function V^pi.
"""

import time

import numpy as np
from scipy.sparse import vstack

from core.model import SPARSE, GenericModel
from core.operators import compute_transition_reward_policy
from core.partition import Partition
from utils.bellman import (
    bellman_policy_operator,
    iterative_policy_evaluation,
)
from utils.projected_bellman import apply_ppbo_until_var_small


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-2,
        verbose: bool = False,
        bellman_updates: int = 15,
        split_method: str = "width",
        n_tiles: int | None = None,
        projected_steps: int = 100,
    ):
        self.model = model
        self.discount = discount
        self.epsilon_final_policy_evaluation = final_precision
        self.epsilon_policy_evaluation = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.split_method = split_method
        self.n_tiles = (
            int(model.state_dim**0.5) + 1 if n_tiles is None else int(n_tiles)
        )
        if self.split_method not in {"width", "tiles"}:
            raise ValueError("split_method must be 'width' or 'tiles'.")

        self.projected_steps = projected_steps

        self.model._convert_model(SPARSE)
        self._transition_stack = vstack(self.model.transition_matrix, format="csr")
        self._reward_by_action = np.ascontiguousarray(self.model.reward_matrix.T)
        self.name = "PDPIM" if self.split_method == "width" else "PDPIMtiles"
        self.partition = Partition(self.model)
        self._partition_ready = False

    def run(self):
        start_time = time.perf_counter()
        self.policy = np.zeros(self.model.state_dim, dtype=int)
        self.value = np.zeros(self.model.state_dim)
        self.contracted_value = np.zeros(self.partition.n_regions)
        states = np.arange(self.model.state_dim)

        while True:
            self.value = self._policy_evaluation(
                self.policy,
                self.epsilon_policy_evaluation,
                self.value,
            )

            q_value = self._q_value(self.value)
            greedy = q_value.argmax(axis=0)
            new_value = q_value[greedy, states]
            improve = new_value > q_value[self.policy, states] + 1e-12
            new_policy = np.where(improve, greedy, self.policy)

            condition_policy = np.all(new_policy == self.policy)

            self.value = new_value

            if condition_policy:
                self._finish(start_time)
                return

            self.policy = new_policy

    def _bellman_policy_value(
        self,
        contracted_value: np.ndarray,
        transition_policy: np.ndarray,
        reward_policy: np.ndarray,
    ) -> np.ndarray:
        value = self.partition.phi @ contracted_value
        return bellman_policy_operator(
            value, self.discount, transition_policy, reward_policy
        )

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsilon_policy_evaluation: float,
        initial_full_value: np.ndarray,
    ) -> np.ndarray:
        epsilon_pbr = epsilon_policy_evaluation / 2
        epsilon_span = epsilon_policy_evaluation / 2

        contracted_value = self.partition.weights @ initial_full_value
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        if self._partition_ready:
            self.partition.compute_agg_trans_reward_pi(transition_policy, reward_policy)
            if self.partition.n_regions > 1:
                contracted_value, _ = apply_ppbo_until_var_small(
                    self.discount,
                    self.partition.aggregate_transition_policy,
                    self.partition.aggregate_reward_policy,
                    epsilon_pbr,
                    contracted_value,
                    self.projected_steps,
                )
            return self.partition.phi @ contracted_value

        while True:
            if self.partition.n_regions > 0.9 * self.model.state_dim:
                value = self.partition.phi @ contracted_value
                return iterative_policy_evaluation(
                    transition_policy,
                    reward_policy,
                    self.discount,
                    epsilon_policy_evaluation,
                    value,
                )

            self.partition.compute_agg_trans_reward_pi(transition_policy, reward_policy)

            pbr_value = np.inf
            if self.partition.n_regions > 1:
                contracted_value, pbr_value = apply_ppbo_until_var_small(
                    self.discount,
                    self.partition.aggregate_transition_policy,
                    self.partition.aggregate_reward_policy,
                    epsilon_pbr,
                    contracted_value,
                    self.projected_steps,
                )

            bellman_of_value = self._bellman_policy_value(
                contracted_value, transition_policy, reward_policy
            )
            bellman_of_value = self._bellman_steps(bellman_of_value)
            span = max(self.partition.span(bellman_of_value))

            if span <= epsilon_span and pbr_value <= epsilon_pbr:
                self._partition_ready = True
                return self.partition.phi @ contracted_value

            if self.split_method == "width":
                parent = self.partition.refine_by_width(bellman_of_value, epsilon_span)
            else:
                parent = self.partition.refine_by_tiles(bellman_of_value, self.n_tiles)
            contracted_value = contracted_value[parent]

            self.partition.compute_agg_trans_reward_pi(transition_policy, reward_policy)
            if self.partition.n_regions > 1:
                contracted_value, _ = apply_ppbo_until_var_small(
                    self.discount,
                    self.partition.aggregate_transition_policy,
                    self.partition.aggregate_reward_policy,
                    epsilon_pbr,
                    contracted_value,
                    self.projected_steps,
                )
            self._partition_ready = True
            return self.partition.phi @ contracted_value

    def _bellman_steps(self, value: np.ndarray) -> np.ndarray:
        for _ in range(self.bellman_updates):
            value = self._q_value(value).max(axis=0)
        return value

    def _q_value(self, value: np.ndarray) -> np.ndarray:
        q_value = self._transition_stack.dot(value).reshape(
            self.model.action_dim, self.model.state_dim
        )
        q_value *= self.discount
        q_value += self._reward_by_action
        return q_value

    def _finish(self, start_time: float):
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, self.policy
        )
        self.value = iterative_policy_evaluation(
            transition_policy,
            reward_policy,
            self.discount,
            self.epsilon_final_policy_evaluation,
            self.value,
        )

        while True:
            q_value = self._q_value(self.value)
            new_value = q_value.max(axis=0)
            if (
                np.max(np.abs(new_value - self.value))
                < self.epsilon_final_policy_evaluation
            ):
                self.policy = q_value.argmax(axis=0)
                self.value = new_value
                break
            self.value = new_value

        self.runtime = time.perf_counter() - start_time
