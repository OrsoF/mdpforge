from time import time

import numpy as np

from core.model import SPARSE, GenericModel
from core.partition import Partition
from utils.bellman import (
    apply_obo_until_var_small,
    bellman_no_max,
    compact_optimal_bellman_operator,
)
from utils.projected_bellman import (
    apply_pobo_until_var_small,
    projected_optimal_bellman_operator_residual,
)


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-2,
        verbose: bool = False,
        bellman_updates: int = 100,
        projected_steps: int = 100,
    ):
        self.model = model
        self.discount = 1.0
        self.epsilon = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.projected_steps = projected_steps
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )

        self.model._convert_model(SPARSE)
        self.name = "PDVI"

        self.partition = Partition(self.model)

        self.epsilon_pbr = self.epsilon / 2
        self.epsilon_span = self.epsilon / 2

    def run(self):
        start_time = time()
        contracted_value = np.zeros(self.partition.n_regions)

        while True:
            if self.partition.n_regions > 0.9 * self.model.state_dim:
                value = self.partition.phi @ contracted_value
                value, _ = apply_obo_until_var_small(
                    self.model, self.discount, 2 * self.epsilon_pbr, value
                )
                self._finish(value, start_time)
                return

            self.partition.compute_agg_trans_reward_v()

            contracted_value, pbr_value = apply_pobo_until_var_small(
                self.model,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.partition.weights,
                self.epsilon_pbr * 100,
                contracted_value,
                max_steps=self.projected_steps,
            )

            bellman_value = self._bellman_steps(contracted_value)
            n_regions = self.partition.n_regions
            self.partition.refine_by_width(bellman_value, self.epsilon_span)
            refined = self.partition.n_regions > n_regions

            if refined:
                contracted_value = self.partition.weights @ bellman_value
                self.partition.compute_agg_trans_reward_v()

                pbr_value = projected_optimal_bellman_operator_residual(
                    self.model,
                    self.discount,
                    contracted_value,
                    self.partition.aggregate_transition_matrix,
                    self.partition.aggregate_reward_matrix,
                    self.partition.weights,
                )

            if not refined and pbr_value <= self.epsilon_pbr:
                self._finish(self.partition.phi @ contracted_value, start_time)
                return

    def _bellman_steps(self, contracted_value: np.ndarray) -> np.ndarray:
        value = self.partition.phi @ contracted_value
        bellman_of_value = compact_optimal_bellman_operator(
            self.model, value, self.discount, self.shared_reward
        )
        for _ in range(self.bellman_updates):
            bellman_of_value = compact_optimal_bellman_operator(
                self.model, bellman_of_value, self.discount, self.shared_reward
            )

        return bellman_of_value

    def _finish(self, value: np.ndarray, start_time: float):
        self.value = value
        self.policy = bellman_no_max(self.model, value, self.discount).argmax(axis=1)
        self.runtime = time() - start_time
