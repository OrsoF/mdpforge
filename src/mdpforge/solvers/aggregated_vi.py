from time import time

import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.core.operators import bellman_operator, compact_optimal_bellman_operator
from mdpforge.core.partition import Partition
from mdpforge.utils.bellman import (
    apply_obo_until_var_small,
    optimal_bellman_residual,
)
from mdpforge.utils.projected_bellman import (
    aggregation_norm_weights,
    apply_pobo_until_var_small,
    projected_optimal_bellman_operator_residual,
)


class Solver:
    solver_type = "vi"

    norm_methods = {
        "region_size",
        "average_reward",
        "average_average_reward",
        "max_reward_region",
    }

    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-3,
        verbose: bool = False,
        bellman_updates: int = 10,
        refinement: str = "width",
        norm_method: str = "region_size",
        with_norm_weights: bool = False,
        n_tiles: int | None = None,
        projected_steps: int = 100,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        if refinement not in {"width", "tiles"}:
            raise ValueError("refinement must be 'width' or 'tiles'")
        self._check_norm_method(norm_method)

        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.refinement = refinement
        self.norm_method = norm_method
        self.with_norm_weights = with_norm_weights
        self.n_tiles = n_tiles
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )

        self.name = self._name()
        self.policy = None

        self.partition = Partition(self.model)
        self.epsilon_pbr = self.epsilon * (1 - self.discount) / 2
        self.epsilon_span = self.epsilon * (1 - self.discount) / 2
        self.projected_steps = projected_steps

    def run(self):
        start_time = time()
        contracted_value = np.zeros(self.partition.n_regions)

        while True:
            switch_to_vi_condition = (
                self.partition.n_regions > 0.9 * self.model.state_dim
            )
            if switch_to_vi_condition:
                value = self.partition.phi @ contracted_value
                value, _ = apply_obo_until_var_small(
                    self.model, self.discount, 2 * self.epsilon_pbr, value
                )

                self._finish(value, start_time)
                return

            self.partition.compute_agg_trans_reward_v()
            norm_weights = self._norm_weights()
            contracted_value, pbr_value = apply_pobo_until_var_small(
                self.model,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.partition.weights,
                self.epsilon_pbr,
                contracted_value,
                norm_weights,
                max_steps=self.projected_steps,
                shift_acceleration=True,
            )

            bellman_value = self._bellman_steps(contracted_value)
            contracted_value = self.partition.weights @ bellman_value
            n_regions = self.partition.n_regions
            if self.refinement == "width":
                span = max(self.partition.span(bellman_value)) if self.verbose else None
                self._refine_partition(bellman_value)
                refined = self.partition.n_regions > n_regions
            else:
                span = max(self.partition.span(bellman_value))
                refined = span >= self.epsilon_span
                if refined:
                    self._refine_partition(bellman_value)

            if self.verbose:
                print("Maximum span : {}".format(span))
                print("K : {}".format(n_regions))

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
                value = self.partition.phi @ contracted_value
                if optimal_bellman_residual(
                    self.model, value, self.discount
                ) <= self.epsilon * (1 - self.discount):
                    self._finish(value, start_time)
                    return

    def _bellman_steps(self, contracted_value: np.ndarray) -> np.ndarray:
        """Optimal Bellman operator."""
        value = self.partition.phi @ contracted_value
        bellman_value = compact_optimal_bellman_operator(
            self.model, value, self.discount, self.shared_reward
        )
        for _ in range(self.bellman_updates):
            bellman_value = compact_optimal_bellman_operator(
                self.model, bellman_value, self.discount, self.shared_reward
            )
        return bellman_value

    def _refine_partition(self, bellman_value: np.ndarray) -> np.ndarray:
        if self.refinement == "tiles":
            n_tiles = self.n_tiles or int(self.model.state_dim ** (1 / 4))
            return self.partition.refine_by_tiles(bellman_value, n_tiles)
        return self.partition.refine_by_width(bellman_value, self.epsilon_span)

    def _norm_weights(self) -> np.ndarray | None:
        if not self.with_norm_weights:
            return None
        return aggregation_norm_weights(self.model, self.partition, self.norm_method)

    def _check_norm_method(self, norm_method: str):
        if norm_method not in self.norm_methods:
            raise ValueError(f"norm_method must be one of {sorted(self.norm_methods)}")

    def _finish(self, value: np.ndarray, start_time: float):
        self.value = value
        self.policy = bellman_operator(self.model, value, self.discount).argmax(axis=1)
        self.runtime = time() - start_time

    def _name(self) -> str:
        if self.with_norm_weights:
            return f"PDVI_{self.norm_method}_{self.refinement}"
        return "PDVItiles" if self.refinement == "tiles" else "PDVI"
