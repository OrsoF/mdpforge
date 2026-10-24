import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import bellman_operator, compact_optimal_bellman_operator
from mdpforge.core.partition import Partition
from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.bellman import (
    apply_obo_until_var_small,
    optimal_bellman_residual,
)
from mdpforge.utils.projected_bellman import (
    aggregation_norm_weights,
    apply_pobo_until_var_small,
    projected_optimal_bellman_operator_residual,
)
from mdpforge.utils.solver_options import (
    bounded_iterations,
    positive_integer,
    positive_real,
)

METADATA = {
    "display_name": "Aggregated VI · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Value iteration with progressively refined state aggregation, "
        "projected Bellman backups and a full-state VI fallback."
    ),
}


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
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        bellman_iterations: int = 10,
        refinement: str = "width",
        norm_method: str = "region_size",
        with_norm_weights: bool = False,
        n_tiles: int | None = None,
        max_projected_iterations: int = 100,
        *,
        max_iterations: int = 100_000,
        max_fallback_iterations: int = 100_000,
        switch_to_vi_ratio: float = 0.9,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        bellman_iterations: int > 0; Bellman updates per batch; default 10.
        refinement: "width" or "tiles"; partition refinement method; default "width".
        norm_method: "region_size", "average_reward", "average_average_reward", or
        "max_reward_region"; aggregation weights; default "region_size".
        with_norm_weights: bool; enable aggregation normalization weights; default
            False.
        n_tiles: None for automatic sizing, or an int > 0; tiles per refinement; default
            None.
        max_projected_iterations: int > 0; projected iteration budget per batch; default
            100.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        max_fallback_iterations: int > 0; full-state iteration budget; default 100_000.
        switch_to_vi_ratio: float in (0, 1]; region fraction triggering full-state VI;
            default 0.9.
        """
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        if refinement not in {"width", "tiles"}:
            raise ValueError("refinement must be 'width' or 'tiles'")
        self._check_norm_method(norm_method)

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
        self.epsilon = final_precision
        self.bellman_iterations = positive_integer(
            bellman_iterations, "bellman_iterations"
        )
        self.refinement = refinement
        self.norm_method = norm_method
        self.with_norm_weights = with_norm_weights
        self.n_tiles = None if n_tiles is None else positive_integer(n_tiles, "n_tiles")
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )

        self.name = self._name()
        self.policy = None

        self.partition = Partition(self.model)
        self.epsilon_pbr = self.epsilon * (1 - self.discount) / 2
        self.epsilon_span = self.epsilon * (1 - self.discount) / 2
        self.max_projected_iterations = positive_integer(
            max_projected_iterations, "max_projected_iterations"
        )

    def run(self) -> SolveResult:
        contracted_value = np.zeros(self.partition.n_regions)

        for _ in bounded_iterations(self.max_iterations, "aggregation"):
            switch_to_vi_condition = (
                self.partition.n_regions
                > self.switch_to_vi_ratio * self.model.state_dim
            )
            if switch_to_vi_condition:
                value = self.partition.phi @ contracted_value
                value, _ = apply_obo_until_var_small(
                    self.model,
                    self.discount,
                    2 * self.epsilon_pbr,
                    value,
                    max_iterations=self.max_fallback_iterations,
                )

                self._finish(value)
                return SolveResult.from_solver(self)

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
                max_steps=self.max_projected_iterations,
                shift_acceleration=True,
            )

            bellman_value = self._bellman_steps(contracted_value)
            contracted_value = self.partition.weights @ bellman_value
            n_regions = self.partition.n_regions
            if self.refinement == "width":
                self._refine_partition(bellman_value)
                refined = self.partition.n_regions > n_regions
            else:
                span = max(self.partition.span(bellman_value))
                refined = span >= self.epsilon_span
                if refined:
                    self._refine_partition(bellman_value)

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
                    self._finish(value)
                    return SolveResult.from_solver(self)

    def _bellman_steps(self, contracted_value: np.ndarray) -> np.ndarray:
        """Optimal Bellman operator."""
        value = self.partition.phi @ contracted_value
        bellman_value = compact_optimal_bellman_operator(
            self.model, value, self.discount, self.shared_reward
        )
        for _ in range(self.bellman_iterations):
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

    def _finish(self, value: np.ndarray):
        self.value = value
        self.policy = bellman_operator(self.model, value, self.discount).argmax(axis=1)

    def _name(self) -> str:
        if self.with_norm_weights:
            return f"PDVI_{self.norm_method}_{self.refinement}"
        return "PDVItiles" if self.refinement == "tiles" else "PDVI"
