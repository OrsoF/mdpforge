"""
This code allows to slice a state space progressively
along an evolving value function.
"""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import (
    compact_optimal_bellman_operator,
    norminf,
)
from mdpforge.core.partition import Partition
from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.solver_options import (
    bounded_iterations,
    positive_integer,
    positive_real,
)

METADATA = {
    "display_name": "Chen aggregated VI · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Value iteration alternating synchronous ground-state Bellman sweeps "
        "and stochastic approximation on sampled aggregate states."
    ),
}


class Solver:
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        aggregation_iterations: int = 40,
        bellman_iterations: int = 100,
        *,
        max_iterations: int = 100_000,
        step_size: float = 0.001,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        aggregation_iterations: int > 0; aggregate updates per batch; default 40.
        bellman_iterations: int > 0; Bellman updates per batch; default 100.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        step_size: finite float > 0; coefficient of the decreasing learning rate;
            default 0.001.
        """
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        positive_real(final_precision, "final_precision")
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        self.step_size = positive_real(step_size, "step_size")
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.aggregation_iterations = positive_integer(
            aggregation_iterations, "aggregation_iterations"
        )
        self.bellman_iterations = positive_integer(
            bellman_iterations, "bellman_iterations"
        )

        self.name = "Chen"
        self.policy = None
        self.shared_reward = np.all(
            self.model.reward_matrix == self.model.reward_matrix[:, [0]]
        )

        self.partition = Partition(self.model)
        self.contracted_value: np.ndarray

    def run(self) -> SolveResult:
        tolerance = self.epsilon * (1 - self.discount)
        self.value = np.zeros((self.model.state_dim))

        for iteration in bounded_iterations(self.max_iterations, "Chen iteration"):
            n = iteration + 1
            for _ in range(self.bellman_iterations):
                new_value = compact_optimal_bellman_operator(
                    self.model, self.value, self.discount, self.shared_reward
                )
                if norminf(new_value - self.value) < tolerance:
                    return SolveResult.from_solver(self)
                self.value = new_value

            self.partition = Partition(self.model)
            self.partition.refine_by_width(self.value, self.epsilon / (n + 1))
            self.contracted_value = self.partition.weights.dot(self.value)
            alpha = self.step_size / (
                n * (self.aggregation_iterations + self.bellman_iterations) + 1
            )

            for _ in range(self.aggregation_iterations):
                self.extended_value = self.partition.phi.dot(self.contracted_value)
                samples = np.fromiter(
                    (
                        region[np.random.randint(len(region))]
                        for region in self.partition.states_in_region
                    ),
                    dtype=int,
                    count=self.partition.n_regions,
                )
                q = np.empty((self.partition.n_regions, self.model.action_dim))
                for aa in range(self.model.action_dim):
                    q[:, aa] = self.model.reward_matrix[samples, aa] + self.discount * (
                        self.model.transition_matrix[aa][samples].dot(
                            self.extended_value
                        )
                    )
                self.contracted_value = (
                    1 - alpha
                ) * self.contracted_value + alpha * q.max(axis=1)

            self.value = self.partition.phi.dot(self.contracted_value)
