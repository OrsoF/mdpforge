"""
Implementation of the Policy Iteration algorithm given page 8 of
"Adaptive Aggregation Methods for Infinite Horizon
Dynamic Programming" - Bertsekas, Castanon.
"""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import (
    bellman_operator,
    bellman_policy_operator,
    compute_transition_reward_policy,
)
from mdpforge.core.partition import Partition
from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.bellman import (
    iterative_policy_evaluation,
)
from mdpforge.utils.solver_options import bounded_iterations

METADATA = {
    "display_name": "Bertsekas adaptive PI · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Policy iteration combining synchronous policy evaluation with "
        "adaptive aggregate linear-system corrections."
    ),
}


class Solver:
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-4,
        beta_1: float = 1e-1,
        beta_2: float = 1e-1,
        n_regions: int = 5,
        *,
        max_iterations: int = 100_000,
        max_evaluation_iterations: int = 100_000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-4.
        beta_1: finite float > 0; aggregate progress factor; default 1e-1.
        beta_2: finite float > 0; slow-progress factor; default 1e-1.
        n_regions: int > 0; aggregate regions; default 5.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            100_000.
        """
        self.max_iterations = max_iterations
        self.max_evaluation_iterations = max_evaluation_iterations
        self.model = model
        self.discount = discount
        self.final_precision = final_precision
        self.epsilon_policy_evaluation = final_precision / 10
        self.epsilon_final_policy_evaluation = final_precision * (1 - discount) / 10
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.n_regions = n_regions

        self.name = "BertsekasAdaptivePI"

        self.fixed_partition = Partition(self.model)

    def span_state_space(self, value: np.ndarray) -> float:
        return value.max() - value.min()

    def run(self) -> SolveResult:
        self.policy = np.zeros(self.model.state_dim, dtype=int)
        self.value = np.zeros(self.model.state_dim)

        for _ in bounded_iterations(self.max_iterations, "policy iteration"):
            self.value = self._policy_evaluation(self.policy, self.value)

            new_policy = self._greedy_policy(self.value, self.policy)

            policy_condition = np.all(new_policy == self.policy)

            if policy_condition:
                transition_policy, reward_policy = compute_transition_reward_policy(
                    self.model, self.policy
                )

                self.value = iterative_policy_evaluation(
                    transition_policy,
                    reward_policy,
                    self.discount,
                    self.epsilon_final_policy_evaluation,
                    self.value,
                    max_iterations=self.max_evaluation_iterations,
                )

                break
            else:
                self.policy = new_policy

        return SolveResult.from_solver(self)

    def _greedy_policy(self, value: np.ndarray, policy: np.ndarray) -> np.ndarray:
        q_value = bellman_operator(self.model, value, self.discount)
        greedy = q_value.argmax(axis=1)
        states = np.arange(self.model.state_dim)
        improve = q_value[states, greedy] > q_value[states, policy] + 1e-12
        return np.where(improve, greedy, policy)

    def _policy_evaluation(
        self, policy: np.ndarray, initial_value: np.ndarray
    ) -> np.ndarray:
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )
        target_span = np.inf
        previous_slow_span = np.inf
        value = initial_value.copy()

        for _ in bounded_iterations(
            self.max_evaluation_iterations, "policy evaluation"
        ):
            bellman_value = bellman_policy_operator(
                value, self.discount, transition_policy, reward_policy
            )
            residual = bellman_value - value
            residual_span = self.span_state_space(residual)

            if residual_span < self.epsilon_policy_evaluation:
                return bellman_value + self._residual_midpoint_shift(residual)
            else:
                # Section VII, Step 3: aggregate after enough progress, but
                # only once successive approximation slows down.
                if residual_span <= target_span and residual_span >= previous_slow_span:
                    target_span = self.beta_1 * residual_span
                else:
                    previous_slow_span = self.beta_2 * residual_span
                    value = bellman_value
                    continue

                partition = Partition.from_value_bins(
                    self.model,
                    residual,
                    self.n_regions,
                )
                self.fixed_partition = partition
                Q = partition.weights
                W = partition.phi
                right_y = Q @ residual

                A = np.eye(partition.n_regions)
                prod = Q @ transition_policy @ W
                left_y_before_inv = A - self.discount * prod

                y = self._solve_aggregate_correction(left_y_before_inv, right_y)
                value = bellman_value + self.discount * transition_policy @ W @ y
                previous_slow_span = np.inf

    def _solve_aggregate_correction(self, left, right) -> np.ndarray:
        left = left.toarray() if hasattr(left, "toarray") else np.asarray(left)
        right = np.asarray(right).ravel()
        return np.linalg.solve(left, right)

    def _residual_midpoint_shift(self, residual: np.ndarray) -> float:
        return (
            0.5
            * self.discount
            / (1 - self.discount)
            * (residual.max() + residual.min())
        )
