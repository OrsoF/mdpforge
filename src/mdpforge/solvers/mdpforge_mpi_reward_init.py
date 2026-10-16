import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.bellman import iterative_policy_evaluation

METADATA = {
    "display_name": "MPI · mdpforge · reward init",
    "family": "mpi",
    "backend": "mdpforge",
    "description": (
        "Modified policy iteration with synchronous, truncated evaluation and "
        "value and policy initialization from immediate rewards."
    ),
    "update_scheme": "synchronous",
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float,
    ):
        super().__init__(model, discount, final_precision)
        self.epsilon_policy_evaluation = final_precision
        self.epsilon_policy_update = final_precision

        self.name = "mdpforge MPI (reward init)"

        self.max_iter_evaluation = 100_000_000
        self.max_iter_policy_update = 100_000_000

    def _solve(self) -> np.ndarray:
        policy = np.argmax(self.model.reward_matrix, axis=1)
        value = self.model.reward_matrix.max(axis=1)
        evaluation_tolerance = self.epsilon_policy_evaluation * (1 - self.discount)
        tolerance = self.epsilon_policy_update * (1 - self.discount)

        for _ in range(self.max_iter_policy_update):
            transition_policy, reward_policy = compute_transition_reward_policy(
                self.model, policy
            )
            value = iterative_policy_evaluation(
                transition_policy,
                reward_policy,
                self.discount,
                evaluation_tolerance,
                value,
                max_iterations=self.max_iter_evaluation,
            )
            q_value = bellman_operator(self.model, value, self.discount)
            new_policy = q_value.argmax(axis=1)

            bellman_residual = np.abs(q_value.max(axis=1) - value).max()
            if bellman_residual < tolerance:
                self.policy = new_policy
                return value
            policy = new_policy

        raise RuntimeError(
            "MPI value precision not reached: policy iteration budget exhausted"
        )
