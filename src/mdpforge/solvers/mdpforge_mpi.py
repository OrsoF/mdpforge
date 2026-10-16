import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.bellman import iterative_policy_evaluation

METADATA = {
    "display_name": "MPI · mdpforge",
    "family": "mpi",
    "backend": "mdpforge",
    "description": (
        "Modified policy iteration with synchronous, truncated policy "
        "evaluation and a randomly initialized policy."
    ),
    "update_scheme": "synchronous",
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-2,
    ):
        super().__init__(model, discount, final_precision)
        self.name = "mdpforge MPI"

        self.max_iter_eval = 10
        self.precision_policy_eval = final_precision
        self.precision_policy_update = final_precision

    def _solve(self) -> np.ndarray:
        policy = np.random.randint(0, self.model.action_dim, size=self.model.state_dim)
        value = np.zeros(self.model.state_dim)
        evaluation_tolerance = self.precision_policy_eval * (1 - self.discount)
        tolerance = self.precision_policy_update * (1 - self.discount)

        while True:
            transition_policy, reward_policy = compute_transition_reward_policy(
                self.model, policy
            )
            value = iterative_policy_evaluation(
                transition_policy,
                reward_policy,
                self.discount,
                evaluation_tolerance,
                value,
                max_iterations=self.max_iter_eval,
            )

            q_value = bellman_operator(self.model, value, self.discount)
            new_policy = q_value.argmax(axis=1)
            bellman_residual = np.abs(q_value.max(axis=1) - value).max()

            if bellman_residual < tolerance:
                self.policy = new_policy
                return value
            policy = new_policy
