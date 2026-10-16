import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "PI · mdpforge",
    "family": "pi",
    "backend": "mdpforge",
    "description": (
        "Policy iteration with sparse linear-system policy evaluation and "
        "greedy policy improvement until policy stability."
    ),
    "update_scheme": "linear_solve",
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float,
    ):
        super().__init__(model, discount, final_precision)
        self.name = "mdpforge PI"

    def _solve(self) -> np.ndarray:
        policy = np.zeros(self.model.state_dim, dtype=int)
        while True:
            value = self._policy_evaluation_matrix(policy)
            new_policy = bellman_operator(self.model, value, self.discount).argmax(
                axis=1
            )
            policy_stable = np.array_equal(new_policy, policy)

            if policy_stable:
                self.policy = policy
                return value
            policy = new_policy

    def _policy_evaluation_matrix(self, policy: np.ndarray) -> np.ndarray:
        """Evaluate a policy by solving its discounted linear system."""
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        system = (
            eye(self.model.state_dim, format="csr") - self.discount * transition_policy
        )
        return spsolve(system, reward_policy)
