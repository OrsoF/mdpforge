import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "PI · mdpforge",
    "family": "pi",
    "backend": "mdpforge",
    "description": (
        "Policy iteration with sparse linear-system policy evaluation and "
        "greedy policy improvement until policy stability."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        *,
        max_iterations: int = 100_000,
        initial_policy: np.ndarray | None = None,
    ):
        """Configure this solver.

        max_iterations: int > 0; outer iteration budget; default 100_000.
        initial_policy: None or an integer vector (S,) in [0, A); starting policy;
            default None.
        """
        super().__init__(model, discount, None, max_iterations=max_iterations)
        _, policy = initial_vectors(initial_policy=initial_policy)
        self.initial_policy = (
            np.zeros(model.state_dim, dtype=int) if policy is None else policy
        )
        self.name = "mdpforge PI"

    def _solve(self) -> SolveResult:
        policy = self.initial_policy.copy()
        for iteration in range(1, self.max_iterations + 1):
            value = self._policy_evaluation_matrix(policy)
            new_policy = bellman_operator(self.model, value, self.discount).argmax(
                axis=1
            )
            policy_stable = np.array_equal(new_policy, policy)

            if not np.all(np.isfinite(value)):
                raise RuntimeError("PI stopped: non_finite policy values")
            if policy_stable:
                return SolveResult(np.asarray(value), policy)
            policy = new_policy
        raise RuntimeError("PI stopped: max_iterations reached")

    def _policy_evaluation_matrix(self, policy: np.ndarray) -> np.ndarray:
        """Evaluate a policy by solving its discounted linear system."""
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        system = (
            eye(self.model.state_dim, format="csr") - self.discount * transition_policy
        )
        return spsolve(system, reward_policy)
