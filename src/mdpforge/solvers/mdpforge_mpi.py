import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.bellman import iterative_policy_evaluation
from mdpforge.utils.solver_options import positive_integer, positive_real

METADATA = {
    "display_name": "MPI · mdpforge",
    "family": "mpi",
    "backend": "mdpforge",
    "description": (
        "Modified policy iteration with synchronous, truncated policy "
        "evaluation and a randomly initialized policy."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-2,
        *,
        max_iterations: int = 100_000,
        max_evaluation_iterations: int = 10,
        evaluation_precision: float | None = None,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-2.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10.
        evaluation_precision: None for final_precision, or a finite float > 0; inner
            target; default None.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.name = "mdpforge MPI"

        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )
        self.evaluation_precision = (
            final_precision
            if evaluation_precision is None
            else positive_real(evaluation_precision, "evaluation_precision")
        )

    def _solve(self) -> SolveResult:
        policy = np.random.randint(0, self.model.action_dim, size=self.model.state_dim)
        value = np.zeros(self.model.state_dim)
        evaluation_tolerance = self.evaluation_precision * (1 - self.discount)
        tolerance = self.final_precision * (1 - self.discount)

        for iteration in range(1, self.max_iterations + 1):
            previous_value = value
            transition_policy, reward_policy = compute_transition_reward_policy(
                self.model, policy
            )
            value = iterative_policy_evaluation(
                transition_policy,
                reward_policy,
                self.discount,
                evaluation_tolerance,
                value,
                max_iterations=self.max_evaluation_iterations,
            )

            q_value = bellman_operator(self.model, value, self.discount)
            new_policy = q_value.argmax(axis=1)
            bellman_residual = np.abs(q_value.max(axis=1) - value).max()

            if self._check_iteration(
                iteration,
                bellman_residual,
                tolerance,
                value,
                q_value,
                stagnated=(
                    np.array_equal(value, previous_value)
                    and np.array_equal(new_policy, policy)
                ),
            ):
                self.policy = new_policy
                return self._result(value)
            policy = new_policy
