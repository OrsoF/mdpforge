import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
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
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
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

        self.max_evaluation_iterations = max_evaluation_iterations
        self.evaluation_precision = (
            final_precision
            if evaluation_precision is None
            else float(evaluation_precision)
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

            if (
                not np.isfinite(bellman_residual)
                or not np.isfinite(value).all()
                or not np.isfinite(q_value).all()
            ):
                raise RuntimeError("MPI stopped: non_finite values")
            if bellman_residual < tolerance:
                return SolveResult(
                    np.asarray(value),
                    new_policy,
                    iteration,
                    "converged",
                )
            if np.array_equal(value, previous_value) and np.array_equal(
                new_policy, policy
            ):
                raise RuntimeError("MPI stopped: stagnation")
            policy = new_policy
        raise RuntimeError("MPI stopped: max_iterations reached")
