"""Native value iteration with successive over-relaxation."""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.mdpforge_vi import Solver as ValueIterationSolver
from mdpforge.utils.solver_options import positive_real

METADATA = {
    "display_name": "VI · mdpforge · SOR",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Sequential Bellman updates with successive over-relaxation and "
        "full-residual stopping for absolute value precision."
    ),
}


class Solver(ValueIterationSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        *,
        relaxation: float = 1.0,
        max_iterations: int = 100_000,
    ):
        """Configure sequential value iteration with relaxation.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        initial_value: None for zeros, or a finite vector of shape (S,).
        relaxation: finite float in (0, 2); update factor; default 1.0.
        max_iterations: int > 0; sweep budget; default 100_000.
        """
        super().__init__(
            model,
            discount,
            final_precision,
            initial_value,
            max_iterations=max_iterations,
        )
        self.relaxation = positive_real(relaxation, "relaxation")
        if self.relaxation >= 2:
            raise ValueError("relaxation must belong to (0, 2)")
        self.name = "mdpforge VISOR"

    def _solve(self) -> SolveResult:
        """Apply relaxed sequential Bellman sweeps until the full residual is small."""
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)
        matrices = self.model.transition_matrix
        rewards = self.model.reward_matrix

        for iteration in range(1, self.max_iterations + 1):
            previous_value = value.copy()
            for state in range(self.model.state_dim):
                best = -np.inf
                for action, matrix in enumerate(matrices):
                    start, end = matrix.indptr[state : state + 2]
                    continuation = np.dot(
                        matrix.data[start:end], value[matrix.indices[start:end]]
                    )
                    candidate = rewards[state, action] + self.discount * continuation
                    best = max(best, candidate)
                value[state] += self.relaxation * (best - value[state])

            residual = np.linalg.norm(
                optimal_bellman_operator(self.model, value, self.discount) - value,
                ord=np.inf,
            )
            if not np.isfinite(residual) or not np.isfinite(value).all():
                raise RuntimeError("VISOR stopped: non_finite values")
            if residual < tolerance:
                return SolveResult(
                    np.asarray(value),
                    None,
                    iteration,
                    "converged",
                )
            if np.array_equal(value, previous_value):
                raise RuntimeError("VISOR stopped: stagnation")
        raise RuntimeError("VISOR stopped: max_iterations reached")
