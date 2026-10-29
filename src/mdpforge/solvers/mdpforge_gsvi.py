"""In-place Gauss-Seidel value iteration for discounted finite MDPs."""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_names import solver_names
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "Gauss-Seidel VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "In-place Bellman sweeps with certified discounted value precision.",
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        *,
        max_iterations: int = 100_000,
    ):
        """Configure state-ordered Gauss-Seidel value iteration."""
        if not 0 < discount < 1:
            raise ValueError("discount must lie strictly between 0 and 1")
        if not np.isfinite(final_precision) or final_precision <= 0:
            raise ValueError("final_precision must be finite and positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        super().__init__(model, discount, final_precision, max_iterations=max_iterations)
        initial, _ = initial_vectors(initial_value)
        if initial is not None and (
            initial.shape != (model.state_dim,) or not np.isfinite(initial).all()
        ):
            raise ValueError("initial_value must be a finite vector of shape (S,)")
        self.initial_value = (
            np.zeros(model.state_dim, dtype=float) if initial is None else initial
        )
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        transitions = self.model.transition_matrix
        rewards = self.model.reward_matrix
        n, n_actions = rewards.shape
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)

        # Use CSR row storage directly; reading updated values gives GS (not Jacobi).
        data = [p.data for p in transitions]
        indices = [p.indices for p in transitions]
        indptr = [p.indptr for p in transitions]
        for _ in range(self.max_iterations):
            for state in range(n):
                best = -np.inf
                for action in range(n_actions):
                    start, stop = indptr[action][state : state + 2]
                    expected = np.dot(
                        data[action][start:stop], value[indices[action][start:stop]]
                    )
                    q = rewards[state, action] + self.discount * expected
                    if q > best:
                        best = q
                value[state] = best

            if not np.isfinite(value).all():
                raise RuntimeError("GSVI stopped: non_finite values")
            residual = optimal_bellman_operator(self.model, value, self.discount) - value
            if np.max(np.abs(residual)) <= tolerance:
                return SolveResult(value, None)
        raise RuntimeError("GSVI stopped: max_iterations reached")
