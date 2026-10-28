import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_names import solver_names
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "VI · mdpforge · Span",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Synchronous value iteration with span-residual stopping and "
        "constant-shift correction for absolute value precision."
    ),
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
        """Configure span-based value iteration.

        final_precision: finite float > 0; absolute value error target.
        initial_value: None for zeros, or a finite NumPy vector of shape (S,).
        max_iterations: int > 0, default 100_000, maximum Bellman updates.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        value, _ = initial_vectors(initial_value)
        self.initial_value = np.zeros(model.state_dim) if value is None else value
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        value = self.initial_value.copy()
        tolerance = 2 * self.final_precision * (1 - self.discount) / self.discount

        for iteration in range(1, self.max_iterations + 1):
            new_value = optimal_bellman_operator(self.model, value, self.discount)

            residual = new_value - value
            bellman_span = np.ptp(residual)
            corrected_value = new_value
            if bellman_span < tolerance:
                # Centering bounds the returned full residual by discount * span / 2.
                midpoint = np.min(residual) / 2 + np.max(residual) / 2
                corrected_value = new_value + (
                    self.discount * midpoint / (1 - self.discount)
                )

            if not np.isfinite(bellman_span) or not np.isfinite(corrected_value).all():
                raise RuntimeError("VI Span stopped: non_finite values")
            if bellman_span < tolerance:
                return SolveResult(np.asarray(corrected_value), None)
            if np.array_equal(new_value, value):
                raise RuntimeError("VI Span stopped: stagnation")
            value = new_value
        raise RuntimeError("VI Span stopped: max_iterations reached")
