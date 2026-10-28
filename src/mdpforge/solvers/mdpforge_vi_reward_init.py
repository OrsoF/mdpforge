import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_names import solver_names

METADATA = {
    "display_name": "VI · mdpforge · reward init",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Synchronous value iteration initialized from maximum immediate "
        "rewards divided by one minus the discount."
    ),
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.initial_value = model.reward_matrix.max(axis=1) / (1 - discount)
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)

        for iteration in range(1, self.max_iterations + 1):
            new_value = optimal_bellman_operator(self.model, value, self.discount)
            bellman_residual = np.linalg.norm(new_value - value, ord=np.inf)
            if not np.isfinite(bellman_residual) or not np.isfinite(new_value).all():
                raise RuntimeError("VI stopped: non_finite values")
            if bellman_residual < tolerance:
                return SolveResult(np.asarray(new_value), None)
            if np.array_equal(new_value, value):
                raise RuntimeError("VI stopped: stagnation")
            value = new_value
        raise RuntimeError("VI stopped: max_iterations reached")
