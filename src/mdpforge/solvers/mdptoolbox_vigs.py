"""
Solver calling the MDP Toolbox Gauss-Seidel Value Iteration solver.
"""

import numpy as np
from mdptoolbox.mdp import ValueIterationGS

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_options import solver_identifier

METADATA = {
    "display_name": "VI · MDPToolbox · GS",
    "family": "vi",
    "backend": "mdptoolbox",
    "description": "MDPToolbox value iteration with Gauss-Seidel sweeps.",
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000_000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; requested max_iter; default 100_000_000;
            the backend may recompute its iteration bound.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.name = "VIGS MDPToolbox"

        self.solver_id = solver_identifier(
            __name__,
            discount=self.discount,
            final_precision=self.final_precision,
            max_iterations=self.max_iterations,
        )

    def _solve(self) -> SolveResult:
        self.vi = ValueIterationGS(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            epsilon=self.final_precision,
            max_iter=self.max_iterations,
        )
        self.vi.run()

        return SolveResult(np.array(self.vi.V), np.array(self.vi.policy))
