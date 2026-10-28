"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import numpy as np
from mdptoolbox.mdp import ValueIteration

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_names import solver_names

METADATA = {
    "display_name": "VI · MDPToolbox",
    "family": "vi",
    "backend": "mdptoolbox",
    "description": (
        "MDPToolbox synchronous value iteration through its public API, "
        "with native initialization."
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
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        self.vi = ValueIteration(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            epsilon=self.final_precision,
            max_iter=self.max_iterations,
        )
        self.vi.run()

        return SolveResult(np.array(self.vi.V), np.array(self.vi.policy))
