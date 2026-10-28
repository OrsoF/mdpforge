"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

import numpy as np
from mdptoolbox.mdp import PolicyIterationModified

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_names import solver_names

METADATA = {
    "display_name": "MPI · MDPToolbox",
    "family": "mpi",
    "backend": "mdptoolbox",
    "description": (
        "MDPToolbox modified policy iteration with synchronous, truncated "
        "policy evaluation."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_evaluation_iterations: int = 10,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10.
        """
        super().__init__(model, discount, float(final_precision))
        self.max_evaluation_iterations = max_evaluation_iterations

        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        self.mpi = PolicyIterationModified(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            epsilon=self.final_precision,
            max_iter=self.max_evaluation_iterations,
        )
        self.mpi.run()

        return SolveResult(np.array(self.mpi.V), np.array(self.mpi.policy))
