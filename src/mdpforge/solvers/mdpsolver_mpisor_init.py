"""
Solver calling MDPSolver MPI with SOR and explicit initialization.
"""

import mdpsolver
import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpsolver import create_mdpsolver, solve_mdpsolver
from mdpforge.utils.solver_options import solver_identifier

METADATA = {
    "display_name": "MPI · MDPSolver · SOR · V₀=0",
    "family": "mpi",
    "backend": "mdpsolver",
    "description": (
        "MDPSolver modified policy iteration with successive over-relaxation "
        "updates and explicitly zero-initialized values and policy."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float,
        parallel: bool = False,
        *,
        max_evaluation_iterations: int = 10,
        relaxation: float = 1.0,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target.
        parallel: bool; backend parallel execution where supported; default False.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10.
        relaxation: float in (0, 2); SOR relaxation factor; default 1.0.
        """
        super().__init__(model, discount, final_precision)
        self.parallel = parallel
        self.initial_value = np.zeros(model.state_dim)
        self.initial_policy = np.zeros(model.state_dim, dtype=int)
        self.max_evaluation_iterations = max_evaluation_iterations
        self.relaxation = float(relaxation)
        self.name = "MPISOR init MDPSolver"
        self.mdl = create_mdpsolver(model, discount, mdpsolver)

        self.solver_id = solver_identifier(
            __name__,
            discount=self.discount,
            final_precision=self.final_precision,
            parallel=self.parallel,
            max_evaluation_iterations=self.max_evaluation_iterations,
            relaxation=self.relaxation,
        )

    def _solve(self) -> SolveResult:
        value, policy = solve_mdpsolver(
            self.mdl,
            algorithm="mpi",
            tolerance=self.final_precision,
            update="sor",
            parallel=self.parallel,
            initial_value=self.initial_value,
            initial_policy=self.initial_policy,
            parIterLim=self.max_evaluation_iterations,
            SORrelaxation=self.relaxation,
        )

        return SolveResult(np.asarray(value), policy)
