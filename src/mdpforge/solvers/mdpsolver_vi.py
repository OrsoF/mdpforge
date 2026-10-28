"""
Solver calling MDPSolver Value Iteration.
"""

import mdpsolver
import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_backends import create_mdpsolver, solve_mdpsolver
from mdpforge.utils.solver_names import solver_names
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "VI · MDPSolver",
    "family": "vi",
    "backend": "mdpsolver",
    "description": "MDPSolver value iteration with standard Bellman sweeps.",
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        parallel: bool = False,
        *,
        initial_value: np.ndarray | None = None,
        initial_policy: np.ndarray | None = None,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        parallel: bool; backend parallel execution where supported; default False.
        initial_value: None or a finite vector of shape (S,); starting value; default
            None.
        initial_policy: None or an integer vector (S,) in [0, A); starting policy;
            default None.
        """
        super().__init__(model, discount, final_precision)
        self.parallel = parallel
        self.initial_value, self.initial_policy = initial_vectors(
            initial_value, initial_policy
        )
        self.name, self.verbose_name = solver_names(Solver, locals())
        self.mdl = create_mdpsolver(model, discount, mdpsolver)

    def _solve(self) -> SolveResult:
        value, policy = solve_mdpsolver(
            self.mdl,
            algorithm="vi",
            tolerance=self.final_precision * (1 - self.discount),
            update="standard",
            parallel=self.parallel,
            initial_value=self.initial_value,
            initial_policy=self.initial_policy,
            verbose=False,
        )

        return SolveResult(np.asarray(value), policy)
