"""
Solver calling MDPSolver MPI with Gauss-Seidel updates.
"""

import mdpsolver
import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpsolver import create_mdpsolver, solve_mdpsolver

METADATA = {
    "display_name": "MPI · MDPSolver · GS",
    "family": "mpi",
    "backend": "mdpsolver",
    "description": (
        "MDPSolver modified policy iteration with Gauss-Seidel value updates "
        "and backend-default initialization."
    ),
    "update_scheme": "gauss_seidel",
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float,
        parallel: bool = False,
    ):
        super().__init__(model, discount, final_precision)
        self.parallel = parallel
        self.name = "MPIGS MDPSolver"
        self.mdl = create_mdpsolver(model, discount, mdpsolver)

    def _solve(self) -> np.ndarray:
        value, self.policy = solve_mdpsolver(
            self.mdl,
            algorithm="mpi",
            tolerance=self.final_precision,
            update="gs",
            parallel=self.parallel,
            verbose=False,
            parIterLim=10,
        )

        return value
