"""
Solver calling MDPSolver MPI with SOR and explicit initialization.
"""

import mdpsolver
import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpsolver import create_mdpsolver, solve_mdpsolver


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
        self.name = "MPISOR init MDPSolver"
        self.mdl = create_mdpsolver(model, discount, mdpsolver)

    def _solve(self) -> np.ndarray:
        value, self.policy = solve_mdpsolver(
            self.mdl,
            algorithm="mpi",
            tolerance=self.final_precision,
            update="sor",
            parallel=self.parallel,
            initValueVector=[0.0] * self.model.state_dim,
            initPolicy=[0] * self.model.state_dim,
        )

        return value
