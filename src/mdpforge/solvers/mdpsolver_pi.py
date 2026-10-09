"""
Solver calling the MDPSolver Policy Iteration solver.
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
        self.name = "PI MDPSolver"
        self.mdl = create_mdpsolver(model, discount, mdpsolver)

    def _solve(self) -> np.ndarray:
        value, self.policy = solve_mdpsolver(
            self.mdl,
            algorithm="pi",
            tolerance=self.final_precision,
            update="standard",
            parallel=self.parallel,
            initValueVector=[0.0] * self.model.state_dim,
            initPolicy=[0] * self.model.state_dim,
            verbose=False,
            postProcessing=False,
            makeFinalCheck=False,
        )

        return value
