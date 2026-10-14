"""
Solver calling the MDPSolver Policy Iteration solver.
"""

import mdpsolver
import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpsolver import create_mdpsolver, solve_mdpsolver

METADATA = {
    "display_name": "PI · MDPSolver",
    "family": "pi",
    "backend": "mdpsolver",
    "description": (
        "MDPSolver policy iteration with standard value updates, explicit "
        "zero initialization and residual certification after postprocessing."
    ),
    "update_scheme": "synchronous",
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
            postProcessing=True,
            makeFinalCheck=False,
        )

        return certify_value(self.model, value, self.discount, self.final_precision)
