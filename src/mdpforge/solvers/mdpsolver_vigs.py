"""
Solver calling MDPSolver VI with Gauss-Seidel updates.
"""

import mdpsolver
import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpsolver import create_mdpsolver, solve_mdpsolver


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        parallel: bool = False,
    ):
        super().__init__(model, discount, final_precision)
        self.parallel = parallel
        self.name = "VIGS MDPSolver"
        self.mdl = create_mdpsolver(model, discount, mdpsolver)

    def _solve(self) -> np.ndarray:
        value, self.policy = solve_mdpsolver(
            self.mdl,
            algorithm="vi",
            tolerance=self.final_precision * (1 - self.discount),
            update="gs",
            parallel=self.parallel,
            verbose=False,
        )

        return certify_value(self.model, value, self.discount, self.final_precision)
