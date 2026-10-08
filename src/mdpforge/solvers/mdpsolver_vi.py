"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import time

import mdpsolver
import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.core.solver import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        parallel: bool = False,
    ):
        self.model = model
        self.discount = discount
        self.parallel = parallel
        self.name = "MDPSolver VI"

        self.epsilon = final_precision

        self.trans, self.rew = self.model._compute_mdpsolver_args()
        self.mdl = mdpsolver.model()
        self.mdl.mdp(
            discount=self.discount,
            rewards=self.rew,
            tranMatElementwise=self.trans,
        )

    def run(self):
        start_time = time.time()

        self.mdl.solve(
            algorithm="vi",
            tolerance=self.epsilon,
            update="standard",
            criterion="discounted",
            parallel=self.parallel,
            verbose=False,
        )

        self.runtime = time.time() - start_time
        self.value = np.asarray(self.mdl.getValueVector(), dtype=float)
        self.policy = np.asarray(self.mdl.getPolicy(), dtype=int)
