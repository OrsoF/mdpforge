"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import time

import mdpsolver
import numpy as np

from mdpforge.core.conversion import compute_mdpsolver_args
from mdpforge.core.model import MDP
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        parallel: bool = False,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.discount = discount
        self.parallel = parallel
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.name = "VISOR MDPSolver"

        self.trans, self.rew = compute_mdpsolver_args(self.model)
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
            tolerance=self.epsilon * (1 - self.discount),
            update="sor",
            criterion="discounted",
            parallel=self.parallel,
            verbose=False,
        )

        self.value = np.asarray(self.mdl.getValueVector(), dtype=float)
        self.policy = np.asarray(self.mdl.getPolicy(), dtype=int)

        self.value = certify_value(self.model, self.value, self.discount, self.epsilon)
        self.runtime = time.time() - start_time
