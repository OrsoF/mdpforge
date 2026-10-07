"""
Solver calling the MDPSolver Policy Iteration solver.
"""

import time

import mdpsolver

from core.model import GenericModel
from core.solver import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.name = "PI MDPSolver"

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
            algorithm="pi",
            tolerance=self.epsilon,
            update="standard",
            criterion="discounted",
            initValueVector=[0] * self.model.state_dim,
            initPolicy=[0] * self.model.state_dim,
            verbose=False,
            postProcessing=False,
            makeFinalCheck=False,
        )

        self.runtime = time.time() - start_time
        self.value = None
        self.policy = None
