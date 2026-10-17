"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import ValueIteration

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "VI · MDPToolbox",
    "family": "vi",
    "backend": "mdptoolbox",
    "description": (
        "MDPToolbox synchronous value iteration through its public API, "
        "with native initialization and independent residual certification."
    ),
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000_000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; requested max_iter; default 100_000_000;
            the backend may recompute its iteration bound.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.name = "VI MDPToolbox"
        self.max_iterations = max_iterations

        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.vi = ValueIteration(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            epsilon=self.epsilon * (1 - self.discount),
            max_iter=self.max_iterations,
        )
        self.vi.run()

        self.value = np.array(self.vi.V)
        self.policy = np.array(self.vi.policy)

        self.value = certify_value(self.model, self.value, self.discount, self.epsilon)
        self.runtime = time.time() - start_time
