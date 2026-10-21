"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import PolicyIterationModified

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.solver_options import positive_integer, positive_real

METADATA = {
    "display_name": "MPI · MDPToolbox",
    "family": "mpi",
    "backend": "mdptoolbox",
    "description": (
        "MDPToolbox modified policy iteration with synchronous, truncated "
        "policy evaluation and independent residual certification."
    ),
}


class Solver:
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_evaluation_iterations: int = 10,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10.
        """
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.gamma = discount
        self.epsilon = positive_real(final_precision, "final_precision")
        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )

        self.name = "MPI MDPToolbox"
        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self) -> SolveResult:
        start_time = time.time()

        self.mpi = PolicyIterationModified(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.gamma,
            epsilon=self.epsilon,
            max_iter=self.max_evaluation_iterations,
        )
        self.mpi.run()

        self.value = certify_value(
            self.model, np.asarray(self.mpi.V), self.gamma, self.epsilon
        )
        self.policy = np.array(self.mpi.policy)
        self.runtime = time.time() - start_time
        return SolveResult.from_solver(self)
