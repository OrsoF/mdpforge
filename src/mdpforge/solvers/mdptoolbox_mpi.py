"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

import numpy as np
from mdptoolbox.mdp import PolicyIterationModified

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
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


class Solver(GenericSolver):
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
        super().__init__(model, discount, final_precision)
        self.model = model
        self.gamma = discount
        self.epsilon = positive_real(final_precision, "final_precision")
        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )

        self.name = "MPI MDPToolbox"
    def _solve(self) -> SolveResult:

        self.mpi = PolicyIterationModified(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.gamma,
            epsilon=self.epsilon,
            max_iter=self.max_evaluation_iterations,
        )
        self.mpi.run()

        value = certify_value(
            self.model, np.asarray(self.mpi.V), self.gamma, self.epsilon
        )
        policy = np.array(self.mpi.policy)
        return SolveResult(value, policy, 0.0)
