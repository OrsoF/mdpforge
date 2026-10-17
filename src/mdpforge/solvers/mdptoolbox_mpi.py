"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import PolicyIterationModified

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.utils.solver_options import positive_integer, positive_real

METADATA = {
    "display_name": "MPI · MDPToolbox",
    "family": "mpi",
    "backend": "mdptoolbox",
    "description": (
        "MDPToolbox modified policy iteration with synchronous, truncated "
        "policy evaluation and independent residual certification."
    ),
    "update_scheme": "synchronous",
}


class _BoundedPolicyIterationModified(PolicyIterationModified):
    """Bound outer policy updates independently of the inner evaluation cap."""

    def __init__(self, *args, max_iterations, **options):
        self.max_iterations = max_iterations
        super().__init__(*args, **options)

    def _bellmanOperator(self, V=None):
        if self.iter > self.max_iterations:
            raise RuntimeError("MDPToolbox MPI policy iteration budget exhausted")
        return super()._bellmanOperator(V)


class Solver:
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_evaluation_iterations: int = 10,
        max_iterations: int = 100_000_000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10.
        max_iterations: int > 0; outer iteration budget; default 100_000_000.
        """
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.gamma = discount
        self.epsilon = positive_real(final_precision, "final_precision")
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )

        self.name = "MPI MDPToolbox"
        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.mpi = _BoundedPolicyIterationModified(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.gamma,
            epsilon=self.epsilon,
            max_iter=self.max_evaluation_iterations,
            max_iterations=self.max_iterations,
        )
        self.mpi.run()

        self.value = certify_value(
            self.model, np.asarray(self.mpi.V), self.gamma, self.epsilon
        )
        self.policy = np.array(self.mpi.policy)
        self.runtime = time.time() - start_time
