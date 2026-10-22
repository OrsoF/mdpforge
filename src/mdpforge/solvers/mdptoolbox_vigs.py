"""
Solver calling the MDP Toolbox Gauss-Seidel Value Iteration solver.
"""

import numpy as np
from mdptoolbox.mdp import ValueIterationGS

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "VI · MDPToolbox · GS",
    "family": "vi",
    "backend": "mdptoolbox",
    "description": (
        "MDPToolbox value iteration with Gauss-Seidel sweeps, constant-reward "
        "initialization handling and residual certification."
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
        self.name = "VIGS MDPToolbox"
        self.max_iterations = max_iterations

    def _solve(self) -> SolveResult:

        transitions = np.array(
            [matrix.toarray() for matrix in self.model.transition_matrix]
        )
        initial_value = 0
        if np.ptp(self.model.reward_matrix.max(axis=1)) == 0:
            # Toolbox's constructor divides by a zero initial Bellman span.
            initial_value = [1.0] + [0.0] * (self.model.state_dim - 1)
        self.vi = ValueIterationGS(
            transitions,
            self.model.reward_matrix,
            discount=self.discount,
            epsilon=self.epsilon * (1 - self.discount),
            max_iter=self.max_iterations,
            initial_value=initial_value,
        )
        self.vi.run()

        value = np.array(self.vi.V)
        policy = np.array(self.vi.policy)

        value = certify_value(self.model, value, self.discount, self.epsilon)
        return SolveResult(value, policy, 0.0)
