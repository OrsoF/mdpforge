"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import numpy as np
from mdptoolbox.mdp import PolicyIteration

from mdpforge.core.model import MDPProtocol
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "PI · MDPToolbox",
    "family": "pi",
    "backend": "mdptoolbox",
    "description": (
        "MDPToolbox policy iteration using its default matrix-based policy "
        "evaluation and greedy policy improvement."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        *,
        max_iterations: int = 100_000_000,
    ):
        """Configure this solver.

        max_iterations: int > 0; outer iteration budget; default 100_000_000.
        """
        super().__init__(model, discount, None, max_iterations=max_iterations)
        self.model = model
        self.discount = discount
        self.name = "PI MDPToolbox"
        self.max_iterations = max_iterations

    def _solve(self) -> SolveResult:

        self.vi = PolicyIteration(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            max_iter=self.max_iterations,
        )
        self.vi.run()
        value = np.array(self.vi.V)
        policy = np.array(self.vi.policy)
        return SolveResult(value, policy, 0.0)
