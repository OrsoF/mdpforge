"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import numpy as np
from mdptoolbox.mdp import PolicyIteration

from mdpforge.core.mdp import MDP
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
        model: MDP,
        discount: float,
        *,
        max_iterations: int = 100_000_000,
    ):
        """Configure this solver.

        max_iterations: int > 0; outer iteration budget; default 100_000_000.
        """
        super().__init__(model, discount, None, max_iterations=max_iterations)
        self.name = "PI MDPToolbox"

    def _solve(self) -> SolveResult:
        self.vi = PolicyIteration(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            max_iter=self.max_iterations,
        )
        self.vi.run()

        return SolveResult(np.array(self.vi.V), np.array(self.vi.policy))
