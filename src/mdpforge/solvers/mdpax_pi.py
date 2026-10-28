"""Discounted Policy Iteration using the optional MDPax/JAX backend."""

import numpy as np
from mdpax.solvers import PolicyIteration

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_backends import make_mdpax_problem
from mdpforge.utils.solver_names import solver_names

METADATA = {
    "display_name": "PI · MDPax",
    "family": "pi",
    "backend": "mdpax",
    "description": "MDPax JAX iterative policy evaluation and improvement.",
}


class Solver(GenericSolver):
    """Run MDPax iterative PolicyIteration on a matrix-defined MDP."""

    solver_type = "pi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 1_000,
        max_evaluation_iterations: int = 10_000,
        max_event_entries: int = 2_000_000,
        max_batch_size: int = 1024,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 1_000.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10_000.
        max_event_entries: int > 0; padded transition-event capacity; default 2_000_000.
        max_batch_size: int > 0; backend batch-size limit; default 1024.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.max_evaluation_iterations = max_evaluation_iterations
        self.max_event_entries = max_event_entries
        self.max_batch_size = max_batch_size
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        problem = make_mdpax_problem(self.model, self.max_event_entries)
        solver = PolicyIteration(
            problem,
            gamma=self.discount,
            epsilon=self.final_precision * (1 - self.discount),
            convergence_test="max_diff",
            max_eval_iter=self.max_evaluation_iterations,
            max_batch_size=self.max_batch_size,
            checkpoint_frequency=0,
            verbose=0,
            jax_double_precision=True,
        )
        result = solver.solve(max_iterations=self.max_iterations)
        values = np.asarray(result.values, dtype=float).reshape(-1)
        policy = np.asarray(result.policy, dtype=int).reshape(-1)
        return SolveResult(np.asarray(values), policy)
