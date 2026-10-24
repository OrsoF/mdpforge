"""Discounted Value Iteration using the optional MDPax/JAX backend."""

import numpy as np
from mdpax.solvers import ValueIteration

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpax import make_mdpax_problem
from mdpforge.utils.solver_options import positive_integer

METADATA = {
    "display_name": "VI · MDPax",
    "family": "vi",
    "backend": "mdpax",
    "description": "MDPax JAX value iteration over padded sparse CSR successors.",
}


class Solver(GenericSolver):
    """Run MDPax ValueIteration on a matrix-defined MDP."""

    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 20_000,
        max_event_entries: int = 2_000_000,
        max_batch_size: int = 1024,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 20_000.
        max_event_entries: int > 0; padded transition-event capacity; default 2_000_000.
        max_batch_size: int > 0; backend batch-size limit; default 1024.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.max_event_entries = positive_integer(
            max_event_entries, "max_event_entries"
        )
        self.max_batch_size = positive_integer(max_batch_size, "max_batch_size")
        self.name = "MDPax VI"

    def _solve(self) -> SolveResult:
        problem = make_mdpax_problem(self.model, self.max_event_entries)
        solver = ValueIteration(
            problem,
            gamma=self.discount,
            epsilon=self.final_precision * (1 - self.discount),
            max_batch_size=self.max_batch_size,
            checkpoint_frequency=0,
            verbose=0,
            jax_double_precision=True,
        )
        result = solver.solve(max_iterations=self.max_iterations)
        values = np.asarray(result.values, dtype=float).reshape(-1)
        policy = np.asarray(result.policy, dtype=int).reshape(-1)
        if values.shape != (self.model.state_dim,):
            raise RuntimeError("MDPax returned an invalid value shape")
        if policy.shape != (self.model.state_dim,) or np.any(
            (policy < 0) | (policy >= self.model.action_dim)
        ):
            raise RuntimeError("MDPax returned an invalid policy")
        return SolveResult(
            np.asarray(values), policy, self._iterations or None, self._stop_reason
        )
