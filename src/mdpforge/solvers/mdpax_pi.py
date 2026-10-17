"""Discounted Policy Iteration using the optional MDPax/JAX backend."""

import numpy as np
from mdpax.solvers import PolicyIteration

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpax import make_mdpax_problem

METADATA = {
    "display_name": "PI · MDPax",
    "family": "pi",
    "backend": "mdpax",
    "description": (
        "MDPax JAX iterative policy evaluation and improvement with "
        "independent Bellman-residual certification."
    ),
    "update_scheme": "synchronous",
}


class Solver(GenericSolver):
    """Run MDPax iterative PolicyIteration on a matrix-defined MDP."""

    solver_type = "pi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 1_000,
        max_eval_iter: int = 10_000,
        max_event_entries: int = 2_000_000,
        max_batch_size: int = 1024,
    ):
        """Options: positive outer/evaluation iteration, event and batch limits."""
        super().__init__(model, discount, final_precision)
        if min(max_iterations, max_eval_iter, max_event_entries, max_batch_size) < 1:
            raise ValueError("MDPax limits and batch size must be positive")
        self.max_iterations = int(max_iterations)
        self.max_eval_iter = int(max_eval_iter)
        self.max_event_entries = int(max_event_entries)
        self.max_batch_size = int(max_batch_size)
        self.name = "MDPax PI"

    def _solve(self) -> np.ndarray:
        problem = make_mdpax_problem(self.model, self.max_event_entries)
        solver = PolicyIteration(
            problem,
            gamma=self.discount,
            epsilon=self.final_precision * (1 - self.discount),
            convergence_test="max_diff",
            max_eval_iter=self.max_eval_iter,
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
        values = certify_value(self.model, values, self.discount, self.final_precision)
        self.policy = policy
        return values
