import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import q_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "QVI · mdpforge",
    "family": "qvi",
    "backend": "mdpforge",
    "description": (
        "Synchronous Q-value iteration with a full Q-Bellman-residual stopping "
        "threshold for absolute value precision."
    ),
}


class Solver(GenericSolver):
    solver_type = "qvi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        initial_q_value: np.ndarray | None = None,
        *,
        max_iterations: int = 100_000,
    ):
        """Configure Q-value iteration.

        initial_q_value: None for zeros, or a finite NumPy array of shape (S, A).
        max_iterations: int > 0, default 100_000, maximum Bellman updates.
        """
        assert 0 < discount < 1
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        q_value, _ = initial_vectors(initial_q_value)
        shape = (model.state_dim, model.action_dim)
        if q_value is not None and (
            q_value.shape != shape or not np.isfinite(q_value).all()
        ):
            raise ValueError(f"initial_q_value must be finite with shape {shape}")
        self.initial_q_value = np.zeros(shape) if q_value is None else q_value
        self.name = __name__.rsplit(".", 1)[-1]

    def _solve(self) -> SolveResult:
        q_value = self.initial_q_value.copy()
        tolerance = self.final_precision * (1 - self.discount)

        for iteration in range(1, self.max_iterations + 1):
            new_q_value = q_optimal_bellman_operator(self.model, q_value, self.discount)
            bellman_residual = np.max(np.abs(new_q_value - q_value))
            if not np.isfinite(bellman_residual) or not np.isfinite(new_q_value).all():
                raise RuntimeError("QVI stopped: non_finite values")
            if bellman_residual < tolerance:
                self.q_value = new_q_value
                return SolveResult(
                    new_q_value.max(axis=1),
                    new_q_value.argmax(axis=1),
                    validation_method="residual",
                )
            if np.array_equal(new_q_value, q_value):
                raise RuntimeError("QVI stopped: stagnation")
            q_value = new_q_value
        raise RuntimeError("QVI stopped: max_iterations reached")
