from time import perf_counter

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.utils.exact_value_function import distance_to_optimal, get_exact_value


class GenericSolver:
    """Base for solvers that implement _solve() and return a value vector.

    Constructor options are available to the algorithm as instance attributes.
    The algorithm may also set self.policy; otherwise it remains None.
    """

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000,
    ) -> None:
        """max_iterations: int > 0, default 100_000, budget for _check_iteration()."""
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        assert np.isfinite(final_precision) and final_precision > 0, (
            "final_precision must be finite and positive"
        )
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.name = self.__class__.__name__
        self.model = model
        self.discount = discount
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.iterations = 0
        self.stop_reason: str | None = None

        self.value: np.ndarray
        self.policy: np.ndarray | None = None
        self.runtime: float = 0.0

    def run(self) -> None:
        """Run the algorithm, store NumPy values and measure elapsed seconds."""
        start = perf_counter()
        self.iterations = 0
        self.stop_reason = None
        try:
            self.value = np.asarray(self._solve())
        finally:
            self.runtime = perf_counter() - start

    def _check_iteration(
        self,
        iteration: int,
        residual: float,
        tolerance: float,
        *values: np.ndarray,
        stagnated: bool = False,
    ) -> bool:
        """Certify convergence or raise with the numerical or iteration stop reason."""
        self.iterations = iteration
        if not np.isfinite(residual) or any(
            not np.all(np.isfinite(value)) for value in values
        ):
            self.stop_reason = "non_finite"
        elif residual < tolerance:
            self.stop_reason = "converged"
            return True
        elif stagnated:
            self.stop_reason = "stagnation"
        elif iteration >= self.max_iterations:
            self.stop_reason = "max_iterations"
        else:
            return False

        raise RuntimeError(
            f"{self.name} stopped: {self.stop_reason} after {iteration} iterations "
            f"(Bellman residual={residual:g}, tolerance={tolerance:g})"
        )

    def _solve(self) -> np.ndarray:
        """Return one value per state; optionally set self.policy."""
        raise NotImplementedError("Subclasses must implement _solve().")

    def distance_to_optimal(self):
        return distance_to_optimal(self.value, self.model, self.discount)

    def bellman_residual(self, ord=np.inf) -> float:
        bellman_value = optimal_bellman_operator(self.model, self.value, self.discount)
        return np.linalg.norm(self.value - bellman_value, ord=ord)

    def plot_value_function(self):
        """Plot the final value function found by the solver."""
        import matplotlib.pyplot as plt

        optimal_value = get_exact_value(self.model, self.discount)
        plt.plot(self.value, label="V")
        plt.plot(optimal_value, label="V*")
        plt.xlabel("State")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
