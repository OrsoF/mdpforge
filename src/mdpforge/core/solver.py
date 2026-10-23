from time import perf_counter

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.exact_value_function import distance_to_optimal, get_exact_value


class GenericSolver:
    """Base for solvers that implement _solve() and return a SolveResult.

    Constructor options are available to the algorithm as instance attributes.
    Computed values and policies are returned through SolveResult.
    """

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float | None = 1e-3,
        *,
        max_iterations: int = 100_000,
    ) -> None:
        """max_iterations: int > 0, default 100_000, budget for _check_iteration()."""
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        assert final_precision is None or (
            np.isfinite(final_precision) and final_precision > 0
        ), "final_precision must be finite and positive"
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.name = self.__class__.__name__
        self.model = model
        self.discount = discount
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self._iterations = 0
        self._stop_reason: str | None = None

    def run(self) -> SolveResult:
        """Run the algorithm and return its result."""
        start = perf_counter()
        self._iterations = 0
        self._stop_reason = None
        result = self._solve()
        result.value = np.asarray(result.value)
        if result.runtime == 0.0:
            result.runtime = perf_counter() - start
        return result

    def _result(self, value: np.ndarray, policy: np.ndarray | None = None) -> SolveResult:
        """Package a computed value with the current policy and stop details."""
        return SolveResult(
            np.asarray(value), policy, 0.0, self._iterations or None, self._stop_reason
        )

    def _check_iteration(
        self,
        iteration: int,
        residual: float,
        tolerance: float,
        *values: np.ndarray,
        stagnated: bool = False,
    ) -> bool:
        """Certify convergence or raise with the numerical or iteration stop reason."""
        self._iterations = iteration
        if not np.isfinite(residual) or any(
            not np.all(np.isfinite(value)) for value in values
        ):
            self._stop_reason = "non_finite"
        elif residual < tolerance:
            self._stop_reason = "converged"
            return True
        elif stagnated:
            self._stop_reason = "stagnation"
        elif iteration >= self.max_iterations:
            self._stop_reason = "max_iterations"
        else:
            return False

        raise RuntimeError(
            f"{self.name} stopped: {self._stop_reason} after {iteration} iterations "
            f"(Bellman residual={residual:g}, tolerance={tolerance:g})"
        )

    def _solve(self) -> SolveResult:
        """Return the computed value and available solution details."""
        raise NotImplementedError("Subclasses must implement _solve().")

    def distance_to_optimal(self, result: SolveResult):
        return distance_to_optimal(result.value, self.model, self.discount)

    def bellman_residual(self, result: SolveResult, ord=np.inf) -> float:
        bellman_value = optimal_bellman_operator(self.model, result.value, self.discount)
        return np.linalg.norm(result.value - bellman_value, ord=ord)

    def plot_value_function(self, result: SolveResult):
        """Plot the final value function found by the solver."""
        import matplotlib.pyplot as plt

        optimal_value = get_exact_value(self.model, self.discount)
        plt.plot(result.value, label="V")
        plt.plot(optimal_value, label="V*")
        plt.xlabel("State")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
