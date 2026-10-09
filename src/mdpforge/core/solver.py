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
        self, model: MDPProtocol, discount: float, final_precision: float = 1e-3
    ) -> None:
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        assert np.isfinite(final_precision) and final_precision > 0, (
            "final_precision must be finite and positive"
        )
        self.name = self.__class__.__name__
        self.model = model
        self.discount = discount
        self.final_precision = final_precision

        self.value: np.ndarray
        self.policy: np.ndarray | None = None
        self.runtime: float = 0.0

    def run(self) -> None:
        """Run the algorithm, store NumPy values and measure elapsed seconds."""
        start = perf_counter()
        self.value = np.asarray(self._solve())
        self.runtime = perf_counter() - start

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
