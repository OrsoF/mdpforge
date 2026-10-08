import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.utils.exact_value_function import distance_to_optimal, get_exact_value


class GenericSolver:
    def __init__(
        self, model: GenericModel, discount: float, final_precision: float
    ) -> None:
        self.name: str
        self.model = model
        self.discount = discount

        self.value: np.ndarray
        self.policy: np.ndarray
        self.runtime: float

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
