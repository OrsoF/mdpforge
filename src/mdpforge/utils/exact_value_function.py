import numpy as np

from mdpforge.core.model import MDP
from mdpforge.utils.paths import SAVED_VALUE_FUNCTIONS_PATH
from mdpforge.utils.persistence import get_cached_object


def compute_value_function(model: MDP, discount: float, precision: float):
    """Solve with VI while preserving the model's matrix representation."""
    from mdpforge.solvers.personal_vi import Solver

    solver = Solver(model, discount, precision)
    solver.run()
    return solver.value


def get_exact_value(model: MDP, discount: float) -> np.ndarray:
    """Load or compute a discounted reference value at precision 1e-6."""
    assert 0 < discount < 1, "discount must be strictly between 0 and 1"
    return get_cached_object(
        SAVED_VALUE_FUNCTIONS_PATH,
        # Old span-based references could carry an arbitrary constant offset.
        f"discounted_absolute_{discount}_{model.name}.pkl",
        lambda: compute_value_function(model, discount, 1e-6),
    )


def distance_to_optimal(
    value: np.ndarray,
    model: MDP,
    discount: float,
    norm_method: float = np.inf,
) -> float:
    if value.ndim == 2:
        value = value.max(axis=1)
    return np.linalg.norm(value - get_exact_value(model, discount), ord=norm_method)


def get_optimal_policy(model: MDP, discount: float) -> np.ndarray:
    from mdpforge.core.operators import bellman_operator

    value = get_exact_value(model, discount)
    return bellman_operator(model, value, discount).argmax(axis=1)
