import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.utils.paths import SAVED_VALUE_FUNCTIONS_PATH
from mdpforge.utils.persistence import get_cached_object


def get_exact_value(model: GenericModel, discount: float) -> np.ndarray:
    """Load or compute a discounted reference value at precision 1e-6."""
    assert 0 < discount < 1, "discount must be strictly between 0 and 1"

    def compute_value():
        from mdpforge.solvers.personal_vi import Solver

        solver = Solver(model, discount, 1e-6, mode=model.get_model_type())
        solver.run()
        return solver.value

    return get_cached_object(
        SAVED_VALUE_FUNCTIONS_PATH,
        # Old span-based references could carry an arbitrary constant offset.
        f"discounted_absolute_{discount}_{model.name}.pkl",
        compute_value,
    )


def distance_to_optimal(
    value: np.ndarray,
    model: GenericModel,
    discount: float,
    norm_method: float = np.inf,
) -> float:
    if value.ndim == 2:
        value = value.max(axis=1)
    return np.linalg.norm(value - get_exact_value(model, discount), ord=norm_method)


def get_optimal_policy(model: GenericModel, discount: float) -> np.ndarray:
    from mdpforge.core.operators import bellman_operator

    value = get_exact_value(model, discount)
    return bellman_operator(model, value, discount).argmax(axis=1)
