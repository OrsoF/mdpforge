import numpy as np

from mdpforge.core.model import MDP
from mdpforge.core.operators import optimal_bellman_operator


def certify_value(model: MDP, value, discount, epsilon):
    """Center a span-based VI result and certify its absolute value error.

    A uniform shift preserves the greedy policy. Centering the Bellman residual
    removes the arbitrary offset left by solvers using a span stopping criterion.
    It performs no additional VI iterations.
    """
    residual = optimal_bellman_operator(model, value, discount) - value
    value = value + (residual.max() + residual.min()) / (2 * (1 - discount))
    residual = optimal_bellman_operator(model, value, discount) - value
    error_bound = np.abs(residual).max() / (1 - discount)
    if not np.isfinite(error_bound) or error_bound > epsilon:
        raise RuntimeError(
            f"VI precision not reached: error bound {error_bound:g} > {epsilon:g}"
        )
    return value
