"""High-precision discounted value references and comparison helpers."""

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.utils.persistence import get_cached_value_function


def compute_value_function(model: MDPProtocol, discount: float, precision: float):
    """Solve with VI while preserving the model's matrix representation."""
    from mdpforge.solvers.mdpforge_vi import Solver

    solver = Solver(model, discount, precision)
    return solver.run().value


def get_exact_value(
    model: MDPProtocol, discount: float, *, cache="readwrite"
) -> np.ndarray:
    """Get a discounted VI reference with absolute error at most 1e-6."""
    return get_cached_value_function(
        model,
        discount,
        1e-6,
        lambda: compute_value_function(model, discount, 1e-6),
        cache=cache,
    )


def distance_to_optimal(value, model: MDPProtocol, discount: float, norm_method=np.inf):
    """Compare a value approximation against a precise VI reference."""
    if value.ndim == 2:
        value = value.max(axis=1)
    return np.linalg.norm(value - get_exact_value(model, discount), ord=norm_method)
