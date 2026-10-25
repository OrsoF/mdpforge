"""Starting-vector conversions and bounded solver iterations."""

import numpy as np


def initial_vectors(model, initial_value=None, initial_policy=None):
    """Copy optional starting vectors."""
    value = (
        None
        if initial_value is None
        else np.array(initial_value, dtype=float, copy=True)
    )
    policy = (
        None
        if initial_policy is None
        else np.array(initial_policy, dtype=int, copy=True)
    )
    return value, policy


def bounded_iterations(max_iterations: int, stage: str):
    """Yield iteration indices and raise if a stage exhausts its budget."""
    for iteration in range(1, max_iterations + 1):
        yield iteration
    raise RuntimeError(f"{stage}: max_iterations ({max_iterations}) exhausted")
