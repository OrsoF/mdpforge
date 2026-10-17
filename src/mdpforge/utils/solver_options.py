"""Validation for explicit solver constructor options."""

import numpy as np


def positive_integer(value, name: str) -> int:
    """Require a strictly positive Python integer."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_real(value, name: str) -> float:
    """Require a finite, strictly positive real number."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be finite and positive")
    try:
        valid = np.isfinite(value) and value > 0
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def initial_vectors(model, initial_value=None, initial_policy=None):
    """Copy and validate optional NumPy starting vectors."""
    value = (
        None
        if initial_value is None
        else np.array(initial_value, dtype=float, copy=True)
    )
    policy = None if initial_policy is None else np.asarray(initial_policy)
    if value is not None and (
        value.shape != (model.state_dim,) or not np.all(np.isfinite(value))
    ):
        raise ValueError("initial_value must contain one finite value per state")
    if policy is not None:
        if (
            policy.shape != (model.state_dim,)
            or not np.issubdtype(policy.dtype, np.integer)
            or np.any((policy < 0) | (policy >= model.action_dim))
        ):
            raise ValueError("initial_policy must contain one valid integer per state")
        policy = policy.astype(int, copy=True)
    return value, policy


def bounded_iterations(max_iterations: int, stage: str):
    """Yield iteration indices and raise if a stage exhausts its budget."""
    for iteration in range(1, max_iterations + 1):
        yield iteration
    raise RuntimeError(f"{stage}: max_iterations ({max_iterations}) exhausted")
