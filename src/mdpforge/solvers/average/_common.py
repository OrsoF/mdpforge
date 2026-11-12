"""Option checks and relative potentials for average-reward adapters."""

import numpy as np

from mdpforge.core.solve_result import SolveResult


def check_precision(final_precision):
    """Require a finite positive native convergence tolerance."""
    if not np.isfinite(final_precision) or final_precision <= 0:
        raise ValueError("final_precision must be finite and positive")


def check_positive_integer(value, name):
    """Require a positive integer budget or capacity."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def relative_result(value, policy, *, gain=None):
    """Return the native potential relative to state zero and any native gain."""
    value = np.asarray(value, dtype=float).copy()
    if not np.isfinite(value).all():
        raise RuntimeError("average-reward solver returned nonfinite values")
    if gain is not None:
        gain = float(gain)
        if not np.isfinite(gain):
            raise RuntimeError("average-reward solver returned a nonfinite gain")
    value -= value[0]
    if not np.isfinite(value).all():
        raise RuntimeError("average-reward relative potential is nonfinite")
    return SolveResult(
        value,
        None if policy is None else np.asarray(policy, dtype=int),
        gain=gain,
    )
