"""Result of one solver run."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SolveResult:
    """Store the solution, validation inputs and optional termination details."""

    value: np.ndarray
    policy: np.ndarray | None = None
    validation_method: str = "unknown"
    lower_value: np.ndarray | None = None
    upper_value: np.ndarray | None = None
    gain: float | None = None
    termination_reason: str = "unknown"
    iterations: int | None = None
