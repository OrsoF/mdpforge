"""Result of one solver run."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SolveResult:
    """Store a solver's value and policy."""

    value: np.ndarray
    policy: np.ndarray | None

    @classmethod
    def from_solver(cls, solver) -> "SolveResult":
        """Capture the outcome of a completed solver run."""
        return cls(solver.value, solver.policy)
