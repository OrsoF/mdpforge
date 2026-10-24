"""Result of one solver run."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SolveResult:
    """Store a solver's value, policy, and available stop details."""

    value: np.ndarray
    policy: np.ndarray | None
    iterations: int | None = None
    stop_reason: str | None = None

    @classmethod
    def from_solver(cls, solver) -> "SolveResult":
        """Capture the outcome of a completed solver run."""
        return cls(
            solver.value,
            solver.policy,
            getattr(solver, "iterations", None) or None,
            getattr(solver, "stop_reason", None),
        )
