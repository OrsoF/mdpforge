"""Result of one solver run."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SolveResult:
    """Store a solver's value, policy and validation method."""

    value: np.ndarray
    policy: np.ndarray | None
    validation_method: str = "unknown"
    lower_value: np.ndarray | None = None
    upper_value: np.ndarray | None = None

    @classmethod
    def from_solver(
        cls, solver, *, validation_method: str = "unknown"
    ) -> "SolveResult":
        """Capture the outcome of a completed solver run."""
        return cls(
            solver.value,
            solver.policy,
            validation_method=validation_method,
            lower_value=(
                getattr(solver, "lower_value", None)
                if validation_method == "bounds"
                else None
            ),
            upper_value=(
                getattr(solver, "upper_value", None)
                if validation_method == "bounds"
                else None
            ),
        )
