"""
Solver calling MDPSolver MPI with SOR and explicit initialization.
"""

import mdpsolver
import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.mdpsolver import create_mdpsolver, solve_mdpsolver
from mdpforge.utils.solver_options import (
    initial_vectors,
    positive_integer,
    positive_real,
)

METADATA = {
    "display_name": "MPI · MDPSolver · SOR · V₀=0",
    "family": "mpi",
    "backend": "mdpsolver",
    "description": (
        "MDPSolver modified policy iteration with successive over-relaxation "
        "updates and explicitly zero-initialized values and policy."
    ),
    "update_scheme": "sor",
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float,
        parallel: bool = False,
        *,
        max_evaluation_iterations: int = 10,
        relaxation: float = 1.0,
        initial_value: np.ndarray | None = None,
        initial_policy: np.ndarray | None = None,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target.
        parallel: bool; backend parallel execution where supported; default False.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10.
        relaxation: float in (0, 2); SOR relaxation factor; default 1.0.
        initial_value: None or a finite vector of shape (S,); starting value; default
            None.
        initial_policy: None or an integer vector (S,) in [0, A); starting policy;
            default None.
        """
        super().__init__(model, discount, final_precision)
        self.parallel = parallel
        self.initial_value, self.initial_policy = initial_vectors(
            model, initial_value, initial_policy
        )
        if self.initial_value is None:
            self.initial_value = np.zeros(model.state_dim)
        if self.initial_policy is None:
            self.initial_policy = np.zeros(model.state_dim, dtype=int)
        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )
        self.relaxation = positive_real(relaxation, "relaxation")
        if self.relaxation >= 2:
            raise ValueError("relaxation must belong to (0, 2)")
        self.name = "MPISOR init MDPSolver"
        self.mdl = create_mdpsolver(model, discount, mdpsolver)

    def _solve(self) -> np.ndarray:
        value, self.policy = solve_mdpsolver(
            self.mdl,
            algorithm="mpi",
            tolerance=self.final_precision,
            update="sor",
            parallel=self.parallel,
            initValueVector=(
                [] if self.initial_value is None else self.initial_value.tolist()
            ),
            initPolicy=(
                [] if self.initial_policy is None else self.initial_policy.tolist()
            ),
            parIterLim=self.max_evaluation_iterations,
            SORrelaxation=self.relaxation,
        )

        return value
