"""iPI GMRES profile for the optional madupite/PETSc backend."""

from mdpforge.core.mdp import MDP
from mdpforge.solvers.madupite_ipi import Solver as MadupiteIPISolver
from mdpforge.utils.solver_options import solver_identifier

METADATA = {
    "display_name": "iPI · madupite (GMRES)",
    "family": "pi",
    "backend": "madupite",
    "description": (
        "Named GMRES profile of madupite inexact policy iteration for "
        "reproducible comparisons."
    ),
}


class Solver(MadupiteIPISolver):
    """Run madupite with a fixed iPI GMRES PETSc solver configuration."""

    solver_type = "pi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 1000,
        max_evaluation_iterations: int = 1000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 1000.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            1000.
        """
        super().__init__(
            model,
            discount,
            final_precision,
            max_iterations=max_iterations,
            max_evaluation_iterations=max_evaluation_iterations,
            ksp_type="gmres",
            pc_type="none",
        )
        self.name = "madupite iPI GMRES"

        self.solver_id = solver_identifier(
            __name__,
            discount=self.discount,
            final_precision=self.final_precision,
            max_iterations=self.max_iterations,
            max_evaluation_iterations=max_evaluation_iterations,
        )
