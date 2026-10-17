"""OPI profile for the optional madupite/PETSc backend."""

from mdpforge.core.model import MDPProtocol
from mdpforge.solvers.madupite_ipi import Solver as MadupiteIPISolver

METADATA = {
    "display_name": "OPI · madupite (Richardson)",
    "family": "mpi",
    "backend": "madupite",
    "description": (
        "Optimistic policy iteration with a fixed number of Richardson "
        "policy-evaluation steps."
    ),
}


class Solver(MadupiteIPISolver):
    """Run madupite with a fixed OPI PETSc solver configuration."""

    solver_type = "mpi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 1000,
        max_evaluation_iterations: int = 20,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 1000.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            20.
        """
        super().__init__(
            model,
            discount,
            final_precision,
            max_iterations=max_iterations,
            max_evaluation_iterations=max_evaluation_iterations,
            ksp_type="richardson",
            pc_type="none",
        )
        self.name = "madupite OPI"
