"""PI profile for the optional madupite/PETSc backend."""

from mdpforge.core.model import MDPProtocol
from mdpforge.solvers.madupite_ipi import Solver as MadupiteIPISolver

METADATA = {
    "display_name": "PI · madupite (LU)",
    "family": "pi",
    "backend": "madupite",
    "description": (
        'Policy iteration with direct LU-preconditioned policy evaluation on '
        'one MPI rank.'
    ),
    "update_scheme": "linear_solve",
}


class Solver(MadupiteIPISolver):
    """Run madupite with a fixed PI PETSc solver configuration."""

    solver_type = "pi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iter_pi: int = 1000,
    ):
        """Options: max_iter_pi > 0."""
        super().__init__(
            model,
            discount,
            final_precision,
            max_iter_pi=max_iter_pi, max_iter_ksp=1,
            ksp_type="richardson",
            pc_type="lu",
        )
        self.name = "madupite PI"
