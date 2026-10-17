"""iPI TFQMR profile for the optional madupite/PETSc backend."""

from mdpforge.core.model import MDPProtocol
from mdpforge.solvers.madupite_ipi import Solver as MadupiteIPISolver

METADATA = {
    "display_name": "iPI · madupite (TFQMR)",
    "family": "pi",
    "backend": "madupite",
    "description": (
        'Inexact policy iteration with PETSc transpose-free QMR policy '
        'evaluation.'
    ),
    "update_scheme": "linear_solve",
}


class Solver(MadupiteIPISolver):
    """Run madupite with a fixed iPI TFQMR PETSc solver configuration."""

    solver_type = "pi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iter_pi: int = 1000,
        max_iter_ksp: int = 1000,
    ):
        """Options: max_iter_pi and max_iter_ksp > 0."""
        super().__init__(
            model,
            discount,
            final_precision,
            max_iter_pi=max_iter_pi, max_iter_ksp=max_iter_ksp,
            ksp_type="tfqmr",
            pc_type="none",
        )
        self.name = "madupite iPI TFQMR"
