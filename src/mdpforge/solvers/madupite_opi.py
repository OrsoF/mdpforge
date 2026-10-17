"""OPI profile for the optional madupite/PETSc backend."""

from mdpforge.core.model import MDPProtocol
from mdpforge.solvers.madupite_ipi import Solver as MadupiteIPISolver

METADATA = {
    "display_name": "OPI · madupite (Richardson)",
    "family": "mpi",
    "backend": "madupite",
    "description": (
        'Optimistic policy iteration with a fixed number of Richardson '
        'policy-evaluation steps.'
    ),
    "update_scheme": "mixed",
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
        max_iter_pi: int = 1000,
        evaluation_sweeps: int = 20,
    ):
        """Options: max_iter_pi and evaluation_sweeps > 0."""
        super().__init__(
            model,
            discount,
            final_precision,
            max_iter_pi=max_iter_pi, max_iter_ksp=evaluation_sweeps,
            ksp_type="richardson",
            pc_type="none",
        )
        self.name = "madupite OPI"
