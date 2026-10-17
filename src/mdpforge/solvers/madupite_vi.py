"""VI profile for the optional madupite/PETSc backend."""

from mdpforge.core.model import MDPProtocol
from mdpforge.solvers.madupite_ipi import Solver as MadupiteIPISolver

METADATA = {
    "display_name": "VI · madupite (Richardson)",
    "family": "vi",
    "backend": "madupite",
    "description": (
        'Value iteration using one unpreconditioned Richardson step per '
        'policy-evaluation sweep.'
    ),
    "update_scheme": "synchronous",
}


class Solver(MadupiteIPISolver):
    """Run madupite with a fixed VI PETSc solver configuration."""

    solver_type = "vi"

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
            pc_type="none",
        )
        self.name = "madupite VI"
