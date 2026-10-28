"""VI profile for the optional madupite/PETSc backend."""

from mdpforge.core.mdp import MDP
from mdpforge.solvers.madupite_ipi import Solver as MadupiteIPISolver
from mdpforge.utils.solver_names import solver_names

METADATA = {
    "display_name": "VI · madupite (Richardson)",
    "family": "vi",
    "backend": "madupite",
    "description": (
        "Value iteration using one unpreconditioned Richardson step per "
        "policy-evaluation sweep."
    ),
}


class Solver(MadupiteIPISolver):
    """Run madupite with a fixed VI PETSc solver configuration."""

    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 1000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 1000.
        """
        super().__init__(
            model,
            discount,
            final_precision,
            max_iterations=max_iterations,
            max_evaluation_iterations=1,
            ksp_type="richardson",
            pc_type="none",
        )
        self.name, self.verbose_name = solver_names(Solver, locals())
