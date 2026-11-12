"""Average-reward modified policy iteration through MDPSolver's public API."""

import mdpsolver

from mdpforge.solvers.average._common import (
    check_positive_integer,
    check_precision,
    relative_result,
)
from mdpforge.utils.solver_backends import create_mdpsolver

METADATA = {
    "display_name": "Average MPI · MDPSolver",
    "family": "mpi",
    "backend": "mdpsolver",
    "description": (
        "MDPSolver average-reward modified policy iteration with standard updates."
    ),
}


class Solver:
    """Use MDPSolver's native partial evaluation for constant-gain problems."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        parallel=False,
        max_evaluation_iterations=10,
    ):
        """Configure native MPI.

        final_precision: finite float > 0, native span tolerance.
        parallel: bool, enable native parallel updates.
        max_evaluation_iterations: int > 0, partial policy-evaluation budget.
        """
        check_precision(final_precision)
        check_positive_integer(max_evaluation_iterations, "max_evaluation_iterations")
        if type(parallel) is not bool:
            raise ValueError("parallel must be a bool")
        self.model = model
        self.final_precision = final_precision
        self.parallel = parallel
        self.max_evaluation_iterations = max_evaluation_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the relative potential and policy; the backend exposes no gain."""
        # The model API requires 0 < discount < 1; average solving ignores it.
        mdp = create_mdpsolver(self.model, 0.99, mdpsolver)
        mdp.solve(
            algorithm="mpi",
            criterion="average",
            update="standard",
            tolerance=self.final_precision,
            parIterLim=self.max_evaluation_iterations,
            parallel=self.parallel,
            verbose=False,
        )
        return relative_result(mdp.getValueVector(), mdp.getPolicy())
