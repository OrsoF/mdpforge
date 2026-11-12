"""Average-reward relative value iteration using MDPax/JAX."""

import numpy as np
from mdpax.solvers import RelativeValueIteration

from mdpforge.solvers.average._common import (
    check_positive_integer,
    check_precision,
    relative_result,
)
from mdpforge.utils.solver_backends import make_mdpax_problem

METADATA = {
    "display_name": "Average RVI · MDPax",
    "family": "vi",
    "backend": "mdpax",
    "description": "MDPax JAX relative value iteration over padded CSR successors.",
}


class Solver:
    """Run native RVI for problems satisfying its span-convergence assumptions."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        max_iterations=20_000,
        max_event_entries=2_000_000,
        max_batch_size=1024,
    ):
        """Configure native RVI.

        final_precision: finite float > 0, native span tolerance.
        max_iterations: int > 0, outer iteration budget.
        max_event_entries: int > 0, padded transition-event capacity.
        max_batch_size: int > 0, states processed per device batch.
        """
        check_precision(final_precision)
        check_positive_integer(max_iterations, "max_iterations")
        check_positive_integer(max_event_entries, "max_event_entries")
        check_positive_integer(max_batch_size, "max_batch_size")
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.max_event_entries = max_event_entries
        self.max_batch_size = max_batch_size
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the relative bias, policy and native reference-state gain."""
        problem = make_mdpax_problem(self.model, self.max_event_entries)
        solver = RelativeValueIteration(
            problem,
            epsilon=self.final_precision,
            max_batch_size=self.max_batch_size,
            checkpoint_frequency=0,
            verbose=0,
            jax_double_precision=True,
        )
        result = solver.solve(max_iterations=self.max_iterations)
        return relative_result(
            np.asarray(result.values, dtype=float).reshape(-1),
            np.asarray(result.policy, dtype=int).reshape(-1),
            gain=result.info.gain,
        )
