"""Average-reward relative value iteration using pymdptoolbox."""

from mdptoolbox.mdp import RelativeValueIteration

from mdpforge.solvers.average._common import (
    check_positive_integer,
    check_precision,
    relative_result,
)

METADATA = {
    "display_name": "Average RVI · pymdptoolbox",
    "family": "vi",
    "backend": "mdptoolbox",
    "description": (
        "pymdptoolbox relative value iteration with native span convergence."
    ),
}


class Solver:
    """Run native RVI for problems satisfying its span-convergence assumptions."""

    def __init__(self, model, final_precision=1e-3, *, max_iterations=100_000):
        """Configure native RVI.

        final_precision: finite float > 0, native span tolerance.
        max_iterations: int > 0, outer iteration budget.
        """
        check_precision(final_precision)
        check_positive_integer(max_iterations, "max_iterations")
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the relative bias, policy and native average reward estimate."""
        solver = RelativeValueIteration(
            [transition.copy() for transition in self.model.transition_matrix],
            self.model.reward_matrix.copy(),
            epsilon=self.final_precision,
            max_iter=self.max_iterations,
        )
        solver.setSilent()
        solver.run()
        return relative_result(solver.V, solver.policy, gain=solver.average_reward)
