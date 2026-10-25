from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.exact_value_function import distance_to_optimal


class GenericSolver:
    """Base for solvers that implement _solve() and return a SolveResult.

    Constructor options are available to the algorithm as instance attributes.
    Computed values and policies are returned through SolveResult.
    """

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float | None = 1e-3,
        *,
        max_iterations: int = 100_000,
    ) -> None:
        """max_iterations: int > 0, default 100_000, outer iteration budget."""
        self.name = self.__class__.__name__
        self.model = model
        self.discount = discount
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self._iterations = 0
        self._stop_reason: str | None = None

    def run(self) -> SolveResult:
        """Run the algorithm and return its result."""
        result = self._solve()
        return result

    def _solve(self) -> SolveResult:
        """Return the computed value and available solution details."""
        raise NotImplementedError("Subclasses must implement _solve().")
