"""Stationary occupancy-measure linear program using Pyomo/HiGHS."""

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from mdpforge.solvers.average._common import check_positive_integer, relative_result

METADATA = {
    "display_name": "Average LP · Pyomo/HiGHS · dual",
    "family": "lp",
    "backend": "linear",
}


class Solver:
    """Solve the occupancy LP for an MDP with a state-independent optimal gain."""

    def __init__(self, model, *, threads: int = 1):
        """threads: positive integer; HiGHS thread count."""
        check_positive_integer(threads, "threads")
        self.model = model
        self.threads = threads
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the scalar gain and a relative feasible Bellman potential."""
        states = range(self.model.state_dim)
        actions = range(self.model.action_dim)
        incoming = [matrix.tocsc() for matrix in self.model.transition_matrix]
        problem = pyo.ConcreteModel()
        problem.x = pyo.Var(states, actions, domain=pyo.NonNegativeReals)
        problem.objective = pyo.Objective(
            expr=pyo.quicksum(
                float(self.model.reward_matrix[state, action])
                * problem.x[state, action]
                for state in states
                for action in actions
            ),
            sense=pyo.maximize,
        )
        # Neu, Jonsson and Gómez (2017), equations (1)–(2).
        problem.flow = pyo.ConstraintList()
        for state in states:
            outgoing = pyo.quicksum(problem.x[state, action] for action in actions)
            inflow = pyo.quicksum(
                float(matrix.data[k]) * problem.x[int(matrix.indices[k]), action]
                for action, matrix in enumerate(incoming)
                for k in range(matrix.indptr[state], matrix.indptr[state + 1])
            )
            problem.flow.add(outgoing - inflow == 0)
        problem.mass = pyo.Constraint(expr=pyo.quicksum(problem.x.values()) == 1)
        solver = Highs()
        solver.highs_options = {"threads": self.threads, "output_flag": False}
        solver.config.load_solution = False
        result = solver.solve(problem)
        if result.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(f"{self.name} failed: {result.termination_condition}")
        duals = solver.get_duals()
        return relative_result(
            [duals[constraint] for constraint in problem.flow.values()],
            None,
            gain=duals[problem.mass],
        )
