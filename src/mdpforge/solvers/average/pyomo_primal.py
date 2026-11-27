"""Scalar-gain average-reward Bellman linear program using Pyomo/HiGHS."""

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from mdpforge.solvers.average._common import check_positive_integer, relative_result

METADATA = {
    "display_name": "Average LP · Pyomo/HiGHS · primal",
    "family": "lp",
    "backend": "linear",
}


class Solver:
    """Solve the Bellman LP for an MDP with a state-independent optimal gain."""

    def __init__(self, model, *, threads: int = 1):
        """threads: positive integer; HiGHS thread count."""
        check_positive_integer(threads, "threads")
        self.model = model
        self.threads = threads
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the scalar gain and a relative feasible Bellman potential."""
        states = range(self.model.state_dim)
        problem = pyo.ConcreteModel()
        problem.gain = pyo.Var(domain=pyo.Reals)
        problem.h = pyo.Var(states, domain=pyo.Reals)
        problem.h[0].fix(0.0)
        problem.objective = pyo.Objective(expr=problem.gain, sense=pyo.minimize)
        problem.bellman = pyo.ConstraintList()
        # Neu, Jonsson and Gómez (2017), equations (3)–(4).
        for action, transition in enumerate(self.model.transition_matrix):
            for state in states:
                lo, hi = transition.indptr[state : state + 2]
                future = pyo.quicksum(
                    float(transition.data[k]) * problem.h[int(transition.indices[k])]
                    for k in range(lo, hi)
                )
                problem.bellman.add(
                    problem.gain + problem.h[state]
                    >= float(self.model.reward_matrix[state, action]) + future
                )
        solver = Highs()
        solver.highs_options = {"threads": self.threads, "output_flag": False}
        solver.config.load_solution = False
        result = solver.solve(problem)
        if result.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(f"{self.name} failed: {result.termination_condition}")
        solver.load_vars()
        return relative_result(
            [pyo.value(problem.h[state]) for state in states],
            None,
            gain=pyo.value(problem.gain),
        )
