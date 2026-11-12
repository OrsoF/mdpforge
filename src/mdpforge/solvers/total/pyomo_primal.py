"""Total-reward value-function primal linear program using Pyomo/HiGHS."""

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options

METADATA = {
    "display_name": "Total LP · Pyomo/HiGHS · primal",
    "family": "lp",
    "backend": "pyomo",
    "description": "Total-reward Bellman-inequality linear program using Pyomo/HiGHS.",
}


class Solver:
    """Solve the total-reward value-function linear program."""

    def __init__(self, model, *, threads: int = 1):
        """threads: positive integer; HiGHS thread count."""
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.model = model
        self.threads = threads
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the optimal value vector."""
        terminal = check_options(self.model, 1.0, 1)
        states = range(self.model.state_dim)
        problem = pyo.ConcreteModel()
        problem.v = pyo.Var(states, domain=pyo.Reals)
        for state in terminal:
            problem.v[state].fix(0.0)
        problem.objective = pyo.Objective(
            expr=pyo.quicksum(problem.v[s] for s in states), sense=pyo.minimize
        )
        problem.bellman = pyo.ConstraintList()
        for action, transition in enumerate(self.model.transition_matrix):
            for state in states:
                lo, hi = transition.indptr[state : state + 2]
                future = pyo.quicksum(
                    float(transition.data[k]) * problem.v[int(transition.indices[k])]
                    for k in range(lo, hi)
                )
                problem.bellman.add(
                    problem.v[state]
                    >= float(self.model.reward_matrix[state, action]) + future
                )
        solver = Highs()
        solver.highs_options = {"threads": self.threads, "output_flag": False}
        solver.config.load_solution = False
        result = solver.solve(problem)
        if result.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(f"{self.name} failed: {result.termination_condition}")
        solver.load_vars()
        value = np.array([pyo.value(problem.v[s]) for s in states], dtype=float)
        return SolveResult(value, None, validation_method="residual")
