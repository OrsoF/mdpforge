"""Discounted value-function primal linear program using Pyomo/HiGHS."""

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "LP · Pyomo/HiGHS · primal",
    "family": "lp",
    "backend": "pyomo",
    "description": (
        "Solve the discounted value-function primal linear program with "
        "Bellman inequality constraints using Pyomo and HiGHS."
    ),
}


class Solver(GenericSolver):
    def __init__(self, model: MDP, discount: float, *, threads: int = 1):
        """threads: int > 0; HiGHS thread count; default 1."""
        assert 0 < discount < 1
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        super().__init__(model, discount, None)
        self.threads = threads
        self.name = __name__.rsplit(".", 1)[-1]

    def _solve(self) -> SolveResult:
        states = range(self.model.state_dim)
        problem = pyo.ConcreteModel()
        problem.v = pyo.Var(states, domain=pyo.Reals)
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
                    >= float(self.model.reward_matrix[state, action])
                    + self.discount * future
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
