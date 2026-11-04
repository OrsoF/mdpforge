"""Discounted occupancy-measure dual linear program using Pyomo/HiGHS."""

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "LP · Pyomo/HiGHS · dual",
    "family": "lp",
    "backend": "pyomo",
    "description": (
        "Solve the discounted occupancy-measure dual linear program using "
        "Pyomo and HiGHS, then recover and evaluate a deterministic policy."
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
        actions = range(self.model.action_dim)
        incoming = [matrix.tocsc() for matrix in self.model.transition_matrix]
        problem = pyo.ConcreteModel()
        problem.x = pyo.Var(states, actions, domain=pyo.NonNegativeReals)
        problem.objective = pyo.Objective(
            expr=pyo.quicksum(
                float(self.model.reward_matrix[s, a]) * problem.x[s, a]
                for s in states
                for a in actions
            ),
            sense=pyo.maximize,
        )
        problem.flow = pyo.ConstraintList()
        for state in states:
            outgoing = pyo.quicksum(problem.x[state, a] for a in actions)
            inflow = pyo.quicksum(
                float(transition.data[k])
                * problem.x[int(transition.indices[k]), action]
                for action, transition in enumerate(incoming)
                for k in range(transition.indptr[state], transition.indptr[state + 1])
            )
            problem.flow.add(outgoing - self.discount * inflow == 1)

        solver = Highs()
        solver.highs_options = {"threads": self.threads, "output_flag": False}
        solver.config.load_solution = False
        result = solver.solve(problem)
        if result.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(f"{self.name} failed: {result.termination_condition}")
        solver.load_vars()
        occupancy = np.array(
            [[pyo.value(problem.x[s, a]) for a in actions] for s in states], dtype=float
        )
        policy = occupancy.argmax(axis=1)
        transition, reward = compute_transition_reward_policy(self.model, policy)
        value = spsolve(
            eye(self.model.state_dim, format="csr") - self.discount * transition, reward
        )
        return SolveResult(np.asarray(value), policy, validation_method="residual")
