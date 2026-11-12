"""Total-reward occupancy-measure dual linear program using Pyomo/HiGHS."""

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.operators import compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options

METADATA = {
    "display_name": "Total LP · Pyomo/HiGHS · dual",
    "family": "lp",
    "backend": "pyomo",
    "description": "Total-reward occupancy-flow linear program using Pyomo/HiGHS.",
}


class Solver:
    """Solve total-reward occupancy flows over nonterminal states."""

    def __init__(self, model, *, threads: int = 1):
        """threads: positive integer; HiGHS thread count."""
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.model = model
        self.threads = threads
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return a deterministic policy and its total-reward value."""
        terminal = check_options(self.model, 1.0, 1)
        transient = [s for s in range(self.model.state_dim) if s not in terminal]
        actions = range(self.model.action_dim)
        incoming = [matrix.tocsc() for matrix in self.model.transition_matrix]
        problem = pyo.ConcreteModel()
        problem.x = pyo.Var(transient, actions, domain=pyo.NonNegativeReals)
        problem.objective = pyo.Objective(
            expr=pyo.quicksum(
                float(self.model.reward_matrix[s, a]) * problem.x[s, a]
                for s in transient
                for a in actions
            ),
            sense=pyo.maximize,
        )
        problem.flow = pyo.ConstraintList()
        for state in transient:
            outgoing = pyo.quicksum(problem.x[state, a] for a in actions)
            inflow = pyo.quicksum(
                float(matrix.data[k]) * problem.x[int(matrix.indices[k]), action]
                for action, matrix in enumerate(incoming)
                for k in range(matrix.indptr[state], matrix.indptr[state + 1])
                if int(matrix.indices[k]) not in terminal
            )
            problem.flow.add(outgoing - inflow == 1)
        solver = Highs()
        solver.highs_options = {"threads": self.threads, "output_flag": False}
        solver.config.load_solution = False
        result = solver.solve(problem)
        if result.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(f"{self.name} failed: {result.termination_condition}")
        solver.load_vars()
        policy = np.zeros(self.model.state_dim, dtype=int)
        occupancy = np.array(
            [[pyo.value(problem.x[s, a]) for a in actions] for s in transient]
        )
        policy[transient] = occupancy.argmax(axis=1)
        transition, reward = compute_transition_reward_policy(self.model, policy)
        submatrix = transition[transient, :][:, transient]
        value = np.zeros(self.model.state_dim)
        value[transient] = spsolve(
            eye(len(transient), format="csr") - submatrix,
            reward[transient],
        )
        if not np.isfinite(value).all():
            raise RuntimeError("dual policy evaluation is singular")
        return SolveResult(value, policy, validation_method="residual")
