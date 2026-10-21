"""
Solve a Discounted MDP with Linear Programming (dual problem) using Gurobi Linear Solver.
"""

import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.gurobi import gurobi_model_creation
from mdpforge.utils.solver_options import positive_integer, positive_real

METADATA = {
    "display_name": "LP · Gurobi · dual",
    "family": "lp",
    "backend": "gurobi",
    "description": (
        "Solve the discounted occupancy-measure dual linear program, then "
        "recover and evaluate a deterministic policy."
    ),
}


class Solver:
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        *,
        threads: int = 1,
        memory_limit_gb: float = 16.0,
    ):
        """Configure this solver.

        threads: int > 0; Gurobi thread count; default 1.
        memory_limit_gb: finite float > 0; Gurobi memory limit in GB; default 16.0.
        """
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.env = model
        self.threads = positive_integer(threads, "threads")
        self.memory_limit_gb = positive_real(memory_limit_gb, "memory_limit_gb")
        self.discount = discount
        self.name = "Gurobi LP Dual"
        self.incoming_transitions = self._build_incoming_transitions()

    def _build_incoming_transitions(self):
        incoming_transitions = []
        for matrix in self.env.transition_matrix:
            action_transitions = [[] for _ in range(self.env.state_dim)]
            coo_matrix = matrix.tocoo()
            for ss1, ss2, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                action_transitions[int(ss2)].append((int(ss1), float(value)))
            incoming_transitions.append(action_transitions)
        return incoming_transitions

    def _create_variables(self):
        from gurobipy import GRB

        # Dual variables
        self.var = {}
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.var[(s, a)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        self.model.update()

    def _define_objective(self):
        from gurobipy import GRB, LinExpr

        self.obj = LinExpr()
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.obj += self.env.reward_matrix[s, a] * self.var[(s, a)]
        self.model.setObjective(self.obj, GRB.MAXIMIZE)

    def _set_constraints(self):
        for s in range(self.env.state_dim):
            sum_value_over_action = sum(
                self.var[(s, a)] for a in range(self.env.action_dim)
            )
            sum_value_neighbors = sum(
                probability * self.var[(sp, aa)]
                for aa in range(self.env.action_dim)
                for sp, probability in self.incoming_transitions[aa][s]
            )
            self.model.addConstr(
                (sum_value_over_action - self.discount * sum_value_neighbors - 1 == 0),
                "Contrainte",
            )

    def run(self) -> SolveResult:
        start_time = time.time()
        self.model = gurobi_model_creation(
            threads=self.threads, memory_limit_gb=self.memory_limit_gb
        )

        # Primal variables
        self._create_variables()

        # Objective definition
        self._define_objective()

        # Constraints
        self._set_constraints()

        # Solving
        self.model.optimize()

        self.runtime = time.time() - start_time
        self.solving_time = self.model.Runtime

        occupancy = np.zeros((self.env.state_dim, self.env.action_dim))
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                occupancy[s, a] = self.var[(s, a)].X

        self.policy = occupancy.argmax(axis=1)
        transition, reward = compute_transition_reward_policy(self.env, self.policy)
        self.value = spsolve(
            eye(self.env.state_dim, format="csr") - self.discount * transition, reward
        )
        return SolveResult.from_solver(self)
