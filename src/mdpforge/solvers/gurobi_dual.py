"""
Solve a Discounted MDP with Linear Programming (dual problem) using Gurobi Linear Solver.
"""

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.gurobi import gurobi_model_creation
from mdpforge.utils.solver_options import solver_identifier

METADATA = {
    "display_name": "LP · Gurobi · dual",
    "family": "lp",
    "backend": "gurobi",
    "description": (
        "Solve the discounted occupancy-measure dual linear program, then "
        "recover and evaluate a deterministic policy."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        *,
        threads: int = 1,
        memory_limit_gb: float = 16.0,
    ):
        """Configure this solver.

        threads: int > 0; Gurobi thread count; default 1.
        memory_limit_gb: finite float > 0; Gurobi memory limit in GB; default 16.0.
        """
        super().__init__(model, discount, None)
        self.env = model
        self.threads = threads
        self.memory_limit_gb = float(memory_limit_gb)
        self.name = "Gurobi LP Dual"
        self.incoming_transitions = self._build_incoming_transitions()

        self.solver_id = solver_identifier(
            __name__,
            discount=self.discount,
            threads=self.threads,
            memory_limit_gb=self.memory_limit_gb,
        )

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

    def _solve(self) -> SolveResult:
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

        occupancy = np.zeros((self.env.state_dim, self.env.action_dim))
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                occupancy[s, a] = self.var[(s, a)].X

        policy = occupancy.argmax(axis=1)
        transition, reward = compute_transition_reward_policy(self.env, policy)
        value = spsolve(
            eye(self.env.state_dim, format="csr") - self.discount * transition, reward
        )
        return SolveResult(value, policy)
