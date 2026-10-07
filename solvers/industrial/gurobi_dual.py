"""
Solve a Discounted MDP with Linear Programming (dual problem) using Gurobi Linear Solver.
"""

import time

import numpy as np

from core.model import GenericModel
from utils.bellman import policy_evaluation
from utils.data_management import gurobi_model_creation


class Solver:
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.env = env
        self.discount = discount
        self.name = "Gurobi LP Dual"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.env.transition_matrix, np.ndarray
        )
        self.incoming_transitions = self._build_incoming_transitions()

    def _build_incoming_transitions(self):
        incoming_transitions = []
        for matrix in self.env.transition_matrix:
            action_transitions = [[] for _ in range(self.env.state_dim)]
            if hasattr(matrix, "tocoo"):
                coo_matrix = matrix.tocoo()
                for ss1, ss2, value in zip(
                    coo_matrix.row, coo_matrix.col, coo_matrix.data
                ):
                    action_transitions[int(ss2)].append((int(ss1), float(value)))
            else:
                for ss1 in range(self.env.state_dim):
                    for ss2 in range(self.env.state_dim):
                        value = matrix[ss1, ss2]
                        if value:
                            action_transitions[ss2].append((ss1, value))
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

    def run(self):
        start_time = time.time()
        self.model = gurobi_model_creation()

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

        self.policy = np.zeros((self.env.state_dim, self.env.action_dim))
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.policy[s, a] = self.var[(s, a)].X

        self.value = policy_evaluation(self.env, self.policy, self.discount, 1e-5)
