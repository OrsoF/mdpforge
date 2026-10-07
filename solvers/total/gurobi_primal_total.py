"""
Solve a Discounted MDP with Linear Programming (primal problem) using Gurobi Linear Solver.
"""

import time

import numpy as np

from core.model import GenericModel
from core.solver import GenericSolver
from utils.data_management import gurobi_model_creation


class Solver(GenericSolver):
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.env = env
        self.name = "LPdual"
        self.outgoing_transitions = self._build_outgoing_transitions()

    def _build_outgoing_transitions(self):
        outgoing_transitions = []
        for matrix in self.env.transition_matrix:
            action_transitions = [[] for _ in range(self.env.state_dim)]
            if hasattr(matrix, "tocoo"):
                coo_matrix = matrix.tocoo()
                for ss1, ss2, value in zip(
                    coo_matrix.row, coo_matrix.col, coo_matrix.data
                ):
                    action_transitions[int(ss1)].append((int(ss2), float(value)))
            else:
                for ss1 in range(self.env.state_dim):
                    for ss2 in range(self.env.state_dim):
                        value = matrix[ss1, ss2]
                        if value:
                            action_transitions[ss1].append((ss2, value))
            outgoing_transitions.append(action_transitions)
        return outgoing_transitions

    def _create_variables(self):
        from gurobipy import GRB

        self.var = {}
        for ss in range(self.env.state_dim):
            self.var[ss] = self.model.addVar(
                vtype=GRB.CONTINUOUS, name="v({})".format(ss), lb=-10000, ub=10000
            )

    def _define_objective(self):
        from gurobipy import GRB

        self.model.setObjective(
            sum(self.var[ss] for ss in range(self.env.state_dim)), GRB.MINIMIZE
        )
        self.model.update()

    def _set_constraints(self):
        for ss1 in range(self.env.state_dim):
            for aa in range(self.env.action_dim):
                neighors_value_sum = sum(
                    probability * self.var[ss2]
                    for ss2, probability in self.outgoing_transitions[aa][ss1]
                )
                self.model.addConstr(
                    self.var[ss1] - self.env.reward_matrix[ss1, aa] + neighors_value_sum
                    <= 0
                )
                self.model.addConstr(
                    self.var[ss1] - self.env.reward_matrix[ss1, aa] + neighors_value_sum
                    >= -1.0
                )

        # self.model.addConstr(0.0 <= self.var[0])
        # self.model.addConstr(self.var[0] <= 5000.0)
        self.model.update()

    def run(self):
        start_time = time.time()
        self.model = gurobi_model_creation()

        # Variables
        self._create_variables()

        # Objective
        self._define_objective()

        # Constraints
        self._set_constraints()

        self.model.optimize()

        self.runtime = time.time() - start_time

        self.value = np.array(self.model.x)
        self.policy = None
