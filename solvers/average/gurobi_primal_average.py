import time

import numpy as np

from core.model import GenericModel
from utils.data_management import gurobi_model_creation


class Solver:
    def __init__(self, env: GenericModel, final_precision: float):
        self.env = env
        self.name = "Gurobi_Average_Primal"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.env.transition_matrix, np.ndarray
        )
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
        self.var["g"] = self.model.addVar(vtype=GRB.CONTINUOUS)
        for s in range(self.env.state_dim):
            self.var[s] = self.model.addVar(vtype=GRB.CONTINUOUS)
        self.model.update()

    def _define_objective(self):
        from gurobipy import GRB, LinExpr

        self.obj = LinExpr()
        self.obj += self.var["g"]
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def _set_constraints(self):
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                sum_1 = self.var["g"] + self.var[(s)]

                sum_2 = sum(
                    probability * self.var[sp]
                    for sp, probability in self.outgoing_transitions[a][s]
                )

                sum_3 = self.env.reward_matrix[s, a]

                self.model.addConstr(sum_1 - sum_2 >= sum_3, "Contrainte%d" % s)

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

        self.value = np.zeros((self.env.state_dim))
        for s in range(self.env.state_dim):
            self.value[s] = self.var[(s)].X
        self.policy = None

        self.cost = self.model.getObjective().getValue()
