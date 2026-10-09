"""
Solve a Discounted MDP with Linear Programming (primal problem) using Gurobi Linear Solver.
"""

import time

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.utils.gurobi import gurobi_model_creation


class Solver:
    def __init__(
        self,
        env: MDPProtocol,
        discount: float,
        final_precision: float,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.env = env
        self.discount = discount
        self.name = "Gurobi LP Primal"
        self.outgoing_transitions = self._build_outgoing_transitions()

    def _build_outgoing_transitions(self):
        outgoing_transitions = []
        for matrix in self.env.transition_matrix:
            action_transitions = [[] for _ in range(self.env.state_dim)]
            coo_matrix = matrix.tocoo()
            for ss1, ss2, value in zip(
                coo_matrix.row, coo_matrix.col, coo_matrix.data
            ):
                action_transitions[int(ss1)].append((int(ss2), float(value)))
            outgoing_transitions.append(action_transitions)
        return outgoing_transitions

    def _create_variables(self):
        from gurobipy import GRB

        self.var = {}
        for ss in range(self.env.state_dim):
            self.var[ss] = self.model.addVar(
                vtype=GRB.CONTINUOUS, name="v({})".format(ss), lb=-np.inf, ub=np.inf
            )

    def _define_objective(self):
        from gurobipy import GRB

        self.model.setObjective(
            sum(self.var[s] for s in range(self.env.state_dim)), GRB.MINIMIZE
        )
        self.model.update()

    def _set_constraints(self):
        for ss in range(self.env.state_dim):
            for aa in range(self.env.action_dim):
                neighors_value_sum = sum(
                    probability * self.var[ss2]
                    for ss2, probability in self.outgoing_transitions[aa][ss]
                )
                self.model.addConstr(
                    self.var[ss]
                    >= self.env.reward_matrix[ss, aa]
                    + self.discount * neighors_value_sum
                )
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
        self.solving_time = self.model.Runtime
        self.value = np.array(self.model.x)
        self.policy = None
