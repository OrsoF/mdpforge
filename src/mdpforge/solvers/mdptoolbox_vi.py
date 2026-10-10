"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import ValueIteration

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver
from mdpforge.core.validation import validate_model


class _SparseValueIteration(ValueIteration):
    """Initialize pymdptoolbox 4.0b3 without its dense input checks/iteration cap.

    The inherited Bellman operator, run loop and span threshold are unchanged.
    Its automatic cap was already discarded by this adapter; computing it is
    quadratic in the state count and fails for zero reward spans or mixing rows.
    Repository validation checks CSR data directly and tolerates roundoff in
    stochastic rows, without changing the supplied probabilities.
    """

    def __init__(self, model, discount, epsilon, max_iter):
        validate_model(model)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.S, self.A = model.state_dim, model.action_dim
        self.P = self._computeTransition(model.transition_matrix)
        self.R = self._computeReward(model.reward_matrix, self.P)
        self.verbose = False
        self.time = None
        self.iter = 0
        self.V = np.zeros(self.S)
        self.policy = None
        self.thresh = self.epsilon * (1 - self.discount) / self.discount


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.name = "VI MDPToolbox"
        self.max_iter = int(1e8)

        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.vi = _SparseValueIteration(
            self.model,
            discount=self.discount,
            epsilon=self.epsilon * (1 - self.discount),
            max_iter=self.max_iter,
        )
        self.vi.run()

        self.value = np.array(self.vi.V)
        self.policy = np.array(self.vi.policy)

        self.value = certify_value(self.model, self.value, self.discount, self.epsilon)
        self.runtime = time.time() - start_time
