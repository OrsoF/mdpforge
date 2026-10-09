from time import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy


class Solver:
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        # Class arguments
        self.model = model
        self.discount = discount

        self.name = "mdpforge PI"

    def run(self):
        start_time = time()

        policy = np.zeros((self.model.state_dim), dtype=int)
        while True:
            value = self._policy_evaluation_matrix(policy)

            new_policy = bellman_operator(self.model, value, self.discount).argmax(
                axis=1
            )
            policy_update_condition = np.all(new_policy == policy)

            if policy_update_condition:
                self.value = value
                self.policy = policy
                self.runtime = time() - start_time
                break
            else:
                policy = new_policy

    def _policy_evaluation_matrix(self, policy: np.ndarray) -> np.ndarray:
        """Evaluate a policy by solving its discounted linear system."""
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        A = eye(self.model.state_dim, format="csr") - self.discount * transition_policy
        return spsolve(A, reward_policy)
