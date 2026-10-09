from time import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.model import MDP
from mdpforge.core.operators import compute_transition_reward_policy


class Solver:
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        # Class arguments
        self.model = model
        self.discount = discount

        self.name = "PI"

        self.max_iter_eval = int(1e8)
        self.precision_policy_eval: float = 1e-2
        self.final_precision = final_precision

    def run(self):
        start_time = time()

        policy = np.zeros((self.model.state_dim), dtype=int)
        value = np.zeros((self.model.state_dim))

        while True:
            value = self._policy_evaluation_matrix(
                policy,
                value,
            )

            new_policy = self.bellman_no_max(value).argmax(axis=1)
            policy_update_condition = np.all(new_policy == policy)

            if policy_update_condition:
                self.value = value
                self.policy = policy
                self.runtime = time() - start_time
                break
            else:
                policy = new_policy

    def _policy_evaluation_matrix(
        self,
        policy: np.ndarray,
        value: np.ndarray,
    ) -> np.ndarray:
        """Policy evaluation using LP solving."""
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        A = eye(self.model.state_dim, format="csr") - self.discount * transition_policy
        b = reward_policy
        # Solve Ax = b for value
        value = spsolve(A, b)

        return value

    def bellman_no_max(self, value: np.ndarray) -> np.ndarray:
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for aa in range(self.model.action_dim):
            q_value[:, aa] = self.model.reward_matrix[
                :, aa
            ] + self.discount * self.model.transition_matrix[aa].dot(value)

        return q_value
