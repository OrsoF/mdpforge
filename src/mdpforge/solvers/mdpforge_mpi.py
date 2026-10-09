from time import time

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy


class Solver:
    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-2,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        # Class arguments
        self.model = model
        self.discount = discount

        self.name = "mdpforge MPI"

        self.max_iter_eval = int(1e1)
        self.precision_policy_eval = final_precision
        self.precision_policy_update = final_precision

    def run(self):
        start_time = time()

        # policy = np.zeros((self.model.state_dim))
        policy = np.random.randint(
            0, self.model.action_dim, size=(self.model.state_dim)
        )
        value = np.zeros((self.model.state_dim))

        while True:
            value = self._policy_evaluation(
                policy,
                self.precision_policy_eval,
                self.max_iter_eval,
                value,
            )

            q_value = bellman_operator(self.model, value, self.discount)
            new_policy = q_value.argmax(axis=1)
            tolerance = self.precision_policy_update * (1 - self.discount)
            variation = np.absolute(q_value.max(axis=1) - value).max()

            if variation < tolerance:
                self.value = value
                self.policy = new_policy
                self.runtime = time() - start_time
                break
            else:
                policy = new_policy

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsi_eval: float,
        max_iteration_evaluation: int,
        value: np.ndarray,
    ) -> np.ndarray:
        """Policy evaluation step of the modified policy iteration algorithm."""
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)

        tolerance = (1 - self.discount) * epsi_eval

        while True:
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy.dot(value)
            variation = np.absolute(new_value - value).max()
            value = new_value
            if variation < tolerance or eval_iter == max_iteration_evaluation:
                break

        return value

    def _compute_transition_reward_pi(self, policy):
        policy = policy.astype(int)
        return compute_transition_reward_policy(self.model, policy)
