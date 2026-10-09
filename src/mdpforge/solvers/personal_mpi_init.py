from time import time

import numpy as np

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
        self.epsilon_policy_evaluation = final_precision
        self.epsilon_policy_update = final_precision

        self.name = "MPI"

        self.max_iter_evaluation = int(1e8)
        self.max_iter_policy_update = int(1e8)

        # print("Retirer le tocsr dans Policy Iteration modified.")

    def run(self):
        start_time = time()
        self.policy = np.zeros((self.model.state_dim))
        self.policy = np.argmax(self.model.reward_matrix, axis=1)
        # self.value = np.zeros((self.model.state_dim))
        self.value = self.model.reward_matrix.max(axis=1)

        policy_update_iter = 0
        tolerance = self.epsilon_policy_update * (1 - self.discount)
        while True:
            policy_update_iter += 1
            self.value = self._policy_evaluation(
                self.policy,
                self.epsilon_policy_evaluation,
                self.max_iter_evaluation,
                self.value,
            )
            q_value = bellman_operator(self.model, self.value, self.discount)
            new_policy = q_value.argmax(axis=1)

            variation_condition = (
                np.absolute(q_value.max(axis=1) - self.value).max() < tolerance
            )
            max_iter_condition = policy_update_iter == self.max_iter_policy_update
            self.policy = new_policy

            if variation_condition or max_iter_condition:
                self.runtime = time() - start_time
                break

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsilon_policy_evaluation: float,
        max_iteration_evaluation: int,
        value: np.ndarray,
    ) -> np.ndarray:
        eval_iter = 0
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )
        tolerance = (1 - self.discount) * epsilon_policy_evaluation

        while True:
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy.dot(value)
            variation = np.absolute(new_value - value).max()
            if variation < tolerance or eval_iter == max_iteration_evaluation:
                return new_value
            else:
                value = new_value
