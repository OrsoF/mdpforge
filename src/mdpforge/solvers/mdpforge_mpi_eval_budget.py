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
        proba: float = 0.01,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        # Class arguments
        self.model = model
        self.discount = discount
        tolerance = final_precision * (1 - discount)
        self.variation_policy_evaluation = tolerance
        self.variation_policy_update = tolerance

        self.name = "mdpforge MPI (evaluation budget)"

        self.max_iter_evaluation = int(1e8)
        self.max_iter_policy_update = int(1e8)
        self.proba = proba

        # print("Retirer le tocsr dans Policy Iteration modified.")

    def run(self):
        start_time = time()
        self.policy = np.zeros((self.model.state_dim))
        self.value = np.zeros((self.model.state_dim))

        variation = np.inf

        while True:
            # Policy Upgrade
            q_value = bellman_operator(self.model, self.value, self.discount)
            self.value = q_value.max(axis=1)
            self.policy = q_value.argmax(axis=1)

            transition_policy, reward_policy = compute_transition_reward_policy(
                self.model, self.policy
            )
            for _ in range(int(1 / self.proba)):
                new_value = reward_policy + self.discount * transition_policy.dot(
                    self.value
                )
                variation = np.absolute(new_value - self.value).max()
                self.value = new_value
                if variation <= self.variation_policy_evaluation:
                    break
            new_value = reward_policy + self.discount * transition_policy.dot(
                self.value
            )
            self.value = new_value

            q_value = bellman_operator(self.model, self.value, self.discount)
            variation = np.absolute(q_value.max(axis=1) - self.value).max()
            variation_condition = variation <= self.variation_policy_update

            if variation_condition:
                self.policy = q_value.argmax(axis=1)
                self.runtime = time() - start_time
                break
