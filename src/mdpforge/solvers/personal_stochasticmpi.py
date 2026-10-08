from time import time

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.model import SPARSE, GenericModel


class Solver:
    def __init__(
        self,
        model: GenericModel,
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

        self.name = "StochasticMPI"

        self.model._convert_model(SPARSE)
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
            q_value = self.bellman_no_max(self.value)
            self.value = q_value.max(axis=1)
            self.policy = q_value.argmax(axis=1)

            transition_policy, reward_policy = self._compute_transition_reward_pi(
                self.policy
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

            q_value = self.bellman_no_max(self.value)
            variation = np.absolute(q_value.max(axis=1) - self.value).max()
            variation_condition = variation <= self.variation_policy_update

            if variation_condition:
                self.policy = q_value.argmax(axis=1)
                self.runtime = time() - start_time
                break

    def _compute_transition_reward_pi(self, policy):
        transition_policy = lil_matrix((self.model.state_dim, self.model.state_dim))
        reward_policy = np.zeros(self.model.state_dim)
        for aa in range(self.model.action_dim):
            ind = (policy == aa).nonzero()[0]
            if ind.size > 0:
                for ss in ind:
                    transition_policy[[ss], :] = self.model.transition_matrix[aa][
                        [ss], :
                    ].toarray()
                reward_policy[ind] = self.model.reward_matrix[ind, aa]
        transition_policy = transition_policy.tocsr()
        return transition_policy, reward_policy

    def bellman_no_max(self, value):
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for aa in range(self.model.action_dim):
            q_value[:, aa] = self.model.reward_matrix[
                :, aa
            ] + self.discount * self.model.transition_matrix[aa].dot(value)

        return q_value
