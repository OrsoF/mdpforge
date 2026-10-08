from time import time

import numpy as np
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import spsolve

from mdpforge.core.model import SPARSE, GenericModel


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        # Class arguments
        self.model = model
        self.discount = discount

        self.name = "PIM"
        self.model._convert_model(SPARSE)

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
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)

        A = eye(self.model.state_dim, format="csr") - self.discount * transition_policy
        b = reward_policy
        # Solve Ax = b for value
        value = spsolve(A, b)

        return value

    def _compute_transition_reward_pi(self, policy):
        """Compute the transition and reward matrices for the given policy."""
        # Methode 1
        # transition_policy = lil_matrix((self.model.state_dim, self.model.state_dim))
        # reward_policy = np.zeros(self.model.state_dim)
        # for aa in range(self.model.action_dim):
        #     ind: np.ndarray = (policy == aa).nonzero()[0]
        #     if ind.size > 0:
        #         # transition_policy[ind, :] = model.transition_matrix[aa][
        #         #     ind, :
        #         # ].toarray()
        #         for i, elem in enumerate(self.model.transition_matrix[aa][ind, :]):
        #             if elem.toarray()[0, 0] > 0:
        #                 transition_policy[ind, i] = elem
        # # transition_policy = transition_policy.tocsr()
        # return transition_policy, reward_policy

        # Methode 2
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

    def bellman_no_max(self, value: np.ndarray) -> np.ndarray:
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for aa in range(self.model.action_dim):
            q_value[:, aa] = self.model.reward_matrix[
                :, aa
            ] + self.discount * self.model.transition_matrix[aa].dot(value)

        return q_value
