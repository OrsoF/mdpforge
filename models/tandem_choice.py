import itertools

import numpy as np
from scipy.sparse import dok_array

from core.model import GenericModel


def space_dictionary(list_of_tuples: list):
    ranges = (range(*dimension_tuple) for dimension_tuple in list_of_tuples)
    list_of_ranges = tuple(ranges)
    return {elem: i for i, elem in enumerate(itertools.product(*list_of_ranges))}


def params_match_state_dim(state_dim: int, max_index=50) -> tuple:
    """Get symmetric tandem queue parameters that best match state_dim."""
    distance = np.inf
    b1, b2, k1, k2 = 0, 0, 0, 0

    for b1_ in range(1, max_index):
        b2_ = b1_
        for k1_ in range(1, max_index):
            k2_ = k1_
            dim = np.prod([b1_ + 1, b2_ + 1, k1_, k2_])
            if abs(dim - state_dim) < distance:
                distance = abs(dim - state_dim)
                b1, b2, k1, k2 = b1_, b2_, k1_, k2_

    matched_dim = int(np.prod([b1 + 1, b2 + 1, k1, k2]))
    return b1, b2, k1, k2, matched_dim


class Model(GenericModel):
    """
    Tandem queue model with absolute VM-choice actions.

    State:
        s = (m1, m2, k1, k2)

    Action:
        a = (a1, a2), where:
            a1 in {1, ..., K1}
            a2 in {1, ..., K2}

    Here a1 and a2 are absolute chosen VM counts, not increments.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.arrival_rate_lambda = 0.6
        self.mu_1, self.mu_2 = 0.2, 0.2

        self.B1, self.B2, self.K1, self.K2, self.state_dim = params_match_state_dim(
            state_dim
        )

        self.CA = 1.0
        self.CD = 1.0
        self.CH = 1.0
        self.CS = 1.0
        self.CR = 1.0

        # Continuous-time discount gamma
        self.discount = 0.99

        self.lambda_tilde = (
            self.arrival_rate_lambda + self.K1 * self.mu_1 + self.K2 * self.mu_2
        )

        # Discrete-time discount after uniformisation.
        # Use this in the solver, not self.discount.
        self.effective_discount = self.lambda_tilde / (
            self.lambda_tilde + self.discount
        )

        self.action_dim = int(self.K1 * self.K2)
        self.name = "{}_{}_tandem_choice".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.state_space_tuples = [
            (0, self.B1 + 1),
            (0, self.B2 + 1),
            (1, self.K1 + 1),
            (1, self.K2 + 1),
        ]

        self.state_encoding = space_dictionary(self.state_space_tuples)
        self.state_list = list(self.state_encoding.keys())

        self.action_space_tuples = [
            (1, self.K1 + 1),
            (1, self.K2 + 1),
        ]

        self.action_encoding = space_dictionary(self.action_space_tuples)
        self.action_list = list(self.action_encoding.keys())

        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim), dtype=float)
            for _ in range(self.action_dim)
        ]

        self.reward_matrix = np.empty((self.state_dim, self.action_dim), dtype=float)

        self.build_transition_reward_matrices()

    def lambda_function(self, s, a):
        m1, m2, _, _ = self.state_list[s]
        a1, a2 = self.action_list[a]

        return (
            self.arrival_rate_lambda + self.mu_1 * min(m1, a1) + self.mu_2 * min(m2, a2)
        )

    def c1(self, s, a):
        m1, _, _, _ = self.state_list[s]
        a1, _ = self.action_list[a]

        return a1 * self.CS + m1 * self.CH

    def c2(self, s, a):
        _, m2, _, _ = self.state_list[s]
        _, a2 = self.action_list[a]

        return a2 * self.CS + m2 * self.CH

    def h1(self, s, a):
        m1, _, k1, _ = self.state_list[s]
        a1, _ = self.action_list[a]

        lambda_sa = self.lambda_function(s, a)

        activation_cost = self.CA * max(0, a1 - k1)
        deactivation_cost = self.CD * max(0, k1 - a1)

        rejection_cost = (
            self.arrival_rate_lambda
            * self.CR
            * int(m1 == self.B1)
            / (lambda_sa + self.discount)
        )

        return activation_cost + deactivation_cost + rejection_cost

    def h2(self, s, a):
        m1, m2, _, k2 = self.state_list[s]
        a1, a2 = self.action_list[a]

        lambda_sa = self.lambda_function(s, a)

        activation_cost = self.CA * max(0, a2 - k2)
        deactivation_cost = self.CD * max(0, k2 - a2)

        rejection_cost = (
            self.mu_1
            * min(m1, a1)
            * self.CR
            * int(m2 == self.B2)
            / (lambda_sa + self.discount)
        )

        return activation_cost + deactivation_cost + rejection_cost

    def s1p(self, s, a):
        m1, m2, _, _ = self.state_list[s]
        a1, a2 = self.action_list[a]

        return self.state_encoding[
            (
                min(m1 + 1, self.B1),
                m2,
                a1,
                a2,
            )
        ]

    def s2p(self, s, a):
        m1, m2, _, _ = self.state_list[s]
        a1, a2 = self.action_list[a]

        return self.state_encoding[
            (
                max(m1 - 1, 0),
                min(m2 + 1, self.B2),
                a1,
                a2,
            )
        ]

    def s3p(self, s, a):
        m1, m2, _, _ = self.state_list[s]
        a1, a2 = self.action_list[a]

        return self.state_encoding[
            (
                m1,
                max(m2 - 1, 0),
                a1,
                a2,
            )
        ]

    def Reward(self, s, a):
        lambda_sa = self.lambda_function(s, a)

        return (
            (self.c1(s, a) + self.c2(s, a)) / (lambda_sa + self.discount)
            + self.h1(s, a)
            + self.h2(s, a)
        )

    def Reward_tilde(self, s, a):
        lambda_sa = self.lambda_function(s, a)

        return (
            self.Reward(s, a)
            * (lambda_sa + self.discount)
            / (self.lambda_tilde + self.discount)
        )

    def build_transition_reward_matrices(self):
        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                m1, m2, _, _ = self.state_list[ss1]
                a1, a2 = self.action_list[aa]

                rate_1 = self.arrival_rate_lambda
                rate_2 = self.mu_1 * min(m1, a1)
                rate_3 = self.mu_2 * min(m2, a2)

                lambda_sa = rate_1 + rate_2 + rate_3
                pseudo_rate = self.lambda_tilde - lambda_sa

                if pseudo_rate < -1e-12:
                    raise RuntimeError(
                        f"Negative pseudo-rate: {pseudo_rate}. "
                        "Check lambda_tilde >= lambda_sa."
                    )

                pseudo_rate = max(0.0, pseudo_rate)

                ss2_1 = self.s1p(ss1, aa)
                ss2_2 = self.s2p(ss1, aa)
                ss2_3 = self.s3p(ss1, aa)

                self.transition_matrix[aa][ss1, ss2_1] += rate_1 / self.lambda_tilde
                self.transition_matrix[aa][ss1, ss2_2] += rate_2 / self.lambda_tilde
                self.transition_matrix[aa][ss1, ss2_3] += rate_3 / self.lambda_tilde
                self.transition_matrix[aa][ss1, ss1] += pseudo_rate / self.lambda_tilde

                # Negative cost if your solver maximizes rewards.
                self.reward_matrix[ss1, aa] = -self.Reward_tilde(ss1, aa)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

        for aa, matrix in enumerate(self.transition_matrix):
            row_sums = np.asarray(matrix.sum(axis=1)).ravel()
            if not np.allclose(row_sums, 1.0):
                max_error = np.max(np.abs(row_sums - 1.0))
                raise RuntimeError(
                    f"Transition matrix for action {aa} is not stochastic. "
                    f"Max row-sum error: {max_error}"
                )

    def reformat_regions(self):
        self.regions = [[int(state) for state in region] for region in self.regions]
