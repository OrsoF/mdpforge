import itertools

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP


def space_dictionary(list_of_tuples: list):
    ranges = (range(*dimension_tuple) for dimension_tuple in list_of_tuples)
    list_of_ranges = tuple(ranges)
    return {elem: i for i, elem in enumerate(itertools.product(*list_of_ranges))}


def get_params(state_dim: int, max_index=50) -> tuple:
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

    return b1, b2, k1, k2, int(np.prod([b1 + 1, b2 + 1, k1, k2]))


METADATA = {
    "sizes": {
        "small": {
            "requested_state_dim": 1500,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 15000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 50000,
            "parameters": {},
        },
    },
    "category": "service_operations",
    "description": "Service-capacity adjustments for two queues in tandem.",
    "reference": (
        "Ohno & Ichiki (1987). Computing Optimal Policies for Controlled "
        "Tandem Queueing Systems. Operations Research 35(1), 121-126. "
        "https://doi.org/10.1287/opre.35.1.121"
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        # The article has exactly 9 actions:
        # (a1, a2), with a1, a2 in {-1, 0, 1}
        self.requested_state_dim = int(state_dim)
        action_dim = 9

        self.arrival_rate_lambda = 0.6
        self.mu_1, self.mu_2 = 0.2, 0.2

        self.B1, self.B2, self.K1, self.K2, self.state_dim = get_params(state_dim)

        # Paper experiment parameters
        self.CA = 1.0
        self.CD = 1.0
        self.CH = 2.0
        self.CS = 2.0
        self.CR = 10.0

        # Continuous-time discount gamma
        self.discount = 0.9

        self.lambda_tilde = (
            self.arrival_rate_lambda + self.K1 * self.mu_1 + self.K2 * self.mu_2
        )

        self.state_dim = int(np.prod([self.B1 + 1, self.B2 + 1, self.K1, self.K2]))
        self.action_dim = action_dim

        # Use this in the solver, not self.discount
        self.effective_discount = self.lambda_tilde / (
            self.lambda_tilde + self.discount
        )

        self.name = "{}_{}_tandem".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.state_dim = int(np.prod([self.B1 + 1, self.B2 + 1, self.K1, self.K2]))
        self.action_dim = 9

        self.state_space_tuples = [
            (0, self.B1 + 1),
            (0, self.B2 + 1),
            (1, self.K1 + 1),
            (1, self.K2 + 1),
        ]

        self.state_encoding = space_dictionary(self.state_space_tuples)
        self.state_list = list(self.state_encoding.keys())

        self.action_decoding = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        self.action_encoding = {
            action: i for i, action in enumerate(self.action_decoding)
        }

        self.build_p()

        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        self.build_r()

    def n1(self, k):
        return min(max(1, k), self.K1)

    def n2(self, k):
        return min(max(1, k), self.K2)

    def lambda_function(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]

        return (
            self.arrival_rate_lambda
            + self.mu_1 * min(m1, self.n1(k1 + a1))
            + self.mu_2 * min(m2, self.n2(k2 + a2))
        )

    def c1(self, s, a):
        m1, _, k1, _ = self.state_list[s]
        a1, _ = self.action_decoding[a]

        return self.n1(k1 + a1) * self.CS + m1 * self.CH

    def c2(self, s, a):
        _, m2, _, k2 = self.state_list[s]
        _, a2 = self.action_decoding[a]

        return self.n2(k2 + a2) * self.CS + m2 * self.CH

    def h1(self, s, a):
        m1, _, _, _ = self.state_list[s]
        a1, _ = self.action_decoding[a]

        return (
            self.CA * int(a1 == 1)
            + self.CD * int(a1 == -1)
            + self.arrival_rate_lambda
            * self.CR
            * int(m1 == self.B1)
            / (self.lambda_function(s, a) + self.discount)
        )

    def h2(self, s, a):
        m1, m2, k1, _ = self.state_list[s]
        a1, a2 = self.action_decoding[a]

        return (
            self.CA * int(a2 == 1)
            + self.CD * int(a2 == -1)
            + min(m1, self.n1(k1 + a1))
            * self.mu_1
            * self.CR
            * int(m2 == self.B2)
            / (self.lambda_function(s, a) + self.discount)
        )

    def s1p(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]

        return self.state_encoding[
            (
                min(m1 + 1, self.B1),
                m2,
                self.n1(k1 + a1),
                self.n2(k2 + a2),
            )
        ]

    def s2p(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]

        return self.state_encoding[
            (
                max(m1 - 1, 0),
                min(m2 + 1, self.B2),
                self.n1(k1 + a1),
                self.n2(k2 + a2),
            )
        ]

    def s3p(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]

        return self.state_encoding[
            (
                m1,
                max(m2 - 1, 0),
                self.n1(k1 + a1),
                self.n2(k2 + a2),
            )
        ]

    def Reward(self, s, a):
        return (
            (self.c1(s, a) + self.c2(s, a))
            / (self.lambda_function(s, a) + self.discount)
            + self.h1(s, a)
            + self.h2(s, a)
        )

    def Reward_tilde(self, s, a):
        return (
            self.Reward(s, a)
            * (self.lambda_function(s, a) + self.discount)
            / (self.lambda_tilde + self.discount)
        )

    def build_p(self):
        m1, m2, k1, k2 = np.asarray(self.state_list).T
        rows = np.arange(self.state_dim)
        capacities = self.K1 * self.K2
        arrival = (np.minimum(m1 + 1, self.B1) * (self.B2 + 1) + m2) * capacities
        transfer = (
            np.maximum(m1 - 1, 0) * (self.B2 + 1) + np.minimum(m2 + 1, self.B2)
        ) * capacities
        departure = (m1 * (self.B2 + 1) + np.maximum(m2 - 1, 0)) * capacities
        self.transition_matrix = []
        for a1, a2 in self.action_decoding:
            chosen_1 = np.clip(k1 + a1, 1, self.K1)
            chosen_2 = np.clip(k2 + a2, 1, self.K2)
            rate_arrival = np.full(self.state_dim, self.arrival_rate_lambda)
            rate_service_1 = self.mu_1 * np.minimum(m1, chosen_1)
            rate_service_2 = self.mu_2 * np.minimum(m2, chosen_2)
            lambda_sa = rate_arrival + rate_service_1 + rate_service_2
            pseudo_rate = self.lambda_tilde - lambda_sa
            offset = (chosen_1 - 1) * self.K2 + chosen_2 - 1
            matrix = csr_matrix((self.state_dim, self.state_dim))
            # Uniformise after adding events, as in the scalar construction.
            for targets, rates in (
                (arrival + offset, rate_arrival),
                (transfer + offset, rate_service_1),
                (departure + offset, rate_service_2),
                (rows, pseudo_rate),
            ):
                matrix += csr_matrix((rates, (rows, targets)), shape=matrix.shape)
            matrix.data /= self.lambda_tilde
            self.transition_matrix.append(matrix)

    def build_r(self):
        m1, m2, k1, k2 = np.asarray(self.state_list).T
        for a, (a1, a2) in enumerate(self.action_decoding):
            chosen_1 = np.clip(k1 + a1, 1, self.K1)
            chosen_2 = np.clip(k2 + a2, 1, self.K2)
            rate_service_1 = self.mu_1 * np.minimum(m1, chosen_1)
            rate_service_2 = self.mu_2 * np.minimum(m2, chosen_2)
            denominator = (
                self.arrival_rate_lambda + rate_service_1 + rate_service_2 + self.discount
            )
            c1 = chosen_1 * self.CS + m1 * self.CH
            c2 = chosen_2 * self.CS + m2 * self.CH
            h1 = (
                self.CA * int(a1 == 1)
                + self.CD * int(a1 == -1)
                + self.arrival_rate_lambda * self.CR * (m1 == self.B1) / denominator
            )
            h2 = (
                self.CA * int(a2 == 1)
                + self.CD * int(a2 == -1)
                + rate_service_1 * self.CR * (m2 == self.B2) / denominator
            )
            self.reward_matrix[:, a] = -(
                ((c1 + c2) / denominator + h1 + h2)
                * denominator
                / (self.lambda_tilde + self.discount)
            )

    def reformat_regions(self):
        self.regions = [[int(state) for state in region] for region in self.regions]
