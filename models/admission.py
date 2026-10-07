import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    """
    Admission-control CTMDP, uniformized into a finite discrete-time MDP.

    Source:
        pyAdmissionControlOF.py uploaded by the user.
        Based on the admission-control / critical-level-policy model
        described by Wieczorek, Busic, and Hyon.

    State:
        s = (x, j, k)

        x : number of customers in the system, x in {0, ..., S}
        j : pending customer class, j in {0, ..., J}; j=0 means no pending arrival
        k : hyperexponential service phase, k in {0, ..., K-1}

    Actions:
        0 = reject pending customer
        1 = accept pending customer

    Cost convention in source:
        reject: C[j] if j > 0, plus holding cost
        accept: no rejection cost unless system is full, plus holding cost

    Repository convention:
        reward_matrix stores rewards, so we use reward = -cost.

    Dynamics:
        Continuous-time transition rates are built first, then uniformized
        into stochastic transition matrices.
    """

    REJECT = 0
    ACCEPT = 1

    def __init__(self, state_dim: int, action_dim: int):
        # Default from uploaded Marmote example.
        self.S = max(5, int(round(state_dim / 18.0)) - 1)
        self.J = 5
        self.K = 3

        self.lamda = 1.0

        self.class_prob = np.array(
            [0.2, 0.3, 0.25, 0.15, 0.10],
            dtype=float,
        )

        self.rejection_cost = np.array(
            [5.0, 2.0, 1.0, 0.5, 0.25],
            dtype=float,
        )

        self.alpha = np.array(
            [0.5, 0.2499, 0.2501],
            dtype=float,
        )

        self.mu = np.array(
            [2.0, 0.999, 1.001],
            dtype=float,
        )

        self.holding_cost = 0.2

        # State cardinality: (S+1) * (J+1) * K.
        self.state_dim = (self.S + 1) * (self.J + 1) * self.K
        self.action_dim = 2

        self.name = "{}_{}_admission_control".format(
            self.state_dim,
            self.action_dim,
        )

    def _build_model(self):
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        # First build CTMDP off-diagonal rate matrices Q_a.
        rate_matrices = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        max_rate = 0.0

        for s in range(self.state_dim):
            x, j, k = self._decode(s)

            for action in range(self.action_dim):
                cost = self._cost(x, j, k, action)
                self.reward_matrix[s, action] = -cost

                transitions = self._rate_transitions(x, j, k, action)

                row_rate = 0.0
                for x2, j2, k2, rate in transitions:
                    if rate <= 0.0:
                        continue

                    s2 = self._encode(x2, j2, k2)
                    rate_matrices[action][s, s2] += rate
                    row_rate += rate

                max_rate = max(max_rate, row_rate)

        # Uniformization rate.
        # Slight slack avoids numerical issues in rows attaining max_rate.
        self.uniformization_rate = 1.01 * max_rate

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for action in range(self.action_dim):
            Q = rate_matrices[action]

            for s in range(self.state_dim):
                row_start = Q.rows[s]
                row_data = Q.data[s]

                outgoing_rate = 0.0

                for col, rate in zip(row_start, row_data):
                    probability = rate / self.uniformization_rate
                    self.transition_matrix[action][s, col] += probability
                    outgoing_rate += rate

                self.transition_matrix[action][s, s] += (
                    1.0 - outgoing_rate / self.uniformization_rate
                )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _encode(self, x: int, j: int, k: int):
        assert 0 <= x <= self.S
        assert 0 <= j <= self.J
        assert 0 <= k < self.K
        return (x * (self.J + 1) + j) * self.K + k

    def _decode(self, index: int):
        k = index % self.K
        tmp = index // self.K
        j = tmp % (self.J + 1)
        x = tmp // (self.J + 1)
        return x, j, k

    def _cost(self, x: int, j: int, k: int, action: int):
        if action == self.REJECT:
            cost = 0.0

            if j > 0:
                cost += self.rejection_cost[j - 1]

            # Same convention as uploaded file.
            cost += (self.S - x) * self.holding_cost
            return cost

        if action == self.ACCEPT:
            cost = 0.0

            if j > 0:
                if x == self.S:
                    # Mandatory rejection if full.
                    cost += self.rejection_cost[j - 1]
                    cost += (self.S - x) * self.holding_cost
                else:
                    cost += (self.S - (x + 1)) * self.holding_cost
            else:
                cost += (self.S - x) * self.holding_cost

            return cost

        raise ValueError("Invalid action.")

    def _rate_transitions(self, x: int, j: int, k: int, action: int):
        if action == self.REJECT:
            return self._reject_rates(x, j, k)

        if action == self.ACCEPT:
            return self._accept_rates(x, j, k)

        raise ValueError("Invalid action.")

    def _arrival_rates(self, x_next: int, k_next: int):
        transitions = []

        for jprime in range(1, self.J + 1):
            rate = self.lamda * self.class_prob[jprime - 1]
            transitions.append((x_next, jprime, k_next, rate))

        return transitions

    def _departure_rates(self, x_next: int, reset_phase: bool, current_k: int):
        transitions = []

        if reset_phase:
            for kprime in range(self.K):
                rate = self.mu[current_k] * self.alpha[kprime]
                transitions.append((x_next, 0, kprime, rate))
        else:
            rate = self.mu[current_k]
            transitions.append((x_next, 0, 0, rate))

        return transitions

    def _reject_rates(self, x: int, j: int, k: int):
        transitions = []

        if x == self.S:
            # Arrival is lost or pending rejection keeps occupancy at S.
            transitions.extend(self._arrival_rates(x_next=x, k_next=k))

            # Service completion.
            transitions.extend(
                self._departure_rates(
                    x_next=x - 1,
                    reset_phase=True,
                    current_k=k,
                )
            )

        elif x > 1:
            transitions.extend(self._arrival_rates(x_next=x, k_next=k))

            transitions.extend(
                self._departure_rates(
                    x_next=x - 1,
                    reset_phase=True,
                    current_k=k,
                )
            )

        elif x == 1:
            transitions.extend(self._arrival_rates(x_next=x, k_next=k))

            # Last customer leaves: phase resets to 0.
            transitions.extend(
                self._departure_rates(
                    x_next=0,
                    reset_phase=False,
                    current_k=k,
                )
            )

        else:
            # x == 0: no service completion possible.
            for jprime in range(1, self.J + 1):
                rate = self.lamda * self.class_prob[jprime - 1]
                transitions.append((0, jprime, 0, rate))

        return transitions

    def _accept_rates(self, x: int, j: int, k: int):
        transitions = []

        if x == self.S:
            # Full system: accepting is equivalent to mandatory rejection.
            transitions.extend(self._arrival_rates(x_next=x, k_next=k))

            transitions.extend(
                self._departure_rates(
                    x_next=x - 1,
                    reset_phase=True,
                    current_k=k,
                )
            )

        elif x > 1:
            if j > 0:
                # Accepted pending customer joins.
                transitions.extend(self._arrival_rates(x_next=x + 1, k_next=k))

                transitions.extend(
                    self._departure_rates(
                        x_next=x,
                        reset_phase=True,
                        current_k=k,
                    )
                )
            else:
                # No pending customer to accept.
                transitions.extend(self._arrival_rates(x_next=x, k_next=k))

                transitions.extend(
                    self._departure_rates(
                        x_next=x - 1,
                        reset_phase=True,
                        current_k=k,
                    )
                )

        elif x == 1:
            if j > 0:
                transitions.extend(self._arrival_rates(x_next=x + 1, k_next=k))

                transitions.extend(
                    self._departure_rates(
                        x_next=x,
                        reset_phase=True,
                        current_k=k,
                    )
                )
            else:
                transitions.extend(self._arrival_rates(x_next=x, k_next=k))

                # Last customer leaves.
                transitions.extend(
                    self._departure_rates(
                        x_next=0,
                        reset_phase=False,
                        current_k=k,
                    )
                )

        else:
            # x == 0
            if j > 0:
                # Accepting into an empty system chooses a new service phase.
                for jprime in range(1, self.J + 1):
                    for kprime in range(self.K):
                        rate = (
                            self.lamda
                            * self.class_prob[jprime - 1]
                            * self.alpha[kprime]
                        )
                        transitions.append((1, jprime, kprime, rate))

                # Service completion rate from the newly-started phase mixture.
                # This mirrors the uploaded code.
                for kprime in range(self.K):
                    rate = self.mu[kprime] * self.alpha[kprime]
                    transitions.append((0, 0, 0, rate))
            else:
                # No pending customer and empty system: only arrivals.
                for jprime in range(1, self.J + 1):
                    rate = self.lamda * self.class_prob[jprime - 1]
                    transitions.append((0, jprime, 0, rate))

        return transitions
