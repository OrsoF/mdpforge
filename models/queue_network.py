# Source: controlled queueing-network benchmark.

import itertools

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        capacity = int(np.ceil(max(8, state_dim) ** (1.0 / 3.0))) - 1
        self.capacity = max(1, capacity)
        self.queues = 3
        self.state_dim = (self.capacity + 1) ** self.queues
        self.action_dim = self.queues

        self.arrival_probability = 0.25
        self.service_probability = 0.6
        self.routing_probability = 0.7

        self.name = "{}_{}_queue_network_{}".format(
            self.state_dim,
            self.action_dim,
            self.capacity,
        )

    def _build_model(self):
        self.states = list(
            itertools.product(range(self.capacity + 1), repeat=self.queues)
        )
        self.state_to_index = {state: index for index, state in enumerate(self.states)}

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for state_index, state in enumerate(self.states):
            holding_cost = sum((i + 1) * queue for i, queue in enumerate(state))
            for action in range(self.action_dim):
                self.reward_matrix[state_index, action] = -holding_cost - 0.1 * action
                distribution: dict[tuple[int, ...], float] = {}

                for arrival in (False, True):
                    p_arrival = (
                        self.arrival_probability
                        if arrival
                        else 1.0 - self.arrival_probability
                    )
                    after_arrival = list(state)
                    if arrival:
                        after_arrival[0] = min(self.capacity, after_arrival[0] + 1)

                    for served in (False, True):
                        p_service = (
                            self.service_probability
                            if served
                            else 1.0 - self.service_probability
                        )
                        after_service = after_arrival.copy()
                        if served and after_arrival[action] > 0:
                            after_service[action] -= 1
                            if action < self.queues - 1:
                                for routed in (False, True):
                                    p_route = (
                                        self.routing_probability
                                        if routed
                                        else 1.0 - self.routing_probability
                                    )
                                    next_state = after_service.copy()
                                    if routed:
                                        next_state[action + 1] = min(
                                            self.capacity,
                                            next_state[action + 1] + 1,
                                        )
                                    self._add_probability(
                                        distribution,
                                        tuple(next_state),
                                        p_arrival * p_service * p_route,
                                    )
                            else:
                                self._add_probability(
                                    distribution,
                                    tuple(after_service),
                                    p_arrival * p_service,
                                )
                        else:
                            self._add_probability(
                                distribution,
                                tuple(after_service),
                                p_arrival * p_service,
                            )

                total = sum(distribution.values())
                for next_state, probability in distribution.items():
                    self.transition_matrix[action][
                        state_index,
                        self.state_to_index[next_state],
                    ] = probability / total

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    @staticmethod
    def _add_probability(
        distribution: dict[tuple[int, ...], float],
        state: tuple[int, ...],
        probability: float,
    ) -> None:
        distribution[state] = distribution.get(state, 0.0) + probability
