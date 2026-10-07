# Source: factored computer-network administration benchmark.

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        machines = int(np.floor(np.log2(max(2, state_dim))))
        self.machines = max(2, machines)
        self.state_dim = 2**self.machines
        self.action_dim = self.machines

        self.fail_probability = 0.05
        self.reboot_success_probability = 0.95
        self.neighbor_bonus = 0.03

        self.name = "{}_{}_sysadmin_{}".format(
            self.state_dim,
            self.action_dim,
            self.machines,
        )

    def _build_model(self):
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for state_index in range(self.state_dim):
            state = self._decode(state_index)
            up_count = sum(state)

            for action in range(self.action_dim):
                self.reward_matrix[state_index, action] = up_count - 0.15
                event_probabilities = []
                for machine, is_up in enumerate(state):
                    if machine == action:
                        up_probability = self.reboot_success_probability
                    else:
                        left_up = state[(machine - 1) % self.machines]
                        right_up = state[(machine + 1) % self.machines]
                        support = self.neighbor_bonus * (left_up + right_up)
                        if is_up:
                            up_probability = 1.0 - self.fail_probability + support
                        else:
                            up_probability = 0.02 + support
                    target_bit = 1 - is_up
                    change_probability = (
                        1.0 - up_probability if is_up else up_probability
                    )
                    if change_probability > 0.0:
                        event_probabilities.append(
                            (machine, target_bit, change_probability)
                        )

                total_event_probability = sum(
                    probability for _, _, probability in event_probabilities
                )
                if total_event_probability > 0.9:
                    scale = 0.9 / total_event_probability
                    event_probabilities = [
                        (machine, target_bit, probability * scale)
                        for machine, target_bit, probability in event_probabilities
                    ]
                    total_event_probability = 0.9

                stay_probability = max(0.0, 1.0 - total_event_probability)
                self.transition_matrix[action][state_index, state_index] += (
                    stay_probability
                )

                for machine, target_bit, probability in event_probabilities:
                    next_state = list(state)
                    next_state[machine] = target_bit
                    self.transition_matrix[action][
                        state_index,
                        self._encode(tuple(next_state)),
                    ] += probability

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    def _decode(self, state_index: int) -> tuple[int, ...]:
        return tuple((state_index >> bit) & 1 for bit in range(self.machines))

    def _encode(self, state: tuple[int, ...]) -> int:
        value = 0
        for bit, is_up in enumerate(state):
            value += int(is_up) << bit
        return value
