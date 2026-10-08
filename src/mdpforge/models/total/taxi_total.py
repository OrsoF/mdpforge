# Source: Dietterich, T. G. (2000). Hierarchical reinforcement learning with the MAXQ value function decomposition. JAIR 13:227-303.

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.model import GenericModel

WEST, EAST, NORTH, SOUTH, PICKUP, DROPOFF = range(6)
IN_TAXI = 4


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.grid_size = max(int(np.sqrt(state_dim // 20)), 1)
        self.state_dim = self.grid_size**2 * 20
        self.action_dim = 6

        self.name = "{}_{}_taxi_total".format(self.state_dim, self.action_dim)

    def build_state_space(self):
        self.state_list = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for passenger_position in range(5):
                    for destination in range(4):
                        self.state_list.append((x, y, passenger_position, destination))
        self.state_space = {state: ss for (ss, state) in enumerate(self.state_list)}

    def landmark_positions(self):
        return [
            (0, 0),
            (self.grid_size - 1, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, self.grid_size - 1),
        ]

    def move(self, x, y, action):
        if action == WEST:
            return x, max(0, y - 1)
        if action == EAST:
            return x, min(self.grid_size - 1, y + 1)
        if action == NORTH:
            return max(0, x - 1), y
        if action == SOUTH:
            return min(self.grid_size - 1, x + 1), y
        return x, y

    def _build_model(self):
        self.build_state_space()

        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        flags_on_map = self.landmark_positions()

        for ss1 in range(self.state_dim):
            for aa in range(self.action_dim):
                x, y, passenger_position, destination = self.state_list[ss1]

                if passenger_position == destination:
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                    continue

                if aa == PICKUP:
                    if (
                        passenger_position != IN_TAXI
                        and (x, y) == flags_on_map[passenger_position]
                    ):  # Taxi is on passenger and can take it
                        passenger_position = IN_TAXI  # Passenger is now in taxi
                        ss2 = self.state_space[(x, y, passenger_position, destination)]
                        self.transition_matrix[aa][ss1, ss2] = 1.0
                    else:  # Taxi is not on passenger or passenger is already in taxi, no pickup
                        self.transition_matrix[aa][ss1, ss1] = 1.0

                    continue

                elif aa == DROPOFF:
                    if (
                        (x, y) == flags_on_map[destination]
                        and passenger_position == IN_TAXI
                    ):  # Passenger is in taxi, we are at destination
                        self.reward_matrix[ss1, aa] = 1.0

                        # Building the next state
                        ss2 = self.state_space[(x, y, destination, destination)]
                        self.transition_matrix[aa][ss1, ss2] = 1.0

                    else:  # Delivering passenger failed
                        self.transition_matrix[aa][ss1, ss1] = 1.0

                    continue

                x, y = self.move(x, y, aa)
                ss2 = self.state_space[(x, y, passenger_position, destination)]

                self.transition_matrix[aa][ss1, ss2] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
