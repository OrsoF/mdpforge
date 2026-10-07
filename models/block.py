# Source: Meuleau et al. (1998). Solving very large weakly coupled Markov decision processes. AAAI.
# Random block MDP - Weakly coupled MDP:
# - Divide the state space in n blocks
# - Ensure a random evolution through each block
# - Connect blocks with sparse non zeros transitions

import numpy as np
from scipy.sparse import lil_matrix, random

from core.model import GenericModel

rnd_gen = np.random.default_rng(seed=0)


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.block_dim = state_dim // 10
        self.action_dim = action_dim
        self.sparsity_block = 0.85
        self.block_number = 10
        self.state_dim = self.block_dim * self.block_number
        self.connection_ratio = 0.1

        self.name = "{}_{}_block_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.sparsity_block,
            self.block_number,
        )

    def _build_model(self):
        random_generator = np.random.default_rng(seed=0)

        self.reward_matrix = self.state_dim * (
            random_generator.random(size=(self.state_dim, self.action_dim)) - 0.5
        )

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        self.add_block_transition()
        self.add_connections(self.connection_ratio)
        self.normalize()
        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    def build_random_block(self, size):
        return random(
            size,
            size,
            density=self.sparsity_block,
            format="lil",
            data_rvs=np.random.rand,
        )

    def add_block_transition(self):
        for aa in range(self.action_dim):
            for block_index in range(self.block_number):
                start_index = block_index * self.block_dim
                end_index = start_index + self.block_dim
                self.transition_matrix[aa][
                    start_index:end_index, start_index:end_index
                ] = self.build_random_block(self.block_dim)

    def _update_transition_matrix(self, number_connection, aa, block_1, block_2):
        if block_1 != block_2:
            start_state = block_1 * self.block_dim
            end_state = start_state + self.block_dim
            initial_states = rnd_gen.choice(
                range(start_state, end_state),
                size=number_connection,
            )

            start_state = block_2 * self.block_dim
            end_state = (block_2 + 1) * self.block_dim

            final_states = rnd_gen.choice(
                range(start_state, end_state),
                size=number_connection,
            )

            for ss1 in initial_states:
                for ss2 in final_states:
                    self.transition_matrix[aa][ss1, ss2] = rnd_gen.random()

    def add_connections(self, connection_ratio: float):
        number_connection = int(connection_ratio * self.block_dim)
        for aa in range(self.action_dim):
            for block_1 in range(self.block_number):
                for block_2 in range(self.block_number):
                    self._update_transition_matrix(
                        number_connection, aa, block_1, block_2
                    )

    def normalize(self):
        for aa in range(self.action_dim):
            for ss in range(self.state_dim):
                self.transition_matrix[aa][ss, 0] += 1e-6

            row_sum = {}

            row_indices, col_indices = self.transition_matrix[aa].nonzero()
            for ss1, ss2 in zip(row_indices, col_indices):
                if ss1 not in row_sum:
                    current_row_sum = self.transition_matrix[aa][[ss1], :].sum()
                    row_sum[ss1] = current_row_sum
                self.transition_matrix[aa][ss1, ss2] = (
                    self.transition_matrix[aa][ss1, ss2] / row_sum[ss1]
                )
