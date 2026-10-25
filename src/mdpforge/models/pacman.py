"""Reduced single-ghost Pac-Man MDP, not a conversion of the QVBS JANI file."""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "games_and_puzzles",
    "description": (
        "Reduced Pac-Man: two pellets, moving ghost, capture and terminal collection."
    ),
    "reference": "QVBS Pacman benchmark: https://qcomp.org/benchmarks/",
    "sizes": {
        "small": {"requested_state_dim": 6600, "parameters": {}},
        "medium": {"requested_state_dim": 26400, "parameters": {}},
        "large": {"requested_state_dim": 105600, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 326}


class Model(MDP):
    """State (player_cell, ghost_cell, collected_mask) plus win/loss."""

    def __init__(self, state_dim: int = 326):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        self.width = max(3, round((max(0, state_dim - 2) / 4) ** 0.25))
        self.cells = self.width * self.width
        super().__init__(4 * self.cells * self.cells + 2, 4)
        self.requested_state_dim = state_dim
        self.win, self.lose = self.state_dim - 2, self.state_dim - 1
        self.pellets = (self.width - 1, self.cells - 1)
        self.name = (
            f"{self.state_dim}_{self.action_dim}_pacman_1_{self.width}_{self.width}"
        )

    def _encode(self, pac, ghost, mask):
        return ((pac * self.cells + ghost) << 2) | mask

    def _step(self, pos, action):
        y, x = divmod(pos, self.width)
        dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
        return min(self.width - 1, max(0, y + dy)) * self.width + min(
            self.width - 1, max(0, x + dx)
        )

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        neighbors = [
            sorted({self._step(g, a) for a in range(4)}) for g in range(self.cells)
        ]
        for pac in range(self.cells):
            for ghost in range(self.cells):
                for mask in range(4):
                    s = self._encode(pac, ghost, mask)
                    for action in range(4):
                        if pac == ghost:
                            builder.add(action, s, self.lose, 1)
                            continue
                        next_pac = self._step(pac, action)
                        next_mask = mask
                        for bit, cell in enumerate(self.pellets):
                            if next_pac == cell:
                                next_mask |= 1 << bit
                        gained = (next_mask & ~mask).bit_count()
                        builder.rewards[s, action] = -0.1 + 4 * gained
                        p = 1 / len(neighbors[ghost])
                        for next_ghost in neighbors[ghost]:
                            crossed = next_pac == ghost and next_ghost == pac
                            if next_pac == next_ghost or crossed:
                                target = self.lose
                                extra = -20
                            elif next_mask == 3:
                                target = self.win
                                extra = 20
                            else:
                                target = self._encode(next_pac, next_ghost, next_mask)
                                extra = 0
                            builder.rewards[s, action] += p * extra
                            builder.add(action, s, target, p)
        builder.absorbing(self.win)
        builder.absorbing(self.lose)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
