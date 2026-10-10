# Source: McCallum (1996), Whitehead's Blocks World revisited.
# Discrete/factored hand-eye coordination blocks task.

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 96, "action_dim": 6}

METADATA = {
    "tags": ["games", "puzzle"],
    "sizes": {
        "small": {
            "state_dim": 96,
            "parameters": {"state_dim": 96},
            "source": "measured",
        },
        "medium": {
            "state_dim": 9_600,
            "parameters": {"state_dim": 9_600},
            "source": "parameterized",
        },
        "large": {
            "state_dim": 96_000,
            "parameters": {"state_dim": 96_000},
            "source": "parameterized",
        },
    },
    "category": "games",
    "description": (
        "Factored block manipulation with looking, picking and lifting actions. "
        "Larger instances require sequentially solving additional block puzzles."
    ),
    "reference": "McCallum (1996), Whitehead's Blocks World revisited.",
}


class Model(MDP):
    """Factored Blocks World with exactly ``state_dim`` states.

    A 96-state level contains the original red/blue/green puzzle. With more
    states, levels are connected sequentially: lifting green on level j>0
    resets the puzzle on level j-1. Any remainder (<96 states) forms a short
    preparatory chain in which action 3 advances toward the first full puzzle.
    Consequently, no states are silently added, rounded, or discarded.

    Values below 96 use only the preparatory chain (not the original puzzle).
    This scaling increases task horizon, not the combinatorial complexity of
    an individual Blocks World level.

    Actions: 0=look_red, 1=look_blue, 2=look_green, 3=pick,
             4=put_aside, 5=lift_green.
    """

    _LEVEL_SIZE = 96

    def __init__(self, state_dim: int = 96, action_dim: int = 6):
        if isinstance(state_dim, (bool, np.bool_)) or not isinstance(
            state_dim, (int, np.integer)
        ):
            raise TypeError("state_dim must be a positive integer")
        if state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if action_dim != 6:
            raise ValueError("Blocks World has exactly six actions")

        self.action_dim = 6
        self.focus_values = 3  # red, blue, green, per puzzle level
        self.cover_values = 2
        self.holding_values = 4  # none, red, blue, green
        self.done_values = 2
        self.state_dim = int(state_dim)
        self.n_levels, self.n_preparation = divmod(self.state_dim, self._LEVEL_SIZE)
        self._world_limit = self.n_levels * self._LEVEL_SIZE
        self._start_local = self._encode(0, True, True, 0, False)
        self.initial_state = (
            self.state_dim - 1
            if self.n_preparation
            else (self.n_levels - 1) * self._LEVEL_SIZE + self._start_local
        )
        self.name = f"{self.state_dim}_{self.action_dim}_blocks_world"

    def _build_model(self):
        """Build sparse deterministic transitions in O(state_dim * action_dim)."""
        n = self.state_dim
        self.reward_matrix = np.empty((n, self.action_dim), dtype=float)
        self.transition_matrix = []

        # One nonzero per row: CSR can be assembled without LIL/COO overhead.
        index_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
        indptr = np.arange(n + 1, dtype=index_dtype)
        data = np.ones(n, dtype=float)
        world_end = self._world_limit
        world_ids = np.arange(world_end, dtype=index_dtype)
        local_ids = world_ids % self._LEVEL_SIZE
        levels = world_ids // self._LEVEL_SIZE
        prep_ids = np.arange(world_end, n, dtype=index_dtype)

        # Tabulate the unchanged original 96-state dynamics just once.
        base_next = np.empty((self.action_dim, self._LEVEL_SIZE), dtype=np.int32)
        base_reward = np.empty((self.action_dim, self._LEVEL_SIZE), dtype=float)
        for s in range(self._LEVEL_SIZE):
            focus, red_cover, blue_cover, holding, done = self._decode(s)
            for a in range(self.action_dim):
                if done:
                    ns, reward = s, 0.0
                else:
                    ns_tuple, reward = self._step(
                        focus, red_cover, blue_cover, holding, a
                    )
                    ns = self._encode(*ns_tuple)
                base_next[a, s] = ns
                base_reward[a, s] = reward

        for a in range(self.action_dim):
            next_states = np.empty(n, dtype=index_dtype)
            if world_end:
                next_states[:world_end] = (
                    levels * self._LEVEL_SIZE + base_next[a, local_ids]
                )
                self.reward_matrix[:world_end, a] = base_reward[a, local_ids]

                # Finishing a level unlocks the next one. The final level
                # retains the original terminal, absorbing dynamics.
                if a == 5 and self.n_levels > 1:
                    advance = (levels > 0) & (base_reward[a, local_ids] == 50.0)
                    next_states[:world_end][advance] = (
                        (levels[advance] - 1) * self._LEVEL_SIZE
                        + self._start_local
                    )

            if self.n_preparation:
                next_states[world_end:] = prep_ids
                self.reward_matrix[world_end:, a] = -1.0
                if a == 3:
                    next_states[world_end:] = prep_ids - 1
                    # Enter the actual puzzle, not an arbitrary local state.
                    if self.n_levels:
                        next_states[world_end] = (
                            (self.n_levels - 1) * self._LEVEL_SIZE
                            + self._start_local
                        )
                # Below 96 states, the preparation chain ends in state 0.
                if not self.n_levels:
                    next_states[0] = 0
                    self.reward_matrix[0, a] = 0.0

            self.transition_matrix.append(
                csr_matrix((data, next_states, indptr), shape=(n, n))
            )

    def transition(self, s: int, action: int) -> tuple[int, float]:
        """Compute a transition without constructing sparse matrices."""
        if not 0 <= s < self.state_dim:
            raise ValueError("state out of range")
        if not 0 <= action < self.action_dim:
            raise ValueError("action out of range")

        if s >= self._world_limit:
            if self.n_levels == 0 and s == 0:
                return 0, 0.0
            if action != 3:
                return s, -1.0
            if s == self._world_limit and self.n_levels:
                return (self.n_levels - 1) * self._LEVEL_SIZE + self._start_local, -1.0
            return s - 1, -1.0

        level, local = divmod(s, self._LEVEL_SIZE)
        focus, red_cover, blue_cover, holding, done = self._decode(local)
        if done:
            return s, 0.0
        next_tuple, reward = self._step(
            focus, red_cover, blue_cover, holding, action
        )
        if level > 0 and next_tuple[-1]:
            return (level - 1) * self._LEVEL_SIZE + self._start_local, reward
        return level * self._LEVEL_SIZE + self._encode(*next_tuple), reward

    def _encode(self, focus, red_cover, blue_cover, holding, done):
        x = focus
        for v, base in [(red_cover, 2), (blue_cover, 2), (holding, 4), (done, 2)]:
            x = x * base + int(v)
        return x

    def _decode(self, s):
        done = bool(s % 2)
        s //= 2
        holding = s % 4
        s //= 4
        blue_cover = bool(s % 2)
        s //= 2
        red_cover = bool(s % 2)
        s //= 2
        focus = s
        return focus, red_cover, blue_cover, holding, done

    def _green_visible(self, red_cover, blue_cover):
        return not red_cover and not blue_cover

    def _step(self, focus, red_cover, blue_cover, holding, action):
        reward = -1.0
        done = False
        if action in (0, 1, 2):
            focus = action
        elif action == 3 and holding == 0:
            if focus == 0 and red_cover:
                holding = 1
                reward = 2.0
            elif focus == 1 and blue_cover:
                holding = 2
                reward = 2.0
            elif focus == 2 and self._green_visible(red_cover, blue_cover):
                holding = 3
                reward = 10.0
            else:
                reward = -5.0
        elif action == 4:
            if holding == 1:
                red_cover = False
                holding = 0
                reward = 5.0
            elif holding == 2:
                blue_cover = False
                holding = 0
                reward = 5.0
            else:
                reward = -5.0
        elif action == 5:
            if holding == 3 and self._green_visible(red_cover, blue_cover):
                done = True
                reward = 50.0
            else:
                reward = -10.0
        return (focus, red_cover, blue_cover, holding, done), reward
