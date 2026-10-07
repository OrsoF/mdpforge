# Source: McCallum (1996), Whitehead's Blocks World revisited.
# Discrete/factored hand-eye coordination blocks task.

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    """Small factored Blocks World / hand-eye task.

    There are red, blue, and green blocks.  Red and blue may cover green.  The
    agent can look at a block, pick it if visible, put held red/blue aside, and
    lift green once uncovered.  State is enumerated over factored variables.

    Actions: 0=look_red, 1=look_blue, 2=look_green, 3=pick, 4=put_aside, 5=lift_green.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = 6
        self.focus_values = 3  # red, blue, green
        self.cover_values = 2  # present/removed for red and blue cover blocks
        self.holding_values = 4  # none, red, blue, green
        self.done_values = 2
        self.state_dim = (
            self.focus_values
            * self.cover_values
            * self.cover_values
            * self.holding_values
            * self.done_values
        )
        self.name = f"{self.state_dim}_{self.action_dim}_blocks_world"

    def _build_model(self):
        self.reward_matrix = -np.ones((self.state_dim, self.action_dim), dtype=float)
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        for s in range(self.state_dim):
            focus, red_cover, blue_cover, holding, done = self._decode(s)
            if done:
                for a in range(self.action_dim):
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                continue
            for a in range(self.action_dim):
                ns_tuple, reward = self._step(focus, red_cover, blue_cover, holding, a)
                ns = self._encode(*ns_tuple)
                self.transition_matrix[a][s, ns] = 1.0
                self.reward_matrix[s, a] = reward
        self.transition_matrix = [m.tocsr() for m in self.transition_matrix]

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
