"""Reduced solitaire Yahtzee (not the full 13-category game). Up to 4 six-sided
dice and the first K categories from chance, Yahtzee (50 for all dice equal),
ones, twos, three-of-a-kind. No bonus or joker. Three-of-a-kind requires at
least three matching dice. The all-equal category is generalized to D dice.
State (used-category mask, sorted dice, remaining rerolls 0..2); all categories
used is a single absorbing terminal. Actions 0..2**D-1 keep a positional
subset of sorted dice and reroll the rest; actions 2**D..2**D+K-1 score.
An unavailable action is replaced by scoring the first unused category.
Scoring rolls the next hand immediately; its exact multinomial distribution
is used. Initial hands use that same law with mask=0 and two rerolls.
state_dim is a target: if dice_count/category_count omitted choose the largest
supported preset not exceeding it (minimum tiny); otherwise both explicit
values set the exact size. Discounting changes the original total-score task.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from math import comb, factorial

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "games_and_puzzles",
    "description": "Reduced solitaire Yahtzee with exact multiset roll probabilities.",
    "reference": "Glenn (2007), Computer Strategies for Solitaire Yahtzee, IEEE CIG. "
    "https://doi.org/10.1109/CIG.2007.368089",
    "sizes": {
        "tiny": {
            "requested_state_dim": 190,
            "parameters": {"dice_count": 2, "category_count": 2},
        },
        "small": {
            "requested_state_dim": 1177,
            "parameters": {"dice_count": 3, "category_count": 3},
        },
        "medium": {
            "requested_state_dim": 2521,
            "parameters": {"dice_count": 3, "category_count": 4},
        },
        "large": {
            "requested_state_dim": 11719,
            "parameters": {"dice_count": 4, "category_count": 5},
        },
    },
}

TEST_PARAMETERS = {"state_dim": 190, "dice_count": 2, "category_count": 2}


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matrix(n, row_fn):
    """Build CSR directly from rows of (destination, probability) pairs."""
    indptr = [0]
    indices = []
    data = []
    for s in range(n):
        for t, p in row_fn(s):
            if p > 0.0:
                indices.append(t)
                data.append(p)
        indptr.append(len(indices))
    matrix = csr_matrix((data, indices, indptr), shape=(n, n), dtype=float)
    matrix.sum_duplicates()
    return matrix


class Model(MDP):
    def __init__(self, state_dim=190, dice_count=None, category_count=None):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        if dice_count is None and category_count is None:
            dice_count, category_count = 2, 2
            for d, k in ((2, 2), (3, 3), (3, 4), (4, 5)):
                if 3 * comb(d + 5, 5) * (2**k - 1) + 1 <= self.requested_state_dim:
                    dice_count, category_count = d, k
        if dice_count is None or category_count is None:
            raise ValueError("specify both dice_count and category_count")
        self.dice_count = _positive_int(dice_count, "dice_count")
        self.category_count = _positive_int(category_count, "category_count")
        if not 2 <= self.dice_count <= 4 or not 2 <= self.category_count <= 5:
            raise ValueError("dice_count must be 2..4 and category_count 2..5")
        if self.dice_count < 3 and self.category_count == 5:
            raise ValueError("three-of-a-kind requires at least three dice")
        self.hand_count = comb(self.dice_count + 5, 5)
        self.full_mask = 2**self.category_count - 1
        self.state_dim = 3 * self.hand_count * self.full_mask + 1
        self.action_dim = 2**self.dice_count + self.category_count
        self.terminal = self.state_dim - 1
        self.name = f"{self.state_dim}_{self.action_dim}_yahtzee_{self.dice_count}_{self.category_count}"

    def _score(self, hand, category):
        counts = sorted(Counter(hand).values())
        if category == 0:
            return sum(hand)
        if category == 1:
            return 50 if len(counts) == 1 else 0
        if category == 2:
            return hand.count(1)
        if category == 3:
            return 2 * hand.count(2)
        return sum(hand) if max(counts) >= 3 else 0

    def _build_model(self):
        d, k, h = self.dice_count, self.category_count, self.hand_count
        hands = list(combinations_with_replacement(range(1, 7), d))
        hand_index = {hand: i for i, hand in enumerate(hands)}

        @lru_cache(None)
        def rolls(keep):
            missing = d - len(keep)
            out = []
            for roll in combinations_with_replacement(range(1, 7), missing):
                multiplicity = factorial(missing)
                for count in Counter(roll).values():
                    multiplicity //= factorial(count)
                out.append(
                    (hand_index[tuple(sorted(keep + roll))], multiplicity / 6**missing)
                )
            return tuple(out)

        all_rolls = rolls(())
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        self.transition_matrix = []
        for a in range(self.action_dim):

            def row(s):
                if s == self.terminal:
                    return [(s, 1.0)]
                mask, rem = divmod(s, 3 * h)
                r, hand_id = divmod(rem, h)
                hand = hands[hand_id]
                if a < 2**d and r > 0:
                    keep = tuple(hand[i] for i in range(d) if a & (1 << i))
                    return [(mask * 3 * h + (r - 1) * h + j, p) for j, p in rolls(keep)]
                category = a - 2**d
                if category < 0 or mask & (1 << category):
                    category = next(j for j in range(k) if not mask & (1 << j))
                self.reward_matrix[s, a] = self._score(hand, category)
                new_mask = mask | (1 << category)
                if new_mask == self.full_mask:
                    return [(self.terminal, 1.0)]
                return [(new_mask * 3 * h + 2 * h + j, p) for j, p in all_rolls]

            self.transition_matrix.append(_matrix(self.state_dim, row))
