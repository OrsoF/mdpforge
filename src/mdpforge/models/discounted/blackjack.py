"""Infinite deck (ranks 1..9 with probability 1/13, ten with 4/13).
State (player total, usable ace, dealer upcard); hard totals 2..target,
soft totals 12..target. All hard totals are included to model drawing before 12.
Actions 0=hit, 1=stick. The dealer draws until target-4 (17 for target=21),
stands on soft 17, and draws independently; no natural bonus or peek rule.
Reward +/-1 or 0 on settlement, zero otherwise and at the absorbing terminal.
state_dim sets target=max(21,(state_dim+119)//20), yielding 20*target-119
states. Targets >21 are explicit generalized blackjack variants, not canonical
rules. One-step dealer settlement integrates all dealer draws into expected reward.
"""

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "games_and_puzzles",
    "description": "Infinite-deck blackjack with a scalable target score and exact "
    "dealer probabilities.",
    "reference": "Sutton & Barto (2018), Reinforcement Learning: An Introduction, "
    "example 5.1. http://incompleteideas.net/book/the-book-2nd.html",
    "sizes": {
        "tiny": {"requested_state_dim": 301, "parameters": {}},
        "small": {"requested_state_dim": 701, "parameters": {}},
        "medium": {"requested_state_dim": 1501, "parameters": {}},
        "large": {"requested_state_dim": 3101, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 301}


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
    def __init__(self, state_dim=301):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        self.target = max(21, (self.requested_state_dim + 119) // 20)
        self.state_dim = 20 * self.target - 119
        self.action_dim = 2
        self.terminal = self.state_dim - 1
        self.name = f"{self.state_dim}_2_blackjack_{self.target}"

    def _add_card(self, total, soft, card):
        total += card
        if card == 1 and not soft and total + 10 <= self.target:
            total += 10
            soft = True
        if total > self.target and soft:
            total -= 10
            soft = False
        return total, soft

    def _build_model(self):
        cards = tuple((c, 4 / 13 if c == 10 else 1 / 13) for c in range(1, 11))
        states = [
            (t, b, d)
            for b in (False, True)
            for t in range(12 if b else 2, self.target + 1)
            for d in range(1, 11)
        ]
        index = {x: i for i, x in enumerate(states)}
        # Compute dealer distributions in increasing raw-card-sum dependency order.
        threshold = self.target - 4
        # Dictionary keyed by (total, soft); dealer's raw total always increases.
        dist = {}
        for raw in range(self.target, -1, -1):
            for soft in (False, True):
                total = raw + 10 * soft
                if total > self.target:
                    continue
                if total >= threshold:
                    dist[total, soft] = {total: 1.0}
                else:
                    result = defaultdict(float)
                    for c, p in cards:
                        nxt, b = self._add_card(total, soft, c)
                        if nxt > self.target:
                            result[-1] += p
                        else:
                            for v, q in dist[nxt, b].items():
                                result[v] += p * q
                    dist[total, soft] = dict(result)
        dealer = {}
        for d in range(1, 11):
            result = defaultdict(float)
            for c, p in cards:
                t, b = self._add_card(11 if d == 1 else d, d == 1, c)
                for v, q in dist[t, b].items():
                    result[v] += p * q
            dealer[d] = result
        self.reward_matrix = np.zeros((self.state_dim, 2))

        def hit(s):
            if s == self.terminal:
                return [(s, 1.0)]
            t, b, d = states[s]
            row = []
            for c, p in cards:
                nxt, soft = self._add_card(t, b, c)
                if nxt > self.target:
                    row.append((self.terminal, p))
                    self.reward_matrix[s, 0] -= p
                else:
                    row.append((index[nxt, soft, d], p))
            return row

        self.transition_matrix = [
            _matrix(self.state_dim, hit),
            csr_matrix(
                (
                    np.ones(self.state_dim),
                    (np.arange(self.state_dim), np.full(self.state_dim, self.terminal)),
                ),
                shape=(self.state_dim,) * 2,
            ),
        ]
        for s, (t, b, d) in enumerate(states):
            self.reward_matrix[s, 1] = sum(
                p * ((t > v) - (t < v)) for v, p in dealer[d].items()
            )
