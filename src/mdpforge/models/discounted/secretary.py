"""Classical secretary problem under a uniformly random permutation of N ranks.
State 2*(i-1)+b records observation i (1..N) and whether it is a record (b=1).
The first nonrecord state is unreachable from the natural start but is retained
for a rectangular encoding. Actions 0=continue, 1=accept; accepting pays the
conditional success probability i/N for a record, zero otherwise. The next
record indicator has probability 1/(i+1). The last state is absorbing, zero
reward. N=max(2,(state_dim-1)//2); discounting changes the classical objective.
"""

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "abstract",
    "description": "Classical best-choice optimal stopping with random arrival order.",
    "reference": "Ferguson (1989), Who Solved the Secretary Problem?, Statistical "
    "Science 4, 282–289. https://doi.org/10.1214/ss/1177012493",
    "sizes": {
        "tiny": {"requested_state_dim": 1001, "parameters": {}},
        "small": {"requested_state_dim": 10001, "parameters": {}},
        "medium": {"requested_state_dim": 160001, "parameters": {}},
        "large": {"requested_state_dim": 640001, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 21}


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class Model(MDP):
    def __init__(self, state_dim=1001):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        self.candidates = max(2, (self.requested_state_dim - 1) // 2)
        self.state_dim = 2 * self.candidates + 1
        self.action_dim = 2
        self.terminal = self.state_dim - 1
        self.name = f"{self.state_dim}_2_secretary"

    def _build_model(self):
        n = self.candidates
        s = np.arange(2 * n, dtype=np.int64)
        i = s // 2 + 1
        b = s % 2
        active = s[i < n]
        final = np.arange(2 * n - 2, 2 * n + 1)
        p = 1 / (i[i < n] + 1)
        rr = np.concatenate((active, active, final))
        cc = np.concatenate((2 * i[i < n], 2 * i[i < n] + 1, np.full(3, self.terminal)))
        data = np.concatenate((1 - p, p, np.ones(3)))
        shape = (self.state_dim,) * 2
        wait = csr_matrix((data, (rr, cc)), shape=shape)
        accept = csr_matrix(
            (
                np.ones(self.state_dim),
                (np.arange(self.state_dim), np.full(self.state_dim, self.terminal)),
            ),
            shape=shape,
        )
        self.transition_matrix = [wait, accept]
        self.reward_matrix = np.zeros((self.state_dim, 2))
        self.reward_matrix[:-1, 1] = b * i / n
