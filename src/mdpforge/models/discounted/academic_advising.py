"""IPPC 2014 academic advising, generated binary-tree prerequisite graph (course
j has prerequisite floor((j-1)/2) if j>0), all courses required, at most one
course per semester. State is base-3 digits: 0=not taken, 1=failed, 2=passed.
Action 0=no course, action j+1=take course j. Passed courses remain passed;
pass probabilities .8 without prerequisites, .2+.8*k/(1+d) otherwise (d=1).
Reward is -5 while program incomplete, minus 1 first attempt or 2 retake.
state_dim rounds down to a power of 3, minimum one course. Full completion
is not forced terminal; the optimal action is to wait with zero reward.
"""

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "service_operations",
    "description": "Course scheduling with stochastic success and "
    "prerequisite-dependent pass rates.",
    "reference": "Libby Ferland, Academic Advising MDP, IPPC 2014. "
    "https://github.com/pyrddlgym-project/rddlrepository/tree/main/rddlrepository/archive/competitions/IPPC2014/AcademicAdvising/MDP",
    "sizes": {
        "tiny": {"requested_state_dim": 81, "parameters": {}},
        "small": {"requested_state_dim": 2187, "parameters": {}},
        "medium": {"requested_state_dim": 19683, "parameters": {}},
        "large": {"requested_state_dim": 177147, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 27}


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class Model(MDP):
    def __init__(self, state_dim=81):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        self.courses = 1
        while 3 ** (self.courses + 1) <= self.requested_state_dim:
            self.courses += 1
        self.state_dim = 3**self.courses
        self.action_dim = self.courses + 1
        self.name = f"{self.state_dim}_{self.action_dim}_academicadvising"

    def _build_model(self):
        s = np.arange(self.state_dim, dtype=np.int64)
        shape = (self.state_dim,) * 2
        penalty = np.where(s == self.state_dim - 1, 0.0, -5.0)
        self.reward_matrix = np.tile(penalty[:, None], (1, self.action_dim))
        self.transition_matrix = [csr_matrix((np.ones(len(s)), (s, s)), shape=shape)]
        for j in range(self.courses):
            weight = 3**j
            digit = (s // weight) % 3
            if j == 0:
                p = np.full(len(s), 0.8)
            else:
                pre = (s // 3 ** ((j - 1) // 2)) % 3
                p = 0.2 + 0.4 * (pre == 2)
            passed = digit == 2
            fail_dest = s + (1 - digit) * weight
            pass_dest = s + (2 - digit) * weight
            fail_dest = np.where(passed, s, fail_dest)
            data = np.concatenate((np.where(passed, 0, 1 - p), np.where(passed, 1, p)))
            mat = csr_matrix(
                (data, (np.tile(s, 2), np.concatenate((fail_dest, pass_dest)))),
                shape=shape,
            )
            mat.eliminate_zeros()
            self.transition_matrix.append(mat)
            self.reward_matrix[:, j + 1] -= np.where(digit == 0, 1, 2)
