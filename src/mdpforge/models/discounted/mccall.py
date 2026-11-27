"""Discrete-wage McCall model. States 0..n-1 are unemployed with current offer,
 n..2n-1 are employed forever at the corresponding wage. Actions 0=reject
(receive benefit and redraw), 1=accept (earn offered wage immediately and
move to employment). Both actions in employment earn the wage. Offers have
uniform probability over an equally spaced wage grid in [0,2]. state_dim is
rounded down to an even number (minimum 4). benefit is finite in [0,2].
The uniform wage law is a benchmark choice, not an empirical calibration.
"""

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "asset_management",
    "description": "Job-search optimal stopping with discrete wage offers and "
    "permanent employment.",
    "reference": "McCall (1970), Economics of Information and Job Search, QJE 84, "
    "113–126. https://doi.org/10.2307/1879403; "
    "https://python.quantecon.org/mccall_model.html",
    "sizes": {
        "tiny": {"requested_state_dim": 64, "parameters": {}},
        "small": {"requested_state_dim": 256, "parameters": {}},
        "medium": {"requested_state_dim": 1024, "parameters": {}},
        "large": {"requested_state_dim": 4096, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 17}


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class Model(MDP):
    def __init__(self, state_dim=64, benefit=0.5):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        if not np.isfinite(benefit) or not 0 <= benefit <= 2:
            raise ValueError("benefit must be in [0,2]")
        self.benefit = float(benefit)
        self.wage_count = max(2, self.requested_state_dim // 2)
        self.state_dim = 2 * self.wage_count
        self.action_dim = 2
        self.name = f"{self.state_dim}_2_mccall_{self.benefit}"

    def _build_model(self):
        n = self.wage_count
        wages = np.linspace(0, 2, n)
        shape = (2 * n, 2 * n)
        employed = np.arange(n, 2 * n, dtype=np.int32)
        indices = np.concatenate((np.tile(np.arange(n, dtype=np.int32), n), employed))
        ptr = np.concatenate(
            (np.arange(n + 1, dtype=np.int64) * n, n * n + np.arange(1, n + 1))
        )
        data = np.concatenate((np.full(n * n, 1 / n), np.ones(n)))
        reject = csr_matrix((data, indices, ptr), shape=shape)
        accept = csr_matrix(
            (np.ones(2 * n), (np.arange(2 * n), np.tile(employed, 2))), shape=shape
        )
        self.transition_matrix = [reject, accept]
        self.reward_matrix = np.column_stack(
            (np.concatenate((np.full(n, self.benefit), wages)), np.tile(wages, 2))
        )
