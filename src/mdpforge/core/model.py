from typing import Protocol

import numpy as np
from scipy.sparse import csr_matrix


class MDPProtocol(Protocol):
    """Shared data interface for built models and matrix-defined MDPs.

    Inheritance is not required; numerical validity is checked separately by
    validate_model().
    """

    state_dim: int
    action_dim: int
    transition_matrix: list[csr_matrix]
    reward_matrix: np.ndarray
    name: str
