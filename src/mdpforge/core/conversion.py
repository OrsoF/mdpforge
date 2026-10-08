from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from mdpforge.core.model import GenericModel


NUMPY, SPARSE = "numpy", "sparse"


def model_to_numpy(model: "GenericModel") -> None:
    """Convert transition and reward matrices to dense numpy arrays in place."""
    try:
        model.transition_matrix = np.array(
            [matrix.toarray() for matrix in model.transition_matrix]
        )
    except AttributeError:
        pass

    try:
        model.reward_matrix = model.reward_matrix.toarray()
    except AttributeError:
        pass


def model_to_sparse(model: "GenericModel") -> None:
    """Convert transition matrices to a list of CSR sparse arrays in place."""
    model.transition_matrix = [csr_array(matrix) for matrix in model.transition_matrix]


def convert_model(model: "GenericModel", mode: str) -> None:
    """Convert model matrices to one of the supported internal formats."""
    if mode == NUMPY:
        model_to_numpy(model)
    elif mode == SPARSE:
        model_to_sparse(model)
    else:
        raise ValueError(f"Unknown model conversion mode: {mode}")
