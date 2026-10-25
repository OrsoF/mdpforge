"""Numerical fingerprints for cached values and experiment metadata."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

SCHEMA_VERSION = 2


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _update_array(digest, value):
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"Unsupported numerical dtype: {arr.dtype}")
    dtype = arr.dtype.newbyteorder("<")
    arr = np.ascontiguousarray(arr, dtype=dtype)
    digest.update(stable_json([list(arr.shape), dtype.str]).encode())
    digest.update(arr.tobytes())


def array_digest(value):
    digest = hashlib.sha256()
    _update_array(digest, value)
    return digest.hexdigest()


def _normalized_csr(matrix):
    result = csr_matrix(matrix, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def sparse_digest(matrix):
    matrix = _normalized_csr(matrix)
    digest = hashlib.sha256()
    _update_array(digest, np.asarray(matrix.shape, dtype=np.int64))
    _update_array(digest, matrix.indptr.astype(np.int64))
    _update_array(digest, matrix.indices.astype(np.int64))
    _update_array(digest, matrix.data)
    return digest.hexdigest()


def numerical_identity(model):
    """Hash actual transitions and rewards, never a model's human-readable name."""
    digest = hashlib.sha256()
    digest.update(b"mdpforge-mdp-content-v2\0")
    digest.update(stable_json([int(model.state_dim), int(model.action_dim)]).encode())
    for matrix in model.transition_matrix:
        digest.update(bytes.fromhex(sparse_digest(matrix)))
    digest.update(bytes.fromhex(array_digest(model.reward_matrix)))
    return digest.hexdigest()


def value_metadata(model, discount, precision):
    """Describe the numerical content and solve settings of a value reference."""
    from mdpforge.solvers import mdpforge_vi

    source_path = Path(mdpforge_vi.__file__)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "value",
        "model_sha256": numerical_identity(model),
        "discount": float(discount).hex(),
        "precision": float(precision).hex(),
        "solver": "mdpforge_vi",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
