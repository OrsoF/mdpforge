"""Stable recipe and numerical identities for versioned MDP caches."""

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse

SCHEMA_VERSION = 2


class CacheIdentityError(ValueError):
    """A model recipe cannot be represented with a reliable cache identity."""


def _canonical(value):
    """Encode only values with deterministic, unambiguous representations."""
    if value is None or isinstance(value, (str, bool, np.bool_)):
        return bool(value) if isinstance(value, np.bool_) else value
    if isinstance(value, (int, np.integer)):
        return {"int": str(int(value))}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise CacheIdentityError("Non-finite recipe parameter")
        return {"float": number.hex()}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise CacheIdentityError("Array parameter needs a numeric dtype")
        return {"array": array_digest(value)}
    if issparse(value):
        return {"sparse": sparse_digest(value)}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [_canonical(v) for v in value]}
    if isinstance(value, dict):
        pairs = [(_canonical(k), _canonical(v)) for k, v in value.items()]
        pairs.sort(key=lambda pair: stable_json(pair[0]))
        return {"dict": pairs}
    if isinstance(value, (set, frozenset)):
        elements = sorted((_canonical(v) for v in value), key=stable_json)
        return {"set": elements}
    raise CacheIdentityError(f"Unsupported recipe parameter: {type(value).__name__}")


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _update_array(digest, value):
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise CacheIdentityError(f"Unsupported numerical dtype: {arr.dtype}")
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


def recipe_identity(model, *, normalize_reward=False):
    """Return a recipe identity or None when the model does not opt in."""
    version = getattr(model, "CACHE_VERSION", None)
    if version is None:
        return None
    if not isinstance(version, int) or version < 1:
        raise CacheIdentityError("CACHE_VERSION must be a positive integer or None")
    config = model.get_cache_config()
    if config is None:
        return None
    implementation = type(model)
    source_file = inspect.getsourcefile(implementation)
    # Source changes invalidate recipes; missing source requires explicit versioning.
    source_hash = (
        hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
        if source_file and Path(source_file).is_file()
        else None
    )
    identity = {
        "schema": SCHEMA_VERSION,
        "kind": "model",
        "class": f"{implementation.__module__}.{implementation.__qualname__}",
        "version": version,
        "source_sha256": source_hash,
        "normalization": bool(normalize_reward),
        "config": _canonical(config),
    }
    return hashlib.sha256(stable_json(identity).encode()).hexdigest()


def value_identity(model, discount, precision):
    """Key value references by exact numerical content and solve settings."""
    from mdpforge.solvers import mdpforge_vi

    source_path = Path(mdpforge_vi.__file__)
    details = {
        "schema": SCHEMA_VERSION,
        "kind": "value",
        "model_sha256": numerical_identity(model),
        "discount": float(discount).hex(),
        "precision": float(precision).hex(),
        "solver": "mdpforge_vi",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(stable_json(details).encode()).hexdigest()
