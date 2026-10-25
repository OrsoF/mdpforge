"""Atomic, verified NPZ storage; no untrusted pickle deserialization."""

import json
import os
import tempfile
from contextlib import contextmanager
from zipfile import BadZipFile

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.utils.cache_keys import (
    array_digest,
    numerical_identity,
    stable_json,
    value_metadata,
)


class CacheCorruptError(ValueError):
    """A cache entry is incomplete, corrupted or incompatible."""


class CacheConflictError(ValueError):
    """Concurrent writers produced different data for the same identity."""


@contextmanager
def _write_lock(path):
    """Serialize publications across processes, including on Windows."""
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
            data = {
                key: archive[key].copy() for key in archive.files if key != "metadata"
            }
        if not isinstance(metadata, dict):
            raise ValueError("Invalid metadata")
        return metadata, data
    except (OSError, ValueError, KeyError, BadZipFile, EOFError, TypeError) as exc:
        raise CacheCorruptError(f"Invalid cache entry {path}: {exc}") from exc


def _write(path, metadata, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".cache-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.frombuffer(stable_json(metadata).encode(), dtype=np.uint8),
                **arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_model_entry(path, key):
    """Read and validate a model payload without modifying the caller."""
    metadata, arrays = _read(path)
    if (
        metadata.get("schema") != 2
        or metadata.get("kind") != "model"
        or metadata.get("key") != key
    ):
        raise CacheCorruptError("Model cache identity mismatch")
    try:
        states, actions = metadata["shape"]
        matrices = [
            csr_matrix(
                (arrays[f"t{i}_data"], arrays[f"t{i}_indices"], arrays[f"t{i}_indptr"]),
                shape=(states, states),
            )
            for i in range(actions)
        ]
        rewards = arrays["rewards"]
        if rewards.shape != (states, actions):
            raise ValueError("Reward dimensions do not match")
        from types import SimpleNamespace

        from mdpforge.core.validation import validate_model

        candidate = SimpleNamespace(
            name="cached_mdp",
            state_dim=states,
            action_dim=actions,
            transition_matrix=matrices,
            reward_matrix=rewards,
        )
        validate_model(candidate)
        if numerical_identity(candidate) != metadata["content_sha256"]:
            raise ValueError("Numerical content checksum mismatch")
        return candidate
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise CacheCorruptError(f"Invalid model entry {path}: {exc}") from exc


def save_model_entry(path, key, model):
    from mdpforge.core.validation import validate_model

    validate_model(model)
    checksum = numerical_identity(model)
    arrays = {"rewards": np.asarray(model.reward_matrix)}
    for i, source in enumerate(model.transition_matrix):
        matrix = csr_matrix(source, dtype=float, copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        matrix.eliminate_zeros()
        arrays[f"t{i}_data"] = matrix.data
        arrays[f"t{i}_indices"] = matrix.indices
        arrays[f"t{i}_indptr"] = matrix.indptr
    metadata = {
        "schema": 2,
        "kind": "model",
        "key": key,
        "shape": [int(model.state_dim), int(model.action_dim)],
        "content_sha256": checksum,
    }
    with _write_lock(path):
        if path.exists():
            try:
                previous = load_model_entry(path, key)
            except CacheCorruptError:
                pass
            else:
                if numerical_identity(previous) != checksum:
                    raise CacheConflictError(
                        "One recipe produced different model matrices"
                    )
                return
        _write(path, metadata, arrays)


def load_value_entry(path, key, model, discount, precision):
    """Return a reference, or None when its inputs or precision are stale."""
    metadata, arrays = _read(path)
    if (
        metadata.get("schema") != 2
        or metadata.get("kind") != "value"
        or metadata.get("key") != key
    ):
        raise CacheCorruptError("Value cache identity mismatch")
    try:
        stored_precision = float.fromhex(metadata["precision"])
        if not np.isfinite(stored_precision) or stored_precision <= 0:
            raise ValueError("Invalid cached precision")
        expected = value_metadata(model, discount, precision)
        if stored_precision > precision or any(
            metadata.get(name) != value
            for name, value in expected.items()
            if name != "precision"
        ):
            return None
        value = arrays["value"]
        if metadata.get("value_sha256") != array_digest(value):
            raise ValueError("Value checksum mismatch")
        return value
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheCorruptError(f"Invalid value entry {path}: {exc}") from exc


def save_value_entry(path, key, model, discount, precision, value, *, replace=False):
    value = np.asarray(value, dtype=float)
    metadata = value_metadata(model, discount, precision) | {
        "key": key,
        "value_sha256": array_digest(value),
    }
    with _write_lock(path):
        if path.exists() and not replace:
            try:
                previous = load_value_entry(path, key, model, discount, precision)
            except CacheCorruptError:
                pass
            else:
                if previous is not None:
                    return previous
        _write(path, metadata, {"value": value})
    return value
