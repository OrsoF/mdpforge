"""Atomic, verified NPZ storage; no untrusted pickle deserialization."""

import json
import os
import tempfile
from contextlib import contextmanager
from zipfile import BadZipFile

import numpy as np

from mdpforge.utils.cache_keys import array_digest, stable_json, value_metadata


class CacheCorruptError(ValueError):
    """A cache entry is incomplete, corrupted or incompatible."""


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


def write_npz(path, **arrays):
    """Publish NPZ arrays atomically through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".cache-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


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
        write_npz(
            path,
            metadata=np.frombuffer(stable_json(metadata).encode(), dtype=np.uint8),
            value=value,
        )
    return value
