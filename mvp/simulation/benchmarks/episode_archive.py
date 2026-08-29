"""Lossless, deterministic helpers for archiving episode evidence.

This module contains no simulation or publication business logic.  It provides
small primitives that a future runner can use to retain an exact JSON-native
payload, a literal-byte digest, and a clearly scoped process-runtime receipt.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import numbers
import os
import tempfile
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # ``resource`` is unavailable on some platforms, including Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised by the Windows test run
    _resource = None


_RUSAGE_FIELDS = (
    "ru_utime",
    "ru_stime",
    "ru_maxrss",
    "ru_minflt",
    "ru_majflt",
    "ru_inblock",
    "ru_oublock",
    "ru_nvcsw",
    "ru_nivcsw",
)
_TYPE_TAG = "__agribrain_episode_json_type__"
_MAPPING_ITEMS = "mapping_items_v1"


def to_json_native(value: Any) -> Any:
    """Recursively convert supported values without rounding numeric leaves.

    NumPy arrays/scalars and similar objects are supported through ``tolist``.
    Non-finite numbers and unsupported objects are rejected rather than
    silently stringified or coerced. Mappings with non-string keys use a
    reversible tagged item-list representation so integer action keys survive
    a write/read/resume cycle exactly.
    """

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"non-finite numeric evidence value: {value!r}")
        return numeric
    if isinstance(value, Mapping):
        if _TYPE_TAG in value or any(not isinstance(key, str) for key in value):
            return {
                _TYPE_TAG: _MAPPING_ITEMS,
                "items": [
                    [to_json_native(key), to_json_native(item)]
                    for key, item in value.items()
                ],
            }
        converted: dict[str, Any] = {}
        for key, item in value.items():
            converted[key] = to_json_native(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_json_native(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is value:
            raise TypeError(f"tolist() did not convert {type(value).__name__}")
        return to_json_native(converted)
    raise TypeError(f"unsupported episode-evidence value: {type(value).__name__}")


def from_json_native(value: Any) -> Any:
    """Reverse tagged mappings emitted by :func:`to_json_native`."""

    if isinstance(value, list):
        return [from_json_native(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_TYPE_TAG, "items"} and value.get(_TYPE_TAG) == (
            _MAPPING_ITEMS
        ):
            items = value.get("items")
            if not isinstance(items, list):
                raise ValueError("tagged evidence mapping has invalid items")
            restored: dict[Any, Any] = {}
            for pair in items:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("tagged evidence mapping has an invalid pair")
                key = from_json_native(pair[0])
                item = from_json_native(pair[1])
                try:
                    if key in restored:
                        raise ValueError("tagged evidence mapping has duplicate keys")
                    restored[key] = item
                except TypeError as exc:
                    raise ValueError("tagged evidence mapping key is unhashable") from exc
            return restored
        return {key: from_json_native(item) for key, item in value.items()}
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for a losslessly converted payload."""

    native = to_json_native(payload)
    return json.dumps(
        native,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    """SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class ArchiveReceipt:
    """Literal on-disk identity returned by archive writes and reads."""

    path: str
    literal_sha256: str
    literal_bytes: int
    canonical_json_sha256: str
    canonical_json_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "literal_sha256": self.literal_sha256,
            "literal_bytes": self.literal_bytes,
            "canonical_json_sha256": self.canonical_json_sha256,
            "canonical_json_bytes": self.canonical_json_bytes,
        }


def _deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=buffer,
        compresslevel=9,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return buffer.getvalue()


def _receipt(path: Path, literal: bytes, canonical: bytes) -> ArchiveReceipt:
    return ArchiveReceipt(
        path=str(path),
        literal_sha256=hashlib.sha256(literal).hexdigest(),
        literal_bytes=len(literal),
        canonical_json_sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_json_bytes=len(canonical),
    )


def write_gzip_json_atomic(path: Path | str, payload: Any) -> ArchiveReceipt:
    """Atomically write deterministic gzip-compressed canonical JSON.

    The returned receipt hashes the literal gzip bytes as well as the
    uncompressed canonical JSON.  A temporary file is created in the target
    directory, flushed and fsynced, then promoted with ``os.replace``.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise ValueError(f"archive target is not a regular file: {target}")
    canonical = canonical_json_bytes(payload)
    literal = _deterministic_gzip(canonical)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(literal)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return _receipt(target, literal, canonical)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def read_gzip_json(
    path: Path | str,
    *,
    expected_literal_sha256: str | None = None,
    require_canonical: bool = True,
) -> tuple[Any, ArchiveReceipt]:
    """Safely read and validate one archive written by this module."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"archive source is not a regular non-symlink file: {source}")
    literal = source.read_bytes()
    literal_sha = hashlib.sha256(literal).hexdigest()
    if expected_literal_sha256 is not None and literal_sha != expected_literal_sha256:
        raise ValueError(
            "literal archive SHA-256 mismatch: "
            f"observed={literal_sha}, expected={expected_literal_sha256}"
        )
    try:
        canonical = gzip.decompress(literal)
    except (OSError, EOFError) as exc:
        raise ValueError(f"invalid gzip episode archive: {source}") from exc
    try:
        encoded_payload = json.loads(
            canonical.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON episode archive: {source}") from exc
    payload = from_json_native(encoded_payload)
    normalized = canonical_json_bytes(payload)
    if require_canonical and normalized != canonical:
        raise ValueError(f"episode archive does not contain canonical JSON: {source}")
    return payload, _receipt(source, literal, normalized)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds",
    ).replace("+00:00", "Z")


def _resource_snapshot() -> dict[str, int | float] | None:
    if _resource is None:
        return None
    usage = _resource.getrusage(_resource.RUSAGE_SELF)
    snapshot: dict[str, int | float] = {}
    for field in _RUSAGE_FIELDS:
        value = getattr(usage, field, None)
        if isinstance(value, numbers.Integral):
            snapshot[field] = int(value)
        elif isinstance(value, numbers.Real) and math.isfinite(float(value)):
            snapshot[field] = float(value)
    return snapshot


@dataclass(frozen=True)
class EpisodeRuntimeReceipt:
    """Measured process/runtime scope for one context-managed episode block."""

    utc_start: str
    utc_end: str
    wall_seconds: float
    process_cpu_seconds: float
    resource_available: bool
    resource_start: dict[str, int | float] | None
    resource_end: dict[str, int | float] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "utc_start": self.utc_start,
            "utc_end": self.utc_end,
            "wall_seconds": self.wall_seconds,
            "process_cpu_seconds": self.process_cpu_seconds,
            "resource_available": self.resource_available,
            "resource_start": self.resource_start,
            "resource_end": self.resource_end,
        }


class EpisodeRuntimeMeasurement(AbstractContextManager["EpisodeRuntimeMeasurement"]):
    """Context manager that measures wall, process CPU, and process resources."""

    def __init__(self) -> None:
        self._receipt: EpisodeRuntimeReceipt | None = None
        self._utc_start = ""
        self._wall_start = 0.0
        self._cpu_start = 0.0
        self._resource_start: dict[str, int | float] | None = None

    def __enter__(self) -> "EpisodeRuntimeMeasurement":
        if self._utc_start:
            raise RuntimeError("runtime measurement cannot be entered twice")
        self._utc_start = _utc_now()
        self._resource_start = _resource_snapshot()
        self._cpu_start = time.process_time()
        self._wall_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        wall_seconds = time.perf_counter() - self._wall_start
        cpu_seconds = time.process_time() - self._cpu_start
        self._receipt = EpisodeRuntimeReceipt(
            utc_start=self._utc_start,
            utc_end=_utc_now(),
            wall_seconds=float(wall_seconds),
            process_cpu_seconds=float(cpu_seconds),
            resource_available=self._resource_start is not None,
            resource_start=self._resource_start,
            resource_end=_resource_snapshot(),
        )
        return False

    @property
    def receipt(self) -> EpisodeRuntimeReceipt:
        if self._receipt is None:
            raise RuntimeError("runtime receipt is available only after context exit")
        return self._receipt


def measure_episode_runtime() -> EpisodeRuntimeMeasurement:
    """Return a fresh context manager for one explicitly scoped episode block."""

    return EpisodeRuntimeMeasurement()


__all__ = (
    "ArchiveReceipt",
    "EpisodeRuntimeMeasurement",
    "EpisodeRuntimeReceipt",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "from_json_native",
    "measure_episode_runtime",
    "read_gzip_json",
    "to_json_native",
    "write_gzip_json_atomic",
)
