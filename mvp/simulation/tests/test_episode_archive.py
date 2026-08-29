"""Focused tests for lossless episode-evidence archiving helpers."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mvp.simulation.benchmarks.episode_archive import (
    canonical_json_bytes,
    canonical_json_sha256,
    measure_episode_runtime,
    read_gzip_json,
    to_json_native,
    write_gzip_json_atomic,
)


def test_json_native_conversion_handles_numpy_and_nested_values() -> None:
    payload = {
        "array": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "scalar": np.float64(0.125),
        "nested": (np.bool_(True), {"value": np.int32(7)}),
    }
    assert to_json_native(payload) == {
        "array": [[1, 2], [3, 4]],
        "scalar": 0.125,
        "nested": [True, {"value": 7}],
    }


def test_exact_float_round_trip_has_no_archive_rounding(tmp_path: Path) -> None:
    value = math.nextafter(0.12345678901234568, math.inf)
    path = tmp_path / "episode.json.gz"
    write_gzip_json_atomic(path, {"value": value})
    payload, _receipt = read_gzip_json(path)
    assert payload["value"].hex() == value.hex()


def test_integer_mapping_keys_round_trip_without_string_coercion(
    tmp_path: Path,
) -> None:
    original = {
        "action_counts": {0: 11, 1: 7, 2: 3},
        "__agribrain_episode_json_type__": "ordinary user field",
    }
    path = tmp_path / "integer-keys.json.gz"
    write_gzip_json_atomic(path, original)
    restored, _receipt = read_gzip_json(path)
    assert restored == original
    assert set(restored["action_counts"]) == {0, 1, 2}


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), np.float64(np.nan)],
)
def test_nonfinite_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"value": value})


def test_canonical_hash_and_gzip_bytes_are_deterministic(tmp_path: Path) -> None:
    left = {"z": [3.0, 2.0], "a": {"x": 1}}
    right = {"a": {"x": 1}, "z": [3.0, 2.0]}
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    first_receipt = write_gzip_json_atomic(first, left)
    second_receipt = write_gzip_json_atomic(second, right)

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_sha256(left) == canonical_json_sha256(right)
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt.literal_sha256 == second_receipt.literal_sha256


def test_atomic_write_readback_and_literal_hash_validation(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "episode.json.gz"
    write_receipt = write_gzip_json_atomic(target, {"seed": 42, "values": [1, 2]})
    payload, read_receipt = read_gzip_json(
        target,
        expected_literal_sha256=write_receipt.literal_sha256,
    )

    assert payload == {"seed": 42, "values": [1, 2]}
    assert read_receipt == write_receipt
    assert write_receipt.literal_bytes == target.stat().st_size
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_gzip_json(target, expected_literal_sha256="0" * 64)


def test_runtime_measurement_returns_utc_cpu_wall_and_resource_receipt() -> None:
    measurement = measure_episode_runtime()
    with pytest.raises(RuntimeError, match="after context exit"):
        _ = measurement.receipt

    with measurement:
        sum(index * index for index in range(5_000))

    receipt = measurement.receipt
    assert receipt.utc_start.endswith("Z") and receipt.utc_end.endswith("Z")
    assert datetime.fromisoformat(receipt.utc_start.replace("Z", "+00:00")) <= (
        datetime.fromisoformat(receipt.utc_end.replace("Z", "+00:00"))
    )
    assert receipt.wall_seconds >= 0.0
    assert receipt.process_cpu_seconds >= 0.0
    assert receipt.resource_available == (receipt.resource_start is not None)
    if receipt.resource_available:
        assert receipt.resource_end is not None
        assert "ru_utime" in receipt.resource_start
        assert "ru_stime" in receipt.resource_end
    assert to_json_native(receipt.as_dict()) == receipt.as_dict()
