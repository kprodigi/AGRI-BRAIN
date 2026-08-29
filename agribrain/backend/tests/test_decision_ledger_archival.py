"""Focused tests for crash-safe and compressed DecisionLedger archiving."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest
from src.chain import decision_ledger as ledger_module
from src.chain.decision_ledger import DecisionLedger, read_jsonl_gzip


def _ledger() -> DecisionLedger:
    ledger = DecisionLedger({"scenario": "heatwave", "episode_index": 1})
    ledger.append({"step": 0, "value": 0.12345678901234568, "nested": [1, 2]})
    ledger.append({"step": 1, "value": -2.5, "nested": {"active": True}})
    return ledger


def test_write_jsonl_preserves_historical_bytes_and_merkle(tmp_path: Path) -> None:
    ledger = _ledger()
    path = tmp_path / "episode.jsonl"
    ledger.write_jsonl(path)

    snapshot = ledger.evidence_payload()
    expected_lines = [
        json.dumps(snapshot["header"], sort_keys=True, default=str),
        *[
            json.dumps(record, sort_keys=True, default=str)
            for record in snapshot["records"]
        ],
    ]
    assert path.read_bytes() == (
        os.linesep.join(expected_lines) + os.linesep
    ).encode("utf-8")
    assert snapshot["header"]["merkle_root"] == ledger.merkle_root()


def test_compressed_archive_bytes_are_deterministic_and_lossless(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    first_receipt = _ledger().write_jsonl_gzip(first)
    second_receipt = _ledger().write_jsonl_gzip(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt.literal_sha256 == second_receipt.literal_sha256
    assert first_receipt.literal_bytes == first.stat().st_size
    assert b"0.12345678901234568" in gzip.decompress(first.read_bytes())


def test_compressed_readback_validates_leaf_and_merkle_integrity(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl.gz"
    receipt = _ledger().write_jsonl_gzip(path)
    payload = read_jsonl_gzip(
        path,
        expected_literal_sha256=receipt.literal_sha256,
    )
    assert payload["header"]["merkle_root"] == receipt.merkle_root
    assert payload["header"]["n_records"] == receipt.n_records == 2

    lines = gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
    record = json.loads(lines[1])
    record["value"] = 999.0
    lines[1] = json.dumps(record, sort_keys=True, default=str)
    tampered = tmp_path / "tampered.jsonl.gz"
    tampered.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))
    with pytest.raises(ValueError, match="leaf hash mismatch"):
        read_jsonl_gzip(tampered)


def test_atomic_failure_preserves_existing_file_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "episode.jsonl"
    original = b"previous-complete-ledger\n"
    path.write_bytes(original)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated promotion failure")

    monkeypatch.setattr(ledger_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated promotion failure"):
        _ledger().write_jsonl(path)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
