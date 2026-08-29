from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from hpc.preserved_raw_manifest import (
    _load_json,
    build_manifest,
    main as raw_manifest_main,
    validate_manifest_document,
    validate_manifest_payload,
)


COMMIT = "a" * 40
TREE = "b" * 64
TAG = "aaaaaaa_20260829_105800"


def _inputs(tmp_path: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    root = tmp_path / "seed_outputs"
    root.mkdir()
    (root / "seed_42.json").write_bytes(b'{"seed":42}\n')
    receipt = tmp_path / "submission.json"
    receipt.write_bytes(b'{"receipt":true}\n')
    return [("seed_outputs", root)], [("submission.json", receipt)]


def _build(tmp_path: Path) -> tuple[dict, list, list]:
    roots, files = _inputs(tmp_path)
    payload = build_manifest(
        kind="core",
        run_tag=TAG,
        simulation_commit=COMMIT,
        simulation_source_tree_sha256=TREE,
        roots=roots,
        files=files,
    )
    return payload, roots, files


def _validate(payload: dict, roots: list, files: list) -> None:
    validate_manifest_payload(
        payload,
        kind="core",
        run_tag=TAG,
        simulation_commit=COMMIT,
        simulation_source_tree_sha256=TREE,
        roots=roots,
        files=files,
    )


def test_manifest_binds_complete_literal_inventory(tmp_path: Path) -> None:
    payload, roots, files = _build(tmp_path)
    _validate(payload, roots, files)
    assert payload["file_count"] == 2
    assert [record["path"] for record in payload["files"]] == [
        "seed_outputs/seed_42.json", "submission.json",
    ]
    assert len(payload["payload_merkle_root"]) == 64
    assert len(payload["manifest_sha256"]) == 64
    validate_manifest_document(
        payload,
        kind="core",
        run_tag=TAG,
        simulation_commit=COMMIT,
        simulation_source_tree_sha256=TREE,
    )


def test_manifest_rejects_changed_missing_or_added_input(tmp_path: Path) -> None:
    payload, roots, files = _build(tmp_path)
    seed = roots[0][1] / "seed_42.json"
    seed.write_bytes(b'{"seed":43}\n')
    with pytest.raises(ValueError, match="inventory changed"):
        _validate(payload, roots, files)

    seed.write_bytes(b'{"seed":42}\n')
    (roots[0][1] / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory changed"):
        _validate(payload, roots, files)


def test_manifest_rejects_resigned_summary_or_merkle_tampering(tmp_path: Path) -> None:
    payload, roots, files = _build(tmp_path)
    candidate = deepcopy(payload)
    candidate["total_bytes"] += 1
    unsigned = dict(candidate)
    unsigned.pop("manifest_sha256")
    from hpc.preserved_raw_manifest import _canonical_sha256

    candidate["manifest_sha256"] = _canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="summary is inconsistent"):
        _validate(candidate, roots, files)


def test_manifest_rejects_symlinks_when_supported(tmp_path: Path) -> None:
    payload, roots, files = _build(tmp_path)
    link = roots[0][1] / "alias.json"
    try:
        link.symlink_to(roots[0][1] / "seed_42.json")
    except OSError:
        pytest.skip("symlinks are not available to this test user")
    with pytest.raises(ValueError, match="contains a symlink"):
        _validate(payload, roots, files)


def test_manifest_rejects_a_symlink_as_the_bound_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "value.json").write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available to this test user")
    with pytest.raises(ValueError, match="input binding.*symbolic link"):
        build_manifest(
            kind="core",
            run_tag=TAG,
            simulation_commit=COMMIT,
            simulation_source_tree_sha256=TREE,
            roots=[("seed_outputs", alias)],
            files=[],
        )


def test_manifest_reader_rejects_a_symlinked_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    alias = tmp_path / "manifest-alias.json"
    try:
        alias.symlink_to(manifest)
    except OSError:
        pytest.skip("symlinks are not available to this test user")
    with pytest.raises(ValueError, match="symbolic link"):
        _load_json(alias)


def test_manifest_validate_cli_rejects_a_symlinked_manifest(tmp_path: Path) -> None:
    payload, roots, files = _build(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    alias = tmp_path / "manifest-alias.json"
    try:
        alias.symlink_to(manifest)
    except OSError:
        pytest.skip("symlinks are not available to this test user")
    with pytest.raises(ValueError, match="symbolic link"):
        raw_manifest_main([
            "validate", "--manifest", str(alias), "--kind", "core",
            "--run-tag", TAG, "--simulation-commit", COMMIT,
            "--simulation-source-tree-sha256", TREE,
            "--input-root", f"seed_outputs={roots[0][1]}",
            "--input-file", f"submission.json={files[0][1]}",
        ])


def test_run_tag_must_remain_bound_to_simulation_commit(tmp_path: Path) -> None:
    roots, files = _inputs(tmp_path)
    with pytest.raises(ValueError, match="run tag is not simulation-commit-bound"):
        build_manifest(
            kind="core",
            run_tag="ccccccc_20260829_105800",
            simulation_commit=COMMIT,
            simulation_source_tree_sha256=TREE,
            roots=roots,
            files=files,
        )
