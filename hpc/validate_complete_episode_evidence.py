#!/usr/bin/env python3
"""Validate and hash-manifest every executed episode before an HPC task exits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _REPO_ROOT / "agribrain" / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from src.chain.decision_ledger import (  # noqa: E402
    read_jsonl_gzip,
    validate_evidence_payload,
)

from mvp.simulation.benchmarks.episode_archive import (  # noqa: E402
    canonical_json_bytes,
    canonical_json_sha256,
    read_gzip_json,
)


def _stream_seed(
    benchmark_seed: int, scenario: str, episode_index: int, stream: str,
) -> int:
    key = (
        f"agribrain-v3|{int(benchmark_seed)}|{scenario}|"
        f"{int(episode_index)}|{stream}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _literal_binding(path: Path) -> tuple[str, int]:
    literal = path.read_bytes()
    return hashlib.sha256(literal).hexdigest(), len(literal)


def _learner_continuation_payload(state: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(state)
    if projected.get("theta_learners"):
        projected.pop("theta_learner", None)
    for field in ("learners_frozen", "learner_phase", "freeze_reason"):
        projected.pop(field, None)
    return projected


def _safe_relative(root: Path, relative: object) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or not value.parts or any(
        part in {"", ".", ".."} for part in value.parts
    ):
        raise ValueError(f"unsafe evidence-relative path: {relative!r}")
    target = root.joinpath(*value.parts)
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"evidence path escapes its ledger root: {relative!r}") from exc
    return target


def _load_plain_ledger(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"plain ledger is not a regular file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"plain decision ledger framing is invalid: {path}")
    try:
        values = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError(f"plain decision ledger JSON is invalid: {path}") from exc
    return validate_evidence_payload({"header": values[0], "records": values[1:]})


def _validate_archive(
    archive_path: Path, ledger_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # ``decision_ledger.relative_path`` is deliberately archive-owner local:
    # each independently resumable arm writes beneath the directory that owns
    # its ``complete_episode_evidence`` tree.  Aggregate completion gates scan
    # a broader root containing many such arms, so resolving the stored path
    # directly against that scan root drops the scenario/condition/seed prefix.
    # Keep the broad root for containment and portable manifest paths, but use
    # the archive's canonical owner for the external-ledger lookup.
    scan_root = ledger_root.resolve()
    resolved_archive = archive_path.resolve()
    owner_root = archive_path.parent.parent.parent.resolve()
    if archive_path.parent.parent.name != "complete_episode_evidence":
        raise ValueError(
            f"archive is outside the canonical evidence tree: {archive_path}"
        )
    try:
        resolved_archive.relative_to(scan_root)
        owner_root.relative_to(scan_root)
    except ValueError as exc:
        raise ValueError(
            f"episode archive escapes its ledger scan root: {archive_path}"
        ) from exc

    payload, archive_receipt = read_gzip_json(archive_path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported episode evidence schema: {archive_path}")
    identity = payload.get("identity")
    rng = payload.get("rng")
    learner = payload.get("learner_state")
    runtime = payload.get("runtime")
    episode = payload.get("episode_result")
    ledger = payload.get("decision_ledger")
    if not all(isinstance(value, dict) for value in (
        identity, rng, learner, runtime, episode, ledger,
    )):
        raise ValueError(f"episode archive lacks required objects: {archive_path}")

    seed = identity.get("benchmark_seed")
    scenario = identity.get("scenario")
    mode = identity.get("mode")
    episode_index = identity.get("episode_index")
    if (
        not isinstance(seed, int)
        or not isinstance(scenario, str)
        or not scenario
        or not isinstance(mode, str)
        or not mode
        or episode_index not in {0, 1, 2, 3}
    ):
        raise ValueError(f"invalid episode identity: {archive_path}")
    if archive_path.name != f"episode_{episode_index}.json.gz":
        raise ValueError(f"archive filename and episode index differ: {archive_path}")
    if archive_path.parent.name != f"{mode}__{scenario}":
        raise ValueError(f"archive directory and identity differ: {archive_path}")

    for stream in ("scenario", "environment", "policy"):
        expected = _stream_seed(seed, scenario, episode_index, stream)
        if rng.get(f"{stream}_seed") != expected:
            raise ValueError(f"{stream} seed mismatch: {archive_path}")
    expected_environment_id = (
        f"seed={seed};scenario={scenario};episode={episode_index};stream=environment"
    )
    expected_policy_id = (
        f"seed={seed};scenario={scenario};episode={episode_index};stream=policy"
    )
    if (
        rng.get("environment_stream_id") != expected_environment_id
        or rng.get("stochastic_stream_id") != expected_environment_id
        or rng.get("policy_stream_id") != expected_policy_id
    ):
        raise ValueError(f"stream identity mismatch: {archive_path}")

    if canonical_json_sha256(payload.get("input_frame")) != payload.get(
        "input_frame_sha256"
    ):
        raise ValueError(f"input-frame hash mismatch: {archive_path}")
    if canonical_json_sha256(learner.get("before")) != learner.get(
        "before_sha256"
    ):
        raise ValueError(f"pre-episode learner-state hash mismatch: {archive_path}")
    if canonical_json_sha256(learner.get("after")) != learner.get(
        "after_sha256"
    ):
        raise ValueError(f"post-episode learner-state hash mismatch: {archive_path}")
    if canonical_json_sha256(
        _learner_continuation_payload(learner.get("before") or {})
    ) != learner.get("continuation_before_sha256"):
        raise ValueError(f"pre-episode continuation hash mismatch: {archive_path}")
    if canonical_json_sha256(
        _learner_continuation_payload(learner.get("after") or {})
    ) != learner.get("continuation_after_sha256"):
        raise ValueError(f"post-episode continuation hash mismatch: {archive_path}")
    if canonical_json_sha256(payload.get("protocol_records")) != payload.get(
        "protocol_records_sha256"
    ):
        raise ValueError(f"protocol-record hash mismatch: {archive_path}")

    if any(
        episode.get(field) != identity.get(field)
        for field in ("benchmark_seed", "episode_index", "episode_phase", "learning_enabled")
    ):
        raise ValueError(f"episode result and archive identity differ: {archive_path}")
    if (
        episode.get("learner_state_before_sha256") != learner.get("before_sha256")
        or episode.get("learner_state_after_sha256") != learner.get("after_sha256")
        or episode.get("learner_continuation_before_sha256")
        != learner.get("continuation_before_sha256")
        or episode.get("learner_continuation_after_sha256")
        != learner.get("continuation_after_sha256")
    ):
        raise ValueError(f"episode result and learner evidence differ: {archive_path}")
    if (
        not isinstance(runtime.get("wall_seconds"), (int, float))
        or float(runtime["wall_seconds"]) < 0
        or not isinstance(runtime.get("process_cpu_seconds"), (int, float))
        or float(runtime["process_cpu_seconds"]) < 0
    ):
        raise ValueError(f"invalid episode runtime receipt: {archive_path}")

    ledger_path = _safe_relative(owner_root, ledger.get("relative_path"))
    observed_sha, observed_bytes = _literal_binding(ledger_path)
    if (
        ledger.get("literal_sha256") != observed_sha
        or ledger.get("literal_bytes") != observed_bytes
    ):
        raise ValueError(f"external decision-ledger binding mismatch: {archive_path}")
    storage = ledger.get("storage")
    if storage == "deterministic_gzip_jsonl":
        ledger_payload = read_jsonl_gzip(
            ledger_path, expected_literal_sha256=observed_sha,
        )
    elif storage == "plain_jsonl":
        ledger_payload = _load_plain_ledger(ledger_path)
    else:
        raise ValueError(f"unknown ledger storage format: {archive_path}")
    header = ledger_payload["header"]
    metadata = header.get("metadata") or {}
    if (
        header.get("merkle_root") != ledger.get("merkle_root")
        or header.get("n_records") != ledger.get("n_records")
        or metadata.get("benchmark_seed") != seed
        or metadata.get("scenario") != scenario
        or metadata.get("mode") != mode
        or metadata.get("episode_index") != episode_index
        or metadata.get("learner_state_before_sha256") != learner.get("before_sha256")
        or metadata.get("learner_state_after_sha256") != learner.get("after_sha256")
        or metadata.get("learner_continuation_before_sha256")
        != learner.get("continuation_before_sha256")
        or metadata.get("learner_continuation_after_sha256")
        != learner.get("continuation_after_sha256")
    ):
        raise ValueError(f"decision-ledger header binding mismatch: {archive_path}")

    relative_archive = resolved_archive.relative_to(scan_root).as_posix()
    record = {
        "archive": relative_archive,
        "archive_literal_sha256": archive_receipt.literal_sha256,
        "archive_literal_bytes": archive_receipt.literal_bytes,
        "archive_canonical_sha256": archive_receipt.canonical_json_sha256,
        "archive_canonical_bytes": archive_receipt.canonical_json_bytes,
        "ledger": ledger_path.resolve().relative_to(scan_root).as_posix(),
        "ledger_literal_sha256": observed_sha,
        "ledger_literal_bytes": observed_bytes,
        "ledger_merkle_root": header["merkle_root"],
        "ledger_n_records": header["n_records"],
        "identity": identity,
        "learner_state_before_sha256": learner["before_sha256"],
        "learner_state_after_sha256": learner["after_sha256"],
        "learner_continuation_before_sha256": learner[
            "continuation_before_sha256"
        ],
        "learner_continuation_after_sha256": learner[
            "continuation_after_sha256"
        ],
    }
    return payload, record


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json_bytes(payload))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_complete_evidence(
    ledger_root: Path,
    *,
    expected_groups: int,
    expected_episodes: int,
    expected_adaptation_ledgers: int,
    expected_final_ledgers: int,
    manifest_path: Path | None,
) -> dict[str, Any]:
    root = ledger_root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"ledger root is not a regular directory: {root}")
    archive_paths = sorted(root.rglob("complete_episode_evidence/*/episode_*.json.gz"))
    if len(archive_paths) != expected_episodes:
        raise ValueError(
            f"episode archive count is {len(archive_paths)}, expected {expected_episodes}"
        )

    records = []
    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    ledgers: set[str] = set()
    adaptation_ledgers = 0
    final_ledgers = 0
    for path in archive_paths:
        payload, record = _validate_archive(path, root)
        identity = payload["identity"]
        # The directory immediately owning complete_episode_evidence is an arm
        # root (one core seed, one H3 stressor/seed, or one structural task arm).
        owner = path.parent.parent.parent.resolve().relative_to(root).as_posix()
        group_key = (owner, identity["scenario"], identity["mode"])
        grouped[group_key].append((identity["episode_index"], payload))
        if record["ledger"] in ledgers:
            raise ValueError(f"decision ledger is bound by multiple archives: {record['ledger']}")
        ledgers.add(record["ledger"])
        if payload["decision_ledger"]["storage"] == "deterministic_gzip_jsonl":
            adaptation_ledgers += 1
        else:
            final_ledgers += 1
        records.append(record)

    if len(grouped) != expected_groups:
        raise ValueError(f"episode group count is {len(grouped)}, expected {expected_groups}")
    if adaptation_ledgers != expected_adaptation_ledgers:
        raise ValueError(
            f"adaptation ledger count is {adaptation_ledgers}, "
            f"expected {expected_adaptation_ledgers}"
        )
    if final_ledgers != expected_final_ledgers:
        raise ValueError(
            f"final ledger count is {final_ledgers}, expected {expected_final_ledgers}"
        )

    sequence_records = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        indices = [item[0] for item in values]
        if indices not in ([3], [0, 1, 2, 3]):
            raise ValueError(f"invalid episode sequence {indices} for {key}")
        for (_left_index, left), (_right_index, right) in zip(
            values, values[1:], strict=False,
        ):
            if (
                left["learner_state"]["continuation_after_sha256"]
                != right["learner_state"]["continuation_before_sha256"]
            ):
                raise ValueError(f"learner-state continuity failure for {key}")
        sequence_records.append({
            "owner": key[0],
            "scenario": key[1],
            "mode": key[2],
            "episode_indices": indices,
            "initial_learner_state_sha256": values[0][1]["learner_state"][
                "before_sha256"
            ],
            "final_learner_state_sha256": values[-1][1]["learner_state"][
                "after_sha256"
            ],
        })

    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "ledger_root": root.name,
        "counts": {
            "episode_groups": len(grouped),
            "executed_episode_archives": len(records),
            "adaptation_episode_ledgers": adaptation_ledgers,
            "final_episode_ledgers": final_ledgers,
            "decision_records": sum(int(record["ledger_n_records"]) for record in records),
        },
        "sequences": sequence_records,
        "artifacts": records,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    if manifest_path is not None:
        _atomic_json(manifest_path, manifest)
        reread = json.loads(manifest_path.read_text(encoding="utf-8"))
        claimed = reread.pop("manifest_sha256", None)
        if claimed != canonical_json_sha256(reread):
            raise RuntimeError("complete episode evidence manifest failed readback")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-adaptation-ledgers", type=int, required=True)
    parser.add_argument("--expected-final-ledgers", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_complete_evidence(
        args.ledger_root,
        expected_groups=args.expected_groups,
        expected_episodes=args.expected_episodes,
        expected_adaptation_ledgers=args.expected_adaptation_ledgers,
        expected_final_ledgers=args.expected_final_ledgers,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
