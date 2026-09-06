from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hpc.validate_complete_episode_evidence import validate_complete_evidence
from mvp.simulation import generate_results as gr
from mvp.simulation.benchmarks.episode_archive import read_gzip_json


def _run_static(frame: pd.DataFrame, ledger_root: Path) -> dict:
    seed = 42
    episode_index = 3
    policy_seed = gr._stream_seed(seed, "baseline", episode_index, "policy")
    with gr.decision_ledger_scope(ledger_root):
        return gr.run_episode(
            frame,
            "static",
            gr.Policy(),
            np.random.default_rng(policy_seed),
            "baseline",
            stoch=gr._STOCH_DISABLED,
            seed=seed,
            benchmark_seed=seed,
            episode_index=episode_index,
            environment_stream_id=gr._stream_id(
                seed, "baseline", episode_index, "environment",
            ),
            policy_stream_id=gr._stream_id(
                seed, "baseline", episode_index, "policy",
            ),
            stochastic_stream_id=gr._stream_id(
                seed, "baseline", episode_index, "environment",
            ),
            learning_enabled=False,
        )


def test_complete_episode_is_resumed_byte_for_byte_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FULL_EVIDENCE_CAPTURE", "1")
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("AGRIBRAIN_SOURCE_TREE_SHA256", "b" * 64)
    monkeypatch.setenv("RUN_TAG", "resume-test")
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(12)
    ledger_root = tmp_path / "ledger"

    first = _run_static(frame, ledger_root)
    archive = Path(first["episode_evidence_path"])
    ledger = Path(first["decision_ledger_path"])
    archive_bytes = archive.read_bytes()
    ledger_bytes = ledger.read_bytes()

    monkeypatch.setattr(
        gr,
        "_run_episode_impl",
        lambda *_args, **_kwargs: pytest.fail(
            "a completed episode was executed again"
        ),
    )
    resumed = _run_static(frame.copy(deep=True), ledger_root)
    assert resumed["_resumed_from_complete_episode_evidence"] is True
    assert resumed["ari"] == first["ari"]
    assert archive.read_bytes() == archive_bytes
    assert ledger.read_bytes() == ledger_bytes

    changed = frame.copy(deep=True)
    changed.loc[changed.index[0], "demand_units"] += 1.0
    with pytest.raises(RuntimeError, match="input frame differs"):
        _run_static(changed, ledger_root)


def _run_learned_panel(frame: pd.DataFrame, ledger_root: Path) -> list[dict]:
    seed = 42
    learner_cache: dict = {}
    episodes = []
    with gr.decision_ledger_scope(ledger_root):
        for episode_index in range(4):
            policy_seed = gr._stream_seed(
                seed, "baseline", episode_index, "policy",
            )
            episodes.append(gr.run_episode(
                frame,
                "no_context",
                gr.Policy(),
                np.random.default_rng(policy_seed),
                "baseline",
                stoch=gr._STOCH_DISABLED,
                seed=seed,
                benchmark_seed=seed,
                episode_index=episode_index,
                environment_stream_id=gr._stream_id(
                    seed, "baseline", episode_index, "environment",
                ),
                policy_stream_id=gr._stream_id(
                    seed, "baseline", episode_index, "policy",
                ),
                stochastic_stream_id=gr._stream_id(
                    seed, "baseline", episode_index, "environment",
                ),
                learner_state_cache=learner_cache,
                learning_enabled=episode_index < 3,
            ))
    return episodes


def test_learned_resume_restores_all_three_adaptation_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FULL_EVIDENCE_CAPTURE", "1")
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("AGRIBRAIN_SOURCE_TREE_SHA256", "b" * 64)
    monkeypatch.setenv("RUN_TAG", "learned-resume-test")
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(8)
    ledger_root = tmp_path / "learned-ledger"
    first = _run_learned_panel(frame, ledger_root)
    manifest = validate_complete_evidence(
        ledger_root,
        expected_groups=1,
        expected_episodes=4,
        expected_adaptation_ledgers=3,
        expected_final_ledgers=1,
        manifest_path=ledger_root / "complete_episode_evidence_manifest.json",
    )
    assert manifest["counts"]["executed_episode_archives"] == 4

    monkeypatch.setattr(
        gr,
        "_run_episode_impl",
        lambda *_args, **_kwargs: pytest.fail(
            "a completed learned episode was executed again"
        ),
    )
    resumed = _run_learned_panel(frame.copy(deep=True), ledger_root)
    assert [episode["ari"] for episode in resumed] == [
        episode["ari"] for episode in first
    ]
    assert all(
        episode["_resumed_from_complete_episode_evidence"] is True
        for episode in resumed
    )


def test_full_context_archive_serializes_channel_and_raw_trace_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FULL_EVIDENCE_CAPTURE", "1")
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("AGRIBRAIN_SOURCE_TREE_SHA256", "b" * 64)
    monkeypatch.setenv("RUN_TAG", "context-archive-test")
    seed = 42
    episode_index = 3
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(4)
    ledger_root = tmp_path / "context-ledger"
    with gr.decision_ledger_scope(ledger_root):
        episode = gr.run_episode(
            frame,
            "agribrain",
            gr.Policy(),
            np.random.default_rng(gr._stream_seed(
                seed, "baseline", episode_index, "policy",
            )),
            "baseline",
            stoch=gr._STOCH_DISABLED,
            seed=seed,
            benchmark_seed=seed,
            episode_index=episode_index,
            environment_stream_id=gr._stream_id(
                seed, "baseline", episode_index, "environment",
            ),
            policy_stream_id=gr._stream_id(
                seed, "baseline", episode_index, "policy",
            ),
            stochastic_stream_id=gr._stream_id(
                seed, "baseline", episode_index, "environment",
            ),
            learner_state_cache={},
            learning_enabled=False,
        )
    payload, _receipt = read_gzip_json(episode["episode_evidence_path"])
    assert payload["trace_exports"]["raw_decision_traces"]
    assert payload["protocol_records"]
    validate_complete_evidence(
        ledger_root,
        expected_groups=1,
        expected_episodes=1,
        expected_adaptation_ledgers=0,
        expected_final_ledgers=1,
        manifest_path=None,
    )


def test_complete_evidence_resolves_nested_structural_arm_ledger_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broad H3 scan root must retain each archive's nested arm owner."""

    monkeypatch.setenv("FULL_EVIDENCE_CAPTURE", "1")
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("AGRIBRAIN_SOURCE_TREE_SHA256", "b" * 64)
    monkeypatch.setenv("RUN_TAG", "nested-structural-owner-test")
    frame = pd.read_csv(gr.DATA_CSV, parse_dates=["timestamp"]).head(1)
    scan_root = tmp_path / "decision_ledgers"
    owner_relative = Path(
        "baseline/structural__point_000__sensor_noise/seed_42"
    )
    arm_root = scan_root / owner_relative

    episodes = _run_learned_panel(frame, arm_root)
    assert Path(episodes[0]["decision_ledger_path"]) == (
        arm_root
        / "adaptation_episode_ledgers/no_context__baseline/episode_0.jsonl.gz"
    )
    assert Path(episodes[-1]["decision_ledger_path"]) == (
        arm_root / "no_context__baseline.jsonl"
    )

    manifest = validate_complete_evidence(
        scan_root,
        expected_groups=1,
        expected_episodes=4,
        expected_adaptation_ledgers=3,
        expected_final_ledgers=1,
        manifest_path=None,
    )
    artifact = manifest["artifacts"][0]
    assert artifact["ledger"] == (
        owner_relative
        / "adaptation_episode_ledgers/no_context__baseline/episode_0.jsonl.gz"
    ).as_posix()
    assert artifact["archive"] == (
        owner_relative
        / "complete_episode_evidence/no_context__baseline/episode_0.json.gz"
    ).as_posix()
    assert manifest["sequences"][0]["owner"] == owner_relative.as_posix()
