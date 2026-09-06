"""Regression tests for ledger-derived publication evidence scope metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mvp.simulation.benchmarks import aggregate_channel_attribution as aca
from mvp.simulation.analysis import explainability_metrics as em
from mvp.simulation.validation import validate_publication_artifacts as vpa
from mvp.simulation.analysis import export_paper_evidence as evidence_export
import hpc.publication_recovery_receipt as recovery_receipt
import hpc.validate_source_checkout as source_checkout


EPISODE_SCOPE = "final episode per scenario-mode-seed arm"
HISTORY_SCOPE = "earlier decisions in the same episode only"


def _identity_payload(
    *, simulation_commit: str, publication_commit: str, run_tag: str,
) -> dict:
    return {"_meta": {
        "git_commit": simulation_commit,
        "source_commit": simulation_commit,
        "simulation_source_commit": simulation_commit,
        "analysis_code_commit": publication_commit,
        "dual_provenance": simulation_commit != publication_commit,
        "run_tag": run_tag,
    }}


def test_paper_export_fresh_identity_validates_fresh_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    run_tag = "aaaaaaa_20260829_105800"
    captured: dict = {}
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", commit)
    monkeypatch.setenv("RUN_TAG", run_tag)
    for name in (
        "AGRIBRAIN_RECOVERY_RECEIPT", "AGRIBRAIN_SIMULATION_COMMIT",
        "AGRIBRAIN_PUBLICATION_CODE_COMMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        source_checkout,
        "validation_errors",
        lambda *, environ, **_kwargs: captured.update(environ=environ) or [],
    )
    payload = _identity_payload(
        simulation_commit=commit, publication_commit=commit, run_tag=run_tag,
    )
    assert evidence_export._publication_export_identity_errors(payload, payload) == []
    assert captured["environ"]["AGRIBRAIN_GIT_COMMIT"] == commit


def test_paper_export_recovery_validates_publication_checkout_but_stamps_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = "a" * 40
    publication = "b" * 40
    run_tag = "aaaaaaa_20260829_105800"
    captured: dict = {}
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", simulation)
    monkeypatch.setenv("AGRIBRAIN_SIMULATION_COMMIT", simulation)
    monkeypatch.setenv("AGRIBRAIN_PUBLICATION_CODE_COMMIT", publication)
    monkeypatch.setenv("AGRIBRAIN_RECOVERY_RECEIPT", "recovery.json")
    monkeypatch.setenv("CORE_SUBMISSION_RECEIPT", "original.json")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("RUN_TAG", run_tag)
    monkeypatch.setattr(
        recovery_receipt, "validate_recovery_receipt_file", lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        source_checkout,
        "validation_errors",
        lambda *, environ, **_kwargs: captured.update(environ=environ) or [],
    )
    payload = _identity_payload(
        simulation_commit=simulation,
        publication_commit=publication,
        run_tag=run_tag,
    )
    assert evidence_export._publication_export_identity_errors(payload, payload) == []
    assert captured["environ"]["AGRIBRAIN_GIT_COMMIT"] == publication

    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", publication)
    errors = evidence_export._publication_export_identity_errors(payload, payload)
    assert any("must retain the simulation commit" in error for error in errors)


def _write_instrumented_ledger(root: Path, seed: int = 1) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "_header": True,
        "merkle_root": "0" * 64,
        "n_records": 1,
        "metadata": {"mode": "agribrain", "scenario": "heatwave", "seed": seed},
    }
    record = {
        "base_logits": [1.0, 0.0, 0.0],
        "context_modifier": [-2.0, 2.0, 0.0],
        "slca_shaping": [0.0, 0.0, 0.0],
        "slca_amp": 0.0,
        "policy_temperature": 1.0,
        "modifier_mcp": [-2.0, 0.0, 0.0],
        "modifier_pirag": [0.0, 2.0, 0.0],
        "psi": [1.0, 0.0, 1.0, 0.0, 0.0],
        "governance_override": False,
        "reward": 0.5,
        "waste": 0.1,
        "slca": 0.7,
        "rho": 0.2,
    }
    path = seed_dir / "agribrain__heatwave.jsonl"
    path.write_text(
        json.dumps(header) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )


def _set_publication_identity(monkeypatch, run_tag: str = "scope_test") -> str:
    commit = aca._git_commit()
    assert commit is not None
    monkeypatch.setenv("AGRIBRAIN_GIT_COMMIT", commit)
    monkeypatch.setenv("RUN_TAG", run_tag)
    monkeypatch.setenv("ARTIFACT_RUN_TAG", run_tag)
    return commit


def test_channel_attribution_output_records_exact_evidence_scope(tmp_path, monkeypatch):
    root = tmp_path / "ledgers"
    _write_instrumented_ledger(root)
    commit = _set_publication_identity(monkeypatch)
    output = tmp_path / "channel_attribution_aggregate.json"

    aca.main([
        "--ledger-root", str(root),
        "--output", str(output),
        "--modes", "agribrain",
        "--scenarios", "heatwave",
    ])

    meta = json.loads(output.read_text(encoding="utf-8"))["_meta"]
    assert meta["source_commit"] == commit
    assert meta["ledger_root"] == root.as_posix()
    assert meta["seed_count"] == 1
    assert meta["run_tag"] == "scope_test"
    assert meta["episode_scope"] == EPISODE_SCOPE
    assert meta["decision_history_scope"] == HISTORY_SCOPE

    replay = tmp_path / "channel_attribution_replay.json"
    aca.main([
        "--ledger-root", str(root),
        "--output", str(replay),
        "--modes", "agribrain",
        "--scenarios", "heatwave",
    ])
    assert replay.read_bytes() == output.read_bytes()


def test_channel_complementarity_output_records_exact_evidence_scope(
    tmp_path, monkeypatch,
):
    from mvp.simulation import _h2_permutation_test as h2

    root = tmp_path / "ledgers"
    _write_instrumented_ledger(root)
    commit = _set_publication_identity(monkeypatch)
    output = tmp_path / "channel_complementarity_test.json"
    monkeypatch.setattr(h2, "B", 32)
    monkeypatch.setattr(h2, "rng", np.random.default_rng(20260605))

    assert h2.main(ledger_root=root, output=output) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["_meta"] == {
        "source_commit": commit,
        "ledger_root": root.as_posix(),
        "seed_count": 1,
        "run_tag": "scope_test",
        "episode_scope": EPISODE_SCOPE,
        "decision_history_scope": HISTORY_SCOPE,
        "source_seed_count": 1,
        "analysis_kind": "conditional_observed_state_feature_group_masking",
        "legacy_filename_notice": (
            "The configured artifact filename predates the current estimand label."
        ),
        "interpretation_limit": (
            "Retrieval results and guards are reused from the observed execution; "
            "the estimates cannot represent disabled communication channels."
        ),
    }
    replay = tmp_path / "channel_complementarity_replay.json"
    assert h2.main(ledger_root=root, output=replay) == 0
    assert replay.read_bytes() == output.read_bytes()


def test_channel_complementarity_cli_requires_run_scoped_paths(tmp_path):
    from mvp.simulation import _h2_permutation_test as h2

    with pytest.raises(SystemExit) as missing_both:
        h2.main([])
    assert missing_both.value.code == 2

    with pytest.raises(SystemExit) as missing_output:
        h2.main(["--ledger-root", str(tmp_path)])
    assert missing_output.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_explainability_seed_count_uses_canonical_seed_directories(
    tmp_path, monkeypatch,
):
    root = tmp_path / "decision_ledger_per_seed"
    for panel_seed, mode_rng_seed in ((42, 9001), (1337, 9002)):
        seed_dir = root / f"seed_{panel_seed}"
        seed_dir.mkdir(parents=True)
        header = {
            "_header": True,
            "merkle_root": "0" * 64,
            "n_records": 0,
            "metadata": {
                "mode": "agribrain",
                "scenario": "heatwave",
                "seed": mode_rng_seed,
            },
        }
        (seed_dir / "agribrain__heatwave.jsonl").write_text(
            json.dumps(header) + "\n", encoding="utf-8",
        )
    _set_publication_identity(monkeypatch)
    output = tmp_path / "explainability_metrics.json"

    assert em.main(["--ledger", str(root), "--output", str(output)]) == 0
    meta = json.loads(output.read_text(encoding="utf-8"))["_meta"]
    assert meta["seed_count"] == 2
    replay = tmp_path / "explainability_replay.json"
    assert em.main(["--ledger", str(root), "--output", str(replay)]) == 0
    assert replay.read_bytes() == output.read_bytes()


def _write_validator_fixture(results: Path, *, history_scope: str = HISTORY_SCOPE) -> None:
    commit = "b" * 40
    run_tag = "abc1234_20260819_120000"
    (results / "artifact_manifest.json").write_text(json.dumps({
        "git_commit": commit,
        "artifact_run_tag": run_tag,
    }), encoding="utf-8")
    meta = {
        "source_commit": commit,
        "ledger_root": (
            "mvp/simulation/results/decision_ledger_per_seed/"
            "abc1234_20260819_120000"
        ),
        "seed_count": 20,
        "run_tag": run_tag,
        "episode_scope": EPISODE_SCOPE,
        "decision_history_scope": history_scope,
    }
    for name in (
        "channel_attribution_aggregate.json",
        "channel_complementarity_test.json",
        "explainability_metrics.json",
    ):
        payload = {"_meta": meta}
        if name == "channel_attribution_aggregate.json":
            payload["_meta"] = {
                **meta, "n_seeds": 20, "seeds": list(range(20)),
            }
            payload["by_scenario_mode"] = {
                scenario: {"agribrain": {"n_seeds": 20}}
                for scenario in (
                    "heatwave", "overproduction", "cyber_outage",
                    "adaptive_pricing", "baseline",
                )
            }
        elif name == "channel_complementarity_test.json":
            payload["_meta"] = {**meta, "source_seed_count": 20}
            payload["n_seeds"] = 20
        elif name == "explainability_metrics.json":
            payload["threshold"] = 0.10
        (results / name).write_text(json.dumps(payload), encoding="utf-8")


def test_publication_validator_accepts_exact_evidence_scope(tmp_path, monkeypatch):
    _write_validator_fixture(tmp_path)
    monkeypatch.setattr(vpa, "RESULTS_DIR", tmp_path)
    vpa._validate_evidence_scope_metadata()


def test_publication_validator_rejects_cross_episode_history_claim(tmp_path, monkeypatch):
    _write_validator_fixture(tmp_path, history_scope="previous episode and current episode")
    monkeypatch.setattr(vpa, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        vpa._validate_evidence_scope_metadata()


def test_paper_evidence_export_unwraps_canonical_payload_and_carries_corrections(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(evidence_export, "RESULTS_DIR", tmp_path)
    summary = {
        "_meta": {
            "git_commit": "c" * 40,
            "n_seeds": 20,
            "seeds_loaded": list(range(20)),
            "bootstrap_alpha": 0.05,
            "n_boot": 10_000,
            "n_perm": 10_000,
            "std_ddof": 1,
            "bca_fallback_stats": {"bca_fallback_rate": 0.0},
        },
        "summary": {
            "baseline": {
                "agribrain": {
                    "ari": {
                        "mean": 0.8, "std": 0.01,
                        "ci_low": 0.78, "ci_high": 0.82,
                    },
                },
            },
        },
    }
    significance = {
        "_meta": {
            "primary_h1_family": "H1 family",
            "primary_h1_correction": "holm_bonferroni",
            "h2_directional_family": "H2 family (20 tests)",
            "h2_directional_correction": "holm_bonferroni",
            "h2_directional_canonical_field": (
                "p_value_adj_holm_h2_directional"
            ),
            "h2_global_support_rule": "all 20 cells",
            "h2_synergy_status": "exploratory",
            "channel_decomposition_family": "auxiliary subset",
            "channel_decomposition_correction": "holm_bonferroni",
            "secondary_correction": "by_fdr",
            "secondary_family_scope": "within scenario",
        },
        "primary_h1_holm_adjusted": {"baseline": 0.01},
        "primary_h1_supported_by_cell": {"baseline": True},
        "primary_h1_supported_all_cells": True,
        "h2_directional_holm_adjusted": {
            "baseline:mcp_only_vs_no_context": 0.04,
        },
        "h2_directional_supported_by_cell": {
            "baseline:mcp_only_vs_no_context": False,
        },
        "h2_directional_supported_all_cells": False,
        "channel_decomposition_holm_adjusted": {
            "baseline:mcp_only_vs_no_context": 0.02,
        },
        "significance": {"baseline": {}},
    }
    (tmp_path / "benchmark_summary.json").write_text(
        json.dumps(summary), encoding="utf-8",
    )
    (tmp_path / "benchmark_significance.json").write_text(
        json.dumps(significance), encoding="utf-8",
    )

    monkeypatch.setattr(
        evidence_export,
        "_publication_export_identity_errors",
        lambda _summary, _significance: [],
    )
    evidence_export.export_latex_benchmark_table()
    exported = json.loads(
        (tmp_path / "paper_benchmark_table.json").read_text(encoding="utf-8")
    )
    assert exported["benchmark"] == summary["summary"]
    assert exported["significance"] == significance["significance"]
    correction = exported["_meta"]["significance_correction_meta"]
    assert correction["primary_h1_correction"] == "holm_bonferroni"
    assert correction["primary_h1_holm_adjusted"] == {"baseline": 0.01}
    assert correction["primary_h1_supported_by_cell"] == {"baseline": True}
    assert correction["primary_h1_supported_all_cells"] is True
    assert correction["h2_directional_correction"] == "holm_bonferroni"
    assert correction["h2_directional_holm_adjusted"] == {
        "baseline:mcp_only_vs_no_context": 0.04,
    }
    assert correction["h2_directional_supported_all_cells"] is False
    assert correction["channel_decomposition_holm_adjusted"] == {
        "baseline:mcp_only_vs_no_context": 0.02,
    }

    monkeypatch.setattr(vpa, "RESULTS_DIR", tmp_path)
    vpa._validate_paper_benchmark_table()
    exported["significance"] = {"tampered": {}}
    (tmp_path / "paper_benchmark_table.json").write_text(
        json.dumps(exported), encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        vpa._validate_paper_benchmark_table()
