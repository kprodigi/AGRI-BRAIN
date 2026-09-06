"""Semantic receipt and exact-figure publication gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from matplotlib import font_manager
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from hpc.validate_and_promote_figures import promote
from mvp.simulation.validation import validate_publication_artifacts as vpa
from mvp.simulation.validation.figure_artifacts import (
    EXPECTED_AGGREGATE_INPUTS,
    EXPECTED_FIGURE_FILES,
    EXPECTED_FIGURE_STEMS,
    EXPECTED_PANEL_GROUPS,
    EXPECTED_SEEDS,
    figure_records,
    validate_figure_directory,
)

_TEST_PDF_FONT = "AgriBrainTestSans"
pdfmetrics.registerFont(TTFont(
    _TEST_PDF_FONT,
    font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")),
))
from mvp.simulation.analysis.publication_figure_style import (
    publication_style_contract,
)


def _record(path):
    payload = path.read_bytes()
    return {
        "file": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _exact_inventory_manifest(*, include_receipt: bool) -> dict:
    commit = "e" * 40
    tag = "eeeeeee_20260828_120000"
    names = vpa._expected_manifest_paths(tag, include_receipt=include_receipt)
    return {
        "schema_version": 2,
        "git_commit": commit,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "dual_provenance": False,
        "git_dirty": False,
        "includes_raw_run_artifacts": True,
        "artifact_run_tag": tag,
        "artifact_count": len(names),
        "artifacts": [
            {"file": name, "bytes": 0, "sha256": "0" * 64}
            for name in sorted(names)
        ],
    }


def test_manifest_inventory_requires_every_raw_protocol_artifact() -> None:
    manifest = _exact_inventory_manifest(include_receipt=True)
    counts = vpa._validate_manifest_inventory(manifest, receipt_expected=True)
    assert counts["benchmark_seed_envelopes"] == 20
    assert counts["primary_retained_decision_ledgers"] == 1100
    assert counts["h3_retained_stressed_decision_ledgers"] == 500
    assert counts["raw_stress_task_files"] == 20

    manifest["artifacts"].pop()
    manifest["artifact_count"] -= 1
    with pytest.raises(ValueError, match="exact protocol inventory"):
        vpa._validate_manifest_inventory(manifest, receipt_expected=True)


def test_manifest_inventory_requires_commit_bound_run_tag() -> None:
    manifest = _exact_inventory_manifest(include_receipt=False)
    manifest["artifact_run_tag"] = "fffffff_20260828_120000"
    with pytest.raises(ValueError, match="commit-bound"):
        vpa._validate_manifest_inventory(manifest, receipt_expected=False)


def test_publication_environment_uses_repair_commit_only_in_recovery() -> None:
    fresh = _exact_inventory_manifest(include_receipt=False)
    assert vpa._publication_execution_commit(fresh) == fresh["git_commit"]

    recovery = dict(fresh)
    recovery.update({
        "publication_code_commit": "f" * 40,
        "dual_provenance": True,
    })
    assert vpa._publication_execution_commit(recovery) == "f" * 40


def test_semantic_receipt_is_bound_to_protocol_manifest_and_run(
    tmp_path, monkeypatch,
):
    repo = tmp_path / "repo"
    results = repo / "mvp" / "simulation" / "results"
    results.mkdir(parents=True)
    protocol = repo / "mvp" / "simulation" / "experiment_protocol.json"
    protocol.write_text('{"locked":true}\n', encoding="utf-8")
    artifact = results / "benchmark_summary.json"
    artifact.write_text('{"summary":{}}\n', encoding="utf-8")
    commit = "a" * 40
    manifest = {
        "schema_version": 2,
        "git_commit": commit,
        "simulation_source_commit": commit,
        "publication_code_commit": commit,
        "dual_provenance": False,
        "git_dirty": False,
        "artifact_run_tag": "aaaaaaa_20260828_120000",
        "artifacts": [_record(artifact)],
    }
    manifest_path = results / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(vpa, "RESULTS_DIR", results)
    monkeypatch.setattr(vpa, "REPO_ROOT", repo)
    inventory = {
        "top_level_artifacts_excluding_receipt": 38,
        "benchmark_seed_envelopes": 20,
        "primary_retained_decision_ledgers": 1100,
        "h3_retained_stressed_decision_ledgers": 500,
        "raw_stress_task_files": 20,
        "core_slurm_submission_receipts": 1,
    }
    monkeypatch.setattr(
        vpa,
        "_validate_manifest_inventory",
        lambda _manifest, *, receipt_expected, recovery_authorization=None: inventory,
    )

    vpa._write_publication_validation_receipt()
    receipt_path = results / vpa.VALIDATION_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["validation_status"] == "PASS"
    assert receipt["fresh_single_commit_run"] is True
    assert receipt["structural_sensitivity"] == {
        "included_in_core_receipt": False,
        "required_for_full_submission_evidence": True,
        "required_separate_receipt": "structural_sensitivity_archive_receipt.json",
    }

    manifest["artifacts"].append(_record(receipt_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    vpa._validate_publication_validation_receipt()

    receipt["protocol"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest["artifacts"][-1] = _record(receipt_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        vpa._validate_publication_validation_receipt()


def _write_figure_set(root: Path, *, commit: str, tag: str) -> None:
    for stem in EXPECTED_FIGURE_STEMS:
        Image.new("RGB", (1200, 900), color=(20, 40, 60)).save(
            root / f"{stem}.png", dpi=(800, 800), optimize=True,
        )
        pdf = canvas.Canvas(str(root / f"{stem}.pdf"), pagesize=(504, 360))
        pdf.setFont(_TEST_PDF_FONT, 12)
        pdf.drawString(36, 300, stem)
        pdf.line(36, 72, 468, 288)
        pdf.save()
    provenance = {
        "schema_version": 3,
        "source_commit": commit,
        "source_commit_semantics": "raw_input_simulation_commit",
        "simulation_source_commit": commit,
        "renderer_code_commit": commit,
        "dual_provenance": False,
        "run_tag": tag,
        "seed_panel": list(EXPECTED_SEEDS),
        "n_seed_envelopes_loaded": len(EXPECTED_SEEDS),
        "seed_input_artifacts": [
            {
                "file": f"benchmark_seeds/seed_{seed}.json",
                "seed": seed,
                "bytes": 1,
                "sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
            }
            for seed in EXPECTED_SEEDS
        ],
        "aggregate_input_artifacts": [
            {
                "file": name,
                "bytes": 1,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            for name in EXPECTED_AGGREGATE_INPUTS
        ],
        "render_input_isolated_snapshot": True,
        "illustrative_seed": 42,
        "publication_style": publication_style_contract(),
        "renderer_environment": {
            "matplotlib": "test",
            "numpy": "test",
            "pillow": "test",
            "resolved_font": {
                "file": "test-font.ttf",
                "bytes": 1,
                "sha256": "8" * 64,
                "resolved_family": "Test Sans",
                "resolved_path": "/test/fonts/test-font.ttf",
            },
        },
        "panels": {
            name: (
                {
                    "fields": list(EXPECTED_AGGREGATE_INPUTS),
                    "aggregation": "test",
                    "n_seeds": 20,
                }
                if name == "cross_scenario_and_secondary" else {}
            )
            for name in EXPECTED_PANEL_GROUPS
        },
        "rendered_artifacts": figure_records(root),
    }
    (root / "figure_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8",
    )


def test_figure_validation_rejects_low_resolution_png(tmp_path):
    commit = "f" * 40
    tag = "fffffff_20260828_120000"
    _write_figure_set(tmp_path, commit=commit, tag=tag)
    Image.new("RGB", (800, 800), color=(20, 40, 60)).save(
        tmp_path / "heatwave.png", dpi=(150, 150),
    )
    with pytest.raises(ValueError, match="short edge|DPI"):
        validate_figure_directory(
            tmp_path, source_commit=commit, run_tag=tag,
        )


def test_figure_validation_rejects_unbound_style_contract(tmp_path):
    commit = "9" * 40
    tag = "9999999_20260828_120000"
    _write_figure_set(tmp_path, commit=commit, tag=tag)
    path = tmp_path / "figure_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["publication_style"]["png"]["dpi"] = 150
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="publication style contract"):
        validate_figure_directory(
            tmp_path, source_commit=commit, run_tag=tag,
        )


def test_exact_figure_inventory_rejects_stale_extra_file(tmp_path, monkeypatch):
    commit = "a" * 40
    tag = "aaaaaaa_20260828_120000"
    _write_figure_set(tmp_path, commit=commit, tag=tag)
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({
        "simulation_source_commit": commit,
        "artifact_run_tag": tag,
    }), encoding="utf-8")
    monkeypatch.setattr(vpa, "RESULTS_DIR", tmp_path)
    vpa._validate_exact_figure_inventory()

    (tmp_path / "fig_stale.png").write_bytes((tmp_path / EXPECTED_FIGURE_FILES[0]).read_bytes())
    with pytest.raises(SystemExit):
        vpa._validate_exact_figure_inventory()


def test_exact_figure_inventory_rejects_decodable_byte_tamper(
    tmp_path, monkeypatch,
):
    commit = "b" * 40
    tag = "bbbbbbb_20260828_120000"
    _write_figure_set(tmp_path, commit=commit, tag=tag)
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({
        "simulation_source_commit": commit,
        "artifact_run_tag": tag,
    }), encoding="utf-8")
    monkeypatch.setattr(vpa, "RESULTS_DIR", tmp_path)
    # This remains a valid PNG but no longer matches the provenance digest.
    Image.new("RGB", (8, 8), color=(200, 40, 60)).save(tmp_path / "heatwave.png")
    with pytest.raises(SystemExit):
        vpa._validate_exact_figure_inventory()


def test_figure_promotion_replaces_only_a_fully_validated_set(tmp_path):
    commit = "c" * 40
    tag = "ccccccc_20260828_120000"
    staging = tmp_path / "staging"
    results = tmp_path / "results"
    staging.mkdir()
    results.mkdir()
    (results / "benchmark_summary.json").write_text("{}\n", encoding="utf-8")
    _write_figure_set(staging, commit=commit, tag=tag)

    promote(staging, results, source_commit=commit, run_tag=tag)

    assert (results / "benchmark_summary.json").read_text(encoding="utf-8") == "{}\n"
    assert {path.name for path in results.glob("*.png")} == {
        name for name in EXPECTED_FIGURE_FILES if name.endswith(".png")
    }
    assert json.loads((results / "figure_provenance.json").read_text())[
        "source_commit"
    ] == commit


def test_figure_promotion_rejects_unbound_staging_before_overwrite(tmp_path):
    commit = "d" * 40
    tag = "ddddddd_20260828_120000"
    staging = tmp_path / "staging"
    results = tmp_path / "results"
    staging.mkdir()
    results.mkdir()
    _write_figure_set(staging, commit=commit, tag=tag)
    sentinel = results / "heatwave.png"
    sentinel.write_bytes(b"existing canonical bytes")
    Image.new("RGB", (1200, 900), color=(1, 2, 3)).save(
        staging / "heatwave.png", dpi=(800, 800), optimize=True,
    )

    with pytest.raises(ValueError, match="hash-bind"):
        promote(staging, results, source_commit=commit, run_tag=tag)
    assert sentinel.read_bytes() == b"existing canonical bytes"


def test_semantic_validator_binds_recovery_renderer_commit() -> None:
    simulation = "a" * 40
    publication = "b" * 40
    manifest = {
        "git_commit": simulation,
        "simulation_source_commit": simulation,
        "publication_code_commit": publication,
    }
    provenance = {
        "schema_version": 3,
        "source_commit": simulation,
        "source_commit_semantics": "raw_input_simulation_commit",
        "simulation_source_commit": simulation,
        "renderer_code_commit": publication,
        "dual_provenance": True,
    }
    vpa._validate_figure_source_identity(provenance, manifest)
    provenance["renderer_code_commit"] = simulation
    with pytest.raises(SystemExit):
        vpa._validate_figure_source_identity(provenance, manifest)


def test_reaggregated_core_comparison_rejects_coherent_summary_edit(
    tmp_path, monkeypatch,
):
    canonical = tmp_path / "canonical"
    regenerated = tmp_path / "regenerated"
    canonical.mkdir()
    regenerated.mkdir()
    payloads = {
        "benchmark_summary.json": {"summary": {"baseline": {"ari": 0.5}}},
        "benchmark_significance.json": {"significance": {"baseline": {}}},
        "secondary_ablation_analysis.json": {
            "secondary_ablations": {"baseline": {}}
        },
    }
    for name, payload in payloads.items():
        encoded = json.dumps(payload, sort_keys=True)
        (canonical / name).write_text(encoded, encoding="utf-8")
        (regenerated / name).write_text(encoded, encoding="utf-8")
    for name in (
        "table1_summary.csv",
        "table2_ablation.csv",
        "secondary_ablation_analysis.csv",
        "h2_directional_evidence.csv",
    ):
        (canonical / name).write_bytes(b"header\r\nvalue\r\n")
        (regenerated / name).write_bytes(b"header\r\nvalue\r\n")
    monkeypatch.setattr(vpa, "RESULTS_DIR", canonical)
    vpa._compare_reaggregated_core_artifacts(regenerated)

    altered = payloads["benchmark_summary.json"]
    altered["summary"]["baseline"]["ari"] = 0.6
    (canonical / "benchmark_summary.json").write_text(
        json.dumps(altered, sort_keys=True), encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        vpa._compare_reaggregated_core_artifacts(regenerated)
