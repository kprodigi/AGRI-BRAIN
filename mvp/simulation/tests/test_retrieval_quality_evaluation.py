"""Tests for fail-closed independent retrieval-quality evidence handling."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from mvp.simulation.retrieval_evaluation.analyze_retrieval_quality import analyze_bundle
from mvp.simulation.retrieval_evaluation.validate_retrieval_evaluation import (
    EvaluationValidationError,
    observed_input_hashes,
    validate_bundle,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata() -> dict:
    return {
        "schema_version": "agribrain-independent-retrieval-evaluation-v1",
        "study_id": "unit-test-study",
        "evidence_scope": "retrieval_ranking_quality_only",
        "downstream_ari_in_scope": False,
        "comparison": {
            "control_system_id": "agribrain_standard_rag",
            "control_retrieval_label": "Standard RAG",
            "candidate_system_id": "agribrain",
            "candidate_retrieval_label": "piR",
            "contrast_direction": "piR_minus_Standard_RAG",
        },
        "systems": [
            {
                "system_id": "agribrain_standard_rag",
                "retrieval_variant": "standard_rag",
                "run_id": "standard-run",
                "code_revision": "test-revision",
                "configuration_sha256": "1" * 64,
            },
            {
                "system_id": "agribrain",
                "retrieval_variant": "pirag",
                "run_id": "pirag-run",
                "code_revision": "test-revision",
                "configuration_sha256": "2" * 64,
            },
        ],
        "study_design": {
            "query_set_fixed_before_judgment": True,
            "document_pool_fixed_before_judgment": True,
            "runs_frozen_before_judgment": True,
            "pooled_documents_deduplicated": True,
            "system_labels_masked": True,
            "presentation_order_randomized": True,
            "run_depth": 3,
        },
        "assessment": {
            "judgment_source": "independent_human",
            "assessor_cohort_id": "test-cohort",
            "assessor_count": 2,
            "assessor_ids_are_pseudonymous": True,
            "independent_of_system_development": True,
            "independence_basis": "unit-test attestation",
            "blinded_to_system_identity": True,
            "blinding_protocol": "pooled documents shown under randomized opaque IDs",
            "judgment_scale": "ordinal_0_2",
            "provenance_record_id": "test-provenance",
        },
        "analysis": {
            "cutoffs": [1, 3],
            "primary_metric": "ndcg",
            "primary_cutoff": 3,
            "query_aggregation": "mean_assessor_metric_then_mean_query",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 17,
        },
        "input_sha256": {},
    }


def _rehash(root: Path) -> None:
    path = root / "evaluation_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["input_sha256"] = observed_input_hashes(root)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _valid_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "study"
    root.mkdir()
    queries = [(f"Q{index}", f"fixed query {index}") for index in range(1, 4)]
    _write_csv(
        root / "query_set.csv",
        ["query_id", "query_text", "query_text_sha256", "query_stratum", "source_record_id"],
        [
            [query_id, text, _sha_text(text), "cold-chain", f"query-source-{query_id}"]
            for query_id, text in queries
        ],
    )

    document_rows = []
    run_rows = []
    judgment_rows = []
    for query_id, _text in queries:
        docs = [f"{query_id}_D{index}" for index in range(1, 4)]
        for index, doc_id in enumerate(docs, start=1):
            document_rows.append(
                [doc_id, hashlib.sha256(f"content-{doc_id}".encode()).hexdigest(), f"source-{doc_id}", f"document {doc_id}"]
            )
        for rank, doc_id in enumerate(reversed(docs), start=1):
            run_rows.append(
                [query_id, "agribrain_standard_rag", rank, doc_id, 1.0 / rank, f"std-{query_id}-{rank}"]
            )
        for rank, doc_id in enumerate(docs, start=1):
            run_rows.append(
                [query_id, "agribrain", rank, doc_id, 1.0 / rank, f"pirag-{query_id}-{rank}"]
            )
        for assessor_id in ("assessor_A01", "assessor_A02"):
            for doc_id, grade in zip(docs, (2, 1, 0)):
                judgment_rows.append(
                    [query_id, doc_id, assessor_id, grade, f"judge-{assessor_id}-{query_id}-{doc_id}"]
                )

    _write_csv(
        root / "document_catalog.csv",
        ["doc_id", "content_sha256", "source_record_id", "descriptor"],
        document_rows,
    )
    _write_csv(
        root / "retrieval_runs.csv",
        ["query_id", "system_id", "rank", "doc_id", "score", "run_record_id"],
        run_rows,
    )
    _write_csv(
        root / "assessors.csv",
        [
            "assessor_id",
            "expertise_category",
            "independent_of_system_development",
            "blinded_to_system_identity",
            "provenance_record_id",
            "judgment_completed_at_utc",
        ],
        [
            ["assessor_A01", "cold-chain domain", "true", "true", "attestation-A01", "2026-08-28T12:00:00Z"],
            ["assessor_A02", "information retrieval", "true", "true", "attestation-A02", "2026-08-28T12:30:00Z"],
        ],
    )
    _write_csv(
        root / "relevance_judgments.csv",
        ["query_id", "doc_id", "assessor_id", "relevance", "judgment_record_id"],
        judgment_rows,
    )
    (root / "evaluation_metadata.json").write_text(
        json.dumps(_metadata(), indent=2) + "\n", encoding="utf-8"
    )
    _rehash(root)
    return root


def test_valid_independent_blinded_bundle_and_paired_analysis(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path)
    validated = validate_bundle(root)
    result = analyze_bundle(validated)

    assert result["evidence_scope"] == "retrieval_ranking_quality_only"
    assert result["downstream_ari_in_scope"] is False
    primary = result["primary_inference"]
    assert primary["metric"] == "ndcg@3"
    assert primary["interval_excludes_zero_in_declared_direction"] is True
    assert primary["not_evidence_for"] == "downstream ARI or operational performance"
    summary = result["metric_summary"]["ndcg@3"]
    assert summary["n_paired_queries"] == 3
    assert summary["pirag_mean"] > summary["standard_rag_mean"]
    assert summary["paired_percentile_bootstrap_interval"][0] > 0


def test_missing_judgment_fails_closed(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "relevance_judgments.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    _rehash(root)
    with pytest.raises(EvaluationValidationError, match="missing 1 explicit pooled judgments"):
        validate_bundle(root)


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("independent_of_system_development", "independent_of_system_development must be true"),
        ("blinded_to_system_identity", "blinded_to_system_identity must be true"),
    ],
)
def test_nonindependent_or_unblinded_assessor_fails_closed(
    tmp_path: Path, column: str, message: str
) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "assessors.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][column] = "false"
    _write_csv(path, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
    _rehash(root)
    with pytest.raises(EvaluationValidationError, match=message):
        validate_bundle(root)


def test_unblinded_study_attestation_fails_closed(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "evaluation_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["study_design"]["system_labels_masked"] = False
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationValidationError, match="system_labels_masked must be"):
        validate_bundle(root)


@pytest.mark.parametrize(
    "field",
    ["independent_of_system_development", "blinded_to_system_identity"],
)
def test_false_cohort_independence_or_blinding_attestation_fails_closed(
    tmp_path: Path, field: str
) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "evaluation_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["assessment"][field] = False
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationValidationError, match=field + " must be"):
        validate_bundle(root)


def test_invalid_ordinal_relevance_fails_closed(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "relevance_judgments.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["relevance"] = "3"
    _write_csv(path, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
    _rehash(root)
    with pytest.raises(EvaluationValidationError, match="relevance must be one of"):
        validate_bundle(root)


def test_changed_query_text_without_fixed_text_hash_fails_closed(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "query_set.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["query_text"] = "changed after the query-set hash was declared"
    _write_csv(path, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
    _rehash(root)
    with pytest.raises(EvaluationValidationError, match="query_text_sha256 does not match"):
        validate_bundle(root)


def test_declared_input_hash_rejects_post_freeze_edit(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "query_set.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["query_stratum"] = "edited-after-hash-freeze"
    _write_csv(path, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
    with pytest.raises(EvaluationValidationError, match="input hash mismatch for query_set.csv"):
        validate_bundle(root)


def test_run_cannot_reference_an_unfixed_document_id(tmp_path: Path) -> None:
    root = _valid_bundle(tmp_path)
    path = root / "retrieval_runs.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["doc_id"] = "UNDECLARED_DOC"
    _write_csv(path, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
    _rehash(root)
    with pytest.raises(EvaluationValidationError, match="references unknown doc_id"):
        validate_bundle(root)


def test_template_is_deliberately_ineligible() -> None:
    template = Path(__file__).parents[1] / "retrieval_evaluation" / "templates"
    with pytest.raises(EvaluationValidationError):
        validate_bundle(template)
