"""Fail-closed validation for an independent retrieval-quality evaluation.

The validator accepts no implicit qrels.  Every document returned by either
retrieval system must have an explicit human relevance judgment from every
declared assessor.  Independence and system-identity blinding are required at
both the study and assessor levels.

This module validates retrieval-ranking evidence only.  It neither reads nor
supports claims about the downstream Adaptive Resilience Index (ARI).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "agribrain-independent-retrieval-evaluation-v1"
CONTROL_SYSTEM_ID = "agribrain_standard_rag"
CANDIDATE_SYSTEM_ID = "agribrain"
EXPECTED_SYSTEMS = (CONTROL_SYSTEM_ID, CANDIDATE_SYSTEM_ID)
EVIDENCE_SCOPE = "retrieval_ranking_quality_only"

INPUT_FILES: Mapping[str, str] = {
    "query_set": "query_set.csv",
    "document_catalog": "document_catalog.csv",
    "retrieval_runs": "retrieval_runs.csv",
    "assessors": "assessors.csv",
    "relevance_judgments": "relevance_judgments.csv",
}

CSV_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "query_set": (
        "query_id",
        "query_text",
        "query_text_sha256",
        "query_stratum",
        "source_record_id",
    ),
    "document_catalog": (
        "doc_id",
        "content_sha256",
        "source_record_id",
        "descriptor",
    ),
    "retrieval_runs": (
        "query_id",
        "system_id",
        "rank",
        "doc_id",
        "score",
        "run_record_id",
    ),
    "assessors": (
        "assessor_id",
        "expertise_category",
        "independent_of_system_development",
        "blinded_to_system_identity",
        "provenance_record_id",
        "judgment_completed_at_utc",
    ),
    "relevance_judgments": (
        "query_id",
        "doc_id",
        "assessor_id",
        "relevance",
        "judgment_record_id",
    ),
}

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ASSESSOR_ID_PATTERN = re.compile(r"^assessor_[A-Za-z0-9_-]{1,64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationValidationError(ValueError):
    """Raised when evidence is not eligible for retrieval-quality analysis."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("retrieval evaluation is ineligible:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class ValidatedEvaluationBundle:
    """Parsed inputs returned only after every eligibility check succeeds."""

    bundle_dir: Path
    metadata: Dict[str, Any]
    query_rows: Tuple[Dict[str, str], ...]
    document_rows: Tuple[Dict[str, str], ...]
    run_rows: Tuple[Dict[str, str], ...]
    assessor_rows: Tuple[Dict[str, str], ...]
    judgment_rows: Tuple[Dict[str, str], ...]
    observed_sha256: Dict[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def observed_input_hashes(bundle_dir: Path | str) -> Dict[str, str]:
    """Return hashes for the five fixed input filenames.

    Missing files are represented as ``MISSING`` so this helper is safe to use
    while preparing a bundle.  It does not make the bundle eligible.
    """

    root = Path(bundle_dir)
    return {
        key: (_sha256_file(root / filename) if (root / filename).is_file() else "MISSING")
        for key, filename in INPUT_FILES.items()
    }


def _read_csv(
    path: Path,
    expected_columns: Sequence[str],
    label: str,
    errors: List[str],
) -> List[Dict[str, str]]:
    if not path.is_file():
        errors.append(f"missing required file: {path.name}")
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            actual = tuple(reader.fieldnames or ())
            if actual != tuple(expected_columns):
                errors.append(
                    f"{path.name} has columns {actual!r}; expected exactly "
                    f"{tuple(expected_columns)!r}"
                )
                return []
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"could not read {path.name}: {exc}")
        return []
    if not rows:
        errors.append(f"{label} is empty")
    for row_number, row in enumerate(rows, start=2):
        if any(value is None for value in row.values()):
            errors.append(f"{path.name}:{row_number} has a malformed field count")
    return rows


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_utc_timestamp(value: str) -> bool:
    if not value or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() is not None


def _require_true(container: Mapping[str, Any], key: str, path: str, errors: List[str]) -> None:
    if container.get(key) is not True:
        errors.append(f"{path}.{key} must be the JSON boolean true")


def _require_nonempty(
    container: Mapping[str, Any], key: str, path: str, errors: List[str]
) -> None:
    if not _is_nonempty_string(container.get(key)):
        errors.append(f"{path}.{key} must be a non-empty string")


def _require_exact_keys(
    container: Mapping[str, Any], expected: Iterable[str], path: str, errors: List[str]
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(container))
    unexpected = sorted(set(container) - expected_set)
    if missing:
        errors.append(f"{path} is missing required keys: {missing!r}")
    if unexpected:
        errors.append(f"{path} has unsupported keys: {unexpected!r}")


def _validate_metadata(metadata: Any, errors: List[str]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        errors.append("evaluation_metadata.json must contain one JSON object")
        return {}

    _require_exact_keys(
        metadata,
        (
            "schema_version",
            "study_id",
            "evidence_scope",
            "downstream_ari_in_scope",
            "comparison",
            "systems",
            "study_design",
            "assessment",
            "analysis",
            "input_sha256",
        ),
        "metadata",
        errors,
    )

    if metadata.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION!r}")
    _require_nonempty(metadata, "study_id", "metadata", errors)
    if metadata.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must equal {EVIDENCE_SCOPE!r}")
    if metadata.get("downstream_ari_in_scope") is not False:
        errors.append("downstream_ari_in_scope must be the JSON boolean false")

    comparison = metadata.get("comparison")
    if not isinstance(comparison, dict):
        errors.append("metadata.comparison must be an object")
        comparison = {}
    expected_comparison = {
        "control_system_id": CONTROL_SYSTEM_ID,
        "control_retrieval_label": "Standard RAG",
        "candidate_system_id": CANDIDATE_SYSTEM_ID,
        "candidate_retrieval_label": "piR",
        "contrast_direction": "piR_minus_Standard_RAG",
    }
    _require_exact_keys(
        comparison, expected_comparison, "metadata.comparison", errors
    )
    for key, expected in expected_comparison.items():
        if comparison.get(key) != expected:
            errors.append(f"metadata.comparison.{key} must equal {expected!r}")

    systems = metadata.get("systems")
    if not isinstance(systems, list):
        errors.append("metadata.systems must be a two-item list")
        systems = []
    seen_systems: set[str] = set()
    expected_variants = {
        CONTROL_SYSTEM_ID: "standard_rag",
        CANDIDATE_SYSTEM_ID: "pirag",
    }
    for index, system in enumerate(systems):
        path = f"metadata.systems[{index}]"
        if not isinstance(system, dict):
            errors.append(f"{path} must be an object")
            continue
        _require_exact_keys(
            system,
            (
                "system_id",
                "retrieval_variant",
                "run_id",
                "code_revision",
                "configuration_sha256",
            ),
            path,
            errors,
        )
        system_id = system.get("system_id")
        if system_id not in expected_variants:
            errors.append(f"{path}.system_id is not one of {EXPECTED_SYSTEMS!r}")
        elif system_id in seen_systems:
            errors.append(f"duplicate system definition for {system_id!r}")
        else:
            seen_systems.add(system_id)
            if system.get("retrieval_variant") != expected_variants[system_id]:
                errors.append(
                    f"{path}.retrieval_variant must equal "
                    f"{expected_variants[system_id]!r}"
                )
        for key in ("run_id", "code_revision"):
            _require_nonempty(system, key, path, errors)
        config_hash = system.get("configuration_sha256")
        if not isinstance(config_hash, str) or not _SHA256_PATTERN.fullmatch(config_hash):
            errors.append(f"{path}.configuration_sha256 must be 64 lowercase hex characters")
    if seen_systems != set(EXPECTED_SYSTEMS):
        errors.append(f"metadata.systems must define exactly {EXPECTED_SYSTEMS!r}")

    design = metadata.get("study_design")
    if not isinstance(design, dict):
        errors.append("metadata.study_design must be an object")
        design = {}
    _require_exact_keys(
        design,
        (
            "query_set_fixed_before_judgment",
            "document_pool_fixed_before_judgment",
            "runs_frozen_before_judgment",
            "pooled_documents_deduplicated",
            "system_labels_masked",
            "presentation_order_randomized",
            "run_depth",
        ),
        "metadata.study_design",
        errors,
    )
    for key in (
        "query_set_fixed_before_judgment",
        "document_pool_fixed_before_judgment",
        "runs_frozen_before_judgment",
        "pooled_documents_deduplicated",
        "system_labels_masked",
        "presentation_order_randomized",
    ):
        _require_true(design, key, "metadata.study_design", errors)
    run_depth = design.get("run_depth")
    if not isinstance(run_depth, int) or isinstance(run_depth, bool) or run_depth < 1:
        errors.append("metadata.study_design.run_depth must be a positive integer")

    assessment = metadata.get("assessment")
    if not isinstance(assessment, dict):
        errors.append("metadata.assessment must be an object")
        assessment = {}
    _require_exact_keys(
        assessment,
        (
            "judgment_source",
            "assessor_cohort_id",
            "assessor_count",
            "assessor_ids_are_pseudonymous",
            "independent_of_system_development",
            "independence_basis",
            "blinded_to_system_identity",
            "blinding_protocol",
            "judgment_scale",
            "provenance_record_id",
        ),
        "metadata.assessment",
        errors,
    )
    if assessment.get("judgment_source") != "independent_human":
        errors.append("metadata.assessment.judgment_source must equal 'independent_human'")
    for key in (
        "assessor_ids_are_pseudonymous",
        "independent_of_system_development",
        "blinded_to_system_identity",
    ):
        _require_true(assessment, key, "metadata.assessment", errors)
    for key in (
        "assessor_cohort_id",
        "independence_basis",
        "blinding_protocol",
        "provenance_record_id",
    ):
        _require_nonempty(assessment, key, "metadata.assessment", errors)
    assessor_count = assessment.get("assessor_count")
    if not isinstance(assessor_count, int) or isinstance(assessor_count, bool) or assessor_count < 1:
        errors.append("metadata.assessment.assessor_count must be a positive integer")
    scale = assessment.get("judgment_scale")
    if scale not in {"binary_0_1", "ordinal_0_2"}:
        errors.append(
            "metadata.assessment.judgment_scale must be 'binary_0_1' or 'ordinal_0_2'"
        )

    analysis = metadata.get("analysis")
    if not isinstance(analysis, dict):
        errors.append("metadata.analysis must be an object")
        analysis = {}
    _require_exact_keys(
        analysis,
        (
            "cutoffs",
            "primary_metric",
            "primary_cutoff",
            "query_aggregation",
            "bootstrap_confidence_level",
            "bootstrap_resamples",
            "bootstrap_seed",
        ),
        "metadata.analysis",
        errors,
    )
    cutoffs = analysis.get("cutoffs")
    if (
        not isinstance(cutoffs, list)
        or not cutoffs
        or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in cutoffs)
        or cutoffs != sorted(set(cutoffs))
    ):
        errors.append("metadata.analysis.cutoffs must be unique positive integers in ascending order")
        cutoffs = []
    if isinstance(run_depth, int) and cutoffs and max(cutoffs) > run_depth:
        errors.append("metadata.analysis.cutoffs cannot exceed study_design.run_depth")
    if analysis.get("primary_metric") != "ndcg":
        errors.append("metadata.analysis.primary_metric must equal 'ndcg'")
    if analysis.get("primary_cutoff") not in cutoffs:
        errors.append("metadata.analysis.primary_cutoff must be listed in analysis.cutoffs")
    if analysis.get("query_aggregation") != "mean_assessor_metric_then_mean_query":
        errors.append(
            "metadata.analysis.query_aggregation must equal "
            "'mean_assessor_metric_then_mean_query'"
        )
    confidence = analysis.get("bootstrap_confidence_level")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.8 <= float(confidence) < 1.0
    ):
        errors.append("metadata.analysis.bootstrap_confidence_level must be in [0.8, 1.0)")
    resamples = analysis.get("bootstrap_resamples")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1000:
        errors.append("metadata.analysis.bootstrap_resamples must be an integer >= 1000")
    seed = analysis.get("bootstrap_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("metadata.analysis.bootstrap_seed must be a non-negative integer")

    input_hashes = metadata.get("input_sha256")
    if not isinstance(input_hashes, dict):
        errors.append("metadata.input_sha256 must be an object")
    elif set(input_hashes) != set(INPUT_FILES):
        errors.append(f"metadata.input_sha256 must have exactly these keys: {tuple(INPUT_FILES)!r}")
    return metadata


def _validate_identifiers(
    rows: Iterable[Dict[str, str]],
    field: str,
    filename: str,
    errors: List[str],
) -> None:
    for row_number, row in enumerate(rows, start=2):
        value = row.get(field, "")
        if not _ID_PATTERN.fullmatch(value):
            errors.append(f"{filename}:{row_number} has invalid {field} {value!r}")


def validate_bundle(bundle_dir: Path | str) -> ValidatedEvaluationBundle:
    """Validate and parse a complete independent-human evaluation bundle.

    Any missing, incomplete, non-independent, unblinded, unjudged, or hash-
    inconsistent input raises :class:`EvaluationValidationError`.
    """

    root = Path(bundle_dir).resolve()
    errors: List[str] = []
    metadata_path = root / "evaluation_metadata.json"
    if not metadata_path.is_file():
        raise EvaluationValidationError(["missing required file: evaluation_metadata.json"])
    try:
        metadata_obj = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError([f"could not read evaluation_metadata.json: {exc}"]) from exc
    metadata = _validate_metadata(metadata_obj, errors)

    tables = {
        key: _read_csv(root / filename, CSV_COLUMNS[key], key, errors)
        for key, filename in INPUT_FILES.items()
    }
    observed_hashes = observed_input_hashes(root)
    expected_hashes = metadata.get("input_sha256", {})
    if isinstance(expected_hashes, dict):
        for key, observed in observed_hashes.items():
            expected = expected_hashes.get(key)
            if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
                errors.append(f"metadata.input_sha256.{key} must be 64 lowercase hex characters")
            elif expected != observed:
                errors.append(
                    f"input hash mismatch for {INPUT_FILES[key]}: "
                    f"declared {expected}, observed {observed}"
                )

    query_rows = tables["query_set"]
    document_rows = tables["document_catalog"]
    run_rows = tables["retrieval_runs"]
    assessor_rows = tables["assessors"]
    judgment_rows = tables["relevance_judgments"]

    for rows, field, filename in (
        (query_rows, "query_id", INPUT_FILES["query_set"]),
        (document_rows, "doc_id", INPUT_FILES["document_catalog"]),
        (run_rows, "query_id", INPUT_FILES["retrieval_runs"]),
        (run_rows, "doc_id", INPUT_FILES["retrieval_runs"]),
        (judgment_rows, "query_id", INPUT_FILES["relevance_judgments"]),
        (judgment_rows, "doc_id", INPUT_FILES["relevance_judgments"]),
    ):
        _validate_identifiers(rows, field, filename, errors)

    query_ids: List[str] = []
    for row_number, row in enumerate(query_rows, start=2):
        query_id = row.get("query_id", "")
        query_ids.append(query_id)
        if not row.get("query_text", "").strip():
            errors.append(f"query_set.csv:{row_number} query_text is empty")
        declared = row.get("query_text_sha256", "")
        observed = _text_sha256(row.get("query_text", ""))
        if declared != observed:
            errors.append(
                f"query_set.csv:{row_number} query_text_sha256 does not match query_text"
            )
        for field in ("query_stratum", "source_record_id"):
            if not row.get(field, "").strip():
                errors.append(f"query_set.csv:{row_number} {field} is empty")
    if len(set(query_ids)) != len(query_ids):
        errors.append("query_set.csv query_id values must be unique")
    if len(query_ids) < 2:
        errors.append("at least two fixed queries are required for paired query-level uncertainty")

    doc_ids: List[str] = []
    content_hashes: List[str] = []
    for row_number, row in enumerate(document_rows, start=2):
        doc_ids.append(row.get("doc_id", ""))
        content_hash = row.get("content_sha256", "")
        content_hashes.append(content_hash)
        if not _SHA256_PATTERN.fullmatch(content_hash):
            errors.append(f"document_catalog.csv:{row_number} content_sha256 is invalid")
        for field in ("source_record_id", "descriptor"):
            if not row.get(field, "").strip():
                errors.append(f"document_catalog.csv:{row_number} {field} is empty")
    if len(set(doc_ids)) != len(doc_ids):
        errors.append("document_catalog.csv doc_id values must be unique")
    if len(set(content_hashes)) != len(content_hashes):
        errors.append("document_catalog.csv has duplicate content_sha256 values; pool is not deduplicated")

    assessor_ids: List[str] = []
    for row_number, row in enumerate(assessor_rows, start=2):
        assessor_id = row.get("assessor_id", "")
        assessor_ids.append(assessor_id)
        if not _ASSESSOR_ID_PATTERN.fullmatch(assessor_id):
            errors.append(
                f"assessors.csv:{row_number} assessor_id must be a pseudonymous "
                "identifier beginning with 'assessor_'"
            )
        if not row.get("expertise_category", "").strip():
            errors.append(f"assessors.csv:{row_number} expertise_category is empty")
        if row.get("independent_of_system_development", "").strip().lower() != "true":
            errors.append(
                f"assessors.csv:{row_number} independent_of_system_development must be true"
            )
        if row.get("blinded_to_system_identity", "").strip().lower() != "true":
            errors.append(f"assessors.csv:{row_number} blinded_to_system_identity must be true")
        if not row.get("provenance_record_id", "").strip():
            errors.append(f"assessors.csv:{row_number} provenance_record_id is empty")
        if not _valid_utc_timestamp(row.get("judgment_completed_at_utc", "")):
            errors.append(
                f"assessors.csv:{row_number} judgment_completed_at_utc must be an ISO-8601 UTC timestamp"
            )
    if len(set(assessor_ids)) != len(assessor_ids):
        errors.append("assessors.csv assessor_id values must be unique")
    declared_assessor_count = (metadata.get("assessment") or {}).get("assessor_count")
    if isinstance(declared_assessor_count, int) and declared_assessor_count != len(assessor_ids):
        errors.append(
            "metadata.assessment.assessor_count does not match the assessor registry"
        )

    query_set = set(query_ids)
    doc_set = set(doc_ids)
    run_depth = (metadata.get("study_design") or {}).get("run_depth")
    run_groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    seen_run_records: set[str] = set()
    for row_number, row in enumerate(run_rows, start=2):
        query_id = row.get("query_id", "")
        system_id = row.get("system_id", "")
        doc_id = row.get("doc_id", "")
        if query_id not in query_set:
            errors.append(f"retrieval_runs.csv:{row_number} references unknown query_id {query_id!r}")
        if system_id not in EXPECTED_SYSTEMS:
            errors.append(f"retrieval_runs.csv:{row_number} has unexpected system_id {system_id!r}")
        if doc_id not in doc_set:
            errors.append(f"retrieval_runs.csv:{row_number} references unknown doc_id {doc_id!r}")
        try:
            rank = int(row.get("rank", ""))
            if str(rank) != row.get("rank", "").strip() or rank < 1:
                raise ValueError
        except ValueError:
            errors.append(f"retrieval_runs.csv:{row_number} rank must be a positive integer")
            rank = -1
        score = row.get("score", "").strip()
        if score:
            try:
                if not math.isfinite(float(score)):
                    raise ValueError
            except ValueError:
                errors.append(f"retrieval_runs.csv:{row_number} score must be blank or finite")
        record_id = row.get("run_record_id", "").strip()
        if not record_id:
            errors.append(f"retrieval_runs.csv:{row_number} run_record_id is empty")
        elif record_id in seen_run_records:
            errors.append(f"retrieval_runs.csv:{row_number} duplicates run_record_id {record_id!r}")
        seen_run_records.add(record_id)
        run_groups.setdefault((query_id, system_id), []).append((rank, doc_id))

    expected_groups = {(query_id, system) for query_id in query_set for system in EXPECTED_SYSTEMS}
    if set(run_groups) != expected_groups:
        missing = sorted(expected_groups - set(run_groups))
        extra = sorted(set(run_groups) - expected_groups)
        if missing:
            errors.append(f"retrieval_runs.csv is missing query/system groups: {missing!r}")
        if extra:
            errors.append(f"retrieval_runs.csv has unexpected query/system groups: {extra!r}")
    if isinstance(run_depth, int):
        for group, values in sorted(run_groups.items()):
            ranks = [rank for rank, _ in values]
            docs = [doc for _, doc in values]
            if sorted(ranks) != list(range(1, run_depth + 1)):
                errors.append(f"retrieval run {group!r} must contain ranks 1..{run_depth}")
            if len(set(docs)) != len(docs):
                errors.append(f"retrieval run {group!r} contains a duplicate doc_id")

    scale = (metadata.get("assessment") or {}).get("judgment_scale")
    allowed_relevance = {0, 1} if scale == "binary_0_1" else {0, 1, 2}
    judgments: Dict[Tuple[str, str, str], int] = {}
    judgment_record_ids: set[str] = set()
    for row_number, row in enumerate(judgment_rows, start=2):
        key = (
            row.get("query_id", ""),
            row.get("doc_id", ""),
            row.get("assessor_id", ""),
        )
        if key[0] not in query_set:
            errors.append(f"relevance_judgments.csv:{row_number} references unknown query_id {key[0]!r}")
        if key[1] not in doc_set:
            errors.append(f"relevance_judgments.csv:{row_number} references unknown doc_id {key[1]!r}")
        if key[2] not in set(assessor_ids):
            errors.append(f"relevance_judgments.csv:{row_number} references unknown assessor_id {key[2]!r}")
        try:
            relevance = int(row.get("relevance", ""))
            if str(relevance) != row.get("relevance", "").strip() or relevance not in allowed_relevance:
                raise ValueError
        except ValueError:
            errors.append(
                f"relevance_judgments.csv:{row_number} relevance must be one of "
                f"{sorted(allowed_relevance)!r}"
            )
            relevance = -1
        if key in judgments:
            errors.append(f"relevance_judgments.csv:{row_number} duplicates judgment key {key!r}")
        judgments[key] = relevance
        record_id = row.get("judgment_record_id", "").strip()
        if not record_id:
            errors.append(f"relevance_judgments.csv:{row_number} judgment_record_id is empty")
        elif record_id in judgment_record_ids:
            errors.append(
                f"relevance_judgments.csv:{row_number} duplicates judgment_record_id {record_id!r}"
            )
        judgment_record_ids.add(record_id)

    pooled_pairs = {
        (query_id, doc_id)
        for (query_id, _system_id), values in run_groups.items()
        for _rank, doc_id in values
    }
    pooled_doc_ids = {doc_id for _query_id, doc_id in pooled_pairs}
    unused_catalog_docs = sorted(doc_set - pooled_doc_ids)
    if unused_catalog_docs:
        errors.append(
            "document_catalog.csv must be the exact deduplicated retrieved pool; "
            f"unused doc_id values: {unused_catalog_docs[:8]!r}"
        )
    expected_judgments = {
        (query_id, doc_id, assessor_id)
        for query_id, doc_id in pooled_pairs
        for assessor_id in assessor_ids
    }
    actual_judgments = set(judgments)
    missing_judgments = sorted(expected_judgments - actual_judgments)
    extra_judgments = sorted(actual_judgments - expected_judgments)
    if missing_judgments:
        preview = missing_judgments[:8]
        errors.append(
            f"missing {len(missing_judgments)} explicit pooled judgments; first keys: {preview!r}"
        )
    if extra_judgments:
        preview = extra_judgments[:8]
        errors.append(
            f"found {len(extra_judgments)} judgments outside the fixed retrieved pool; first keys: {preview!r}"
        )

    if errors:
        raise EvaluationValidationError(errors)
    return ValidatedEvaluationBundle(
        bundle_dir=root,
        metadata=metadata,
        query_rows=tuple(query_rows),
        document_rows=tuple(document_rows),
        run_rows=tuple(run_rows),
        assessor_rows=tuple(assessor_rows),
        judgment_rows=tuple(judgment_rows),
        observed_sha256=observed_hashes,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate independent, blinded human retrieval judgments."
    )
    parser.add_argument("bundle", type=Path, help="directory containing the six bundle files")
    parser.add_argument(
        "--print-observed-hashes",
        action="store_true",
        help="print current input file hashes, including for an incomplete template",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.print_observed_hashes:
        print(json.dumps(observed_input_hashes(args.bundle), indent=2, sort_keys=True))
    try:
        validated = validate_bundle(args.bundle)
    except EvaluationValidationError as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "eligible": True,
                "evidence_scope": EVIDENCE_SCOPE,
                "study_id": validated.metadata["study_id"],
                "queries": len(validated.query_rows),
                "assessors": len(validated.assessor_rows),
                "systems": list(EXPECTED_SYSTEMS),
                "downstream_ari_in_scope": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
