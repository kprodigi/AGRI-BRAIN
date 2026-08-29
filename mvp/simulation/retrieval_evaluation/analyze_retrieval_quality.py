"""Analyze a validated Standard-RAG versus piRAG retrieval study.

The analysis unit is the fixed query.  Metrics are first computed separately
for each independent assessor, then averaged within query.  Uncertainty for the
paired piRAG-minus-Standard-RAG contrast is a seeded percentile bootstrap over
queries.  Only the predeclared nDCG cutoff is inferential; other metrics are
descriptive.

No output is produced unless :mod:`validate_retrieval_evaluation` accepts the
entire evidence bundle.  The resulting evidence concerns retrieval ranking,
not the downstream Adaptive Resilience Index (ARI).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Supports both ``python -m`` and direct script execution.
    from .validate_retrieval_evaluation import (
        CANDIDATE_SYSTEM_ID,
        CONTROL_SYSTEM_ID,
        EVIDENCE_SCOPE,
        ValidatedEvaluationBundle,
        validate_bundle,
    )
except ImportError:  # pragma: no cover - exercised only by direct CLI use
    from validate_retrieval_evaluation import (  # type: ignore
        CANDIDATE_SYSTEM_ID,
        CONTROL_SYSTEM_ID,
        EVIDENCE_SCOPE,
        ValidatedEvaluationBundle,
        validate_bundle,
    )


MetricValue = Optional[float]


def _dcg(relevance: Sequence[int]) -> float:
    return sum(
        ((2.0 ** grade) - 1.0) / math.log2(rank + 1.0)
        for rank, grade in enumerate(relevance, start=1)
    )


def _metrics_at_k(
    ranked_docs: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> Dict[str, MetricValue]:
    retrieved_grades = [qrels[doc_id] for doc_id in ranked_docs[:k]]
    binary = [1 if grade >= 1 else 0 for grade in retrieved_grades]
    all_relevant = sum(1 for grade in qrels.values() if grade >= 1)
    ideal_grades = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_grades)

    precision = sum(binary) / float(k)
    recall = sum(binary) / float(all_relevant) if all_relevant else None
    reciprocal_rank = next(
        (1.0 / rank for rank, value in enumerate(binary, start=1) if value),
        0.0,
    )
    if all_relevant:
        hits = 0
        precision_sum = 0.0
        for rank, value in enumerate(binary, start=1):
            if value:
                hits += 1
                precision_sum += hits / float(rank)
        average_precision = precision_sum / float(min(all_relevant, k))
    else:
        average_precision = None
    return {
        "ndcg": _dcg(retrieved_grades) / ideal_dcg if ideal_dcg else 0.0,
        "precision": precision,
        "recall_in_judged_pool": recall,
        "reciprocal_rank": reciprocal_rank,
        "average_precision_in_judged_pool": average_precision,
    }


def _mean_defined(values: Iterable[MetricValue]) -> MetricValue:
    defined = [float(value) for value in values if value is not None]
    return fmean(defined) if defined else None


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_percentile_bootstrap(
    differences: Sequence[float],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> Optional[Tuple[float, float]]:
    if len(differences) < 2:
        return None
    rng = random.Random(seed)
    count = len(differences)
    bootstrap_means = [
        fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    alpha = 1.0 - confidence_level
    return (
        _percentile(bootstrap_means, alpha / 2.0),
        _percentile(bootstrap_means, 1.0 - alpha / 2.0),
    )


def _metric_seed(base_seed: int, metric_name: str) -> int:
    digest = hashlib.sha256(metric_name.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def _parsed_inputs(bundle: ValidatedEvaluationBundle):
    queries = [row["query_id"] for row in bundle.query_rows]
    assessors = [row["assessor_id"] for row in bundle.assessor_rows]
    rankings: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for row in bundle.run_rows:
        rankings.setdefault((row["query_id"], row["system_id"]), []).append(
            (int(row["rank"]), row["doc_id"])
        )
    ranked_docs = {
        key: [doc_id for _rank, doc_id in sorted(values)]
        for key, values in rankings.items()
    }
    qrels: Dict[Tuple[str, str], Dict[str, int]] = {}
    for row in bundle.judgment_rows:
        qrels.setdefault((row["query_id"], row["assessor_id"]), {})[
            row["doc_id"]
        ] = int(row["relevance"])
    return queries, assessors, ranked_docs, qrels


def analyze_bundle(bundle: ValidatedEvaluationBundle) -> Dict[str, object]:
    metadata = bundle.metadata
    analysis_plan = metadata["analysis"]
    cutoffs = list(analysis_plan["cutoffs"])
    confidence = float(analysis_plan["bootstrap_confidence_level"])
    resamples = int(analysis_plan["bootstrap_resamples"])
    base_seed = int(analysis_plan["bootstrap_seed"])
    queries, assessors, rankings, qrels = _parsed_inputs(bundle)

    per_query: List[Dict[str, object]] = []
    paired_values: Dict[str, List[Tuple[float, float]]] = {}
    for query_id in queries:
        query_result: Dict[str, object] = {"query_id": query_id, "metrics": {}}
        metrics_out: Dict[str, object] = query_result["metrics"]  # type: ignore[assignment]
        for cutoff in cutoffs:
            assessor_metrics: Dict[str, Dict[str, List[MetricValue]]] = {
                system: {} for system in (CONTROL_SYSTEM_ID, CANDIDATE_SYSTEM_ID)
            }
            for assessor_id in assessors:
                grades = qrels[(query_id, assessor_id)]
                for system in (CONTROL_SYSTEM_ID, CANDIDATE_SYSTEM_ID):
                    values = _metrics_at_k(rankings[(query_id, system)], grades, cutoff)
                    for metric, value in values.items():
                        assessor_metrics[system].setdefault(metric, []).append(value)

            metric_names = tuple(assessor_metrics[CONTROL_SYSTEM_ID])
            for metric in metric_names:
                metric_key = f"{metric}@{cutoff}"
                control_value = _mean_defined(assessor_metrics[CONTROL_SYSTEM_ID][metric])
                candidate_value = _mean_defined(assessor_metrics[CANDIDATE_SYSTEM_ID][metric])
                difference = (
                    candidate_value - control_value
                    if control_value is not None and candidate_value is not None
                    else None
                )
                metrics_out[metric_key] = {
                    CONTROL_SYSTEM_ID: control_value,
                    CANDIDATE_SYSTEM_ID: candidate_value,
                    "paired_difference_pirag_minus_standard_rag": difference,
                }
                if difference is not None:
                    paired_values.setdefault(metric_key, []).append(
                        (control_value, candidate_value)  # type: ignore[arg-type]
                    )
        per_query.append(query_result)

    summary: Dict[str, object] = {}
    for metric_key, pairs in sorted(paired_values.items()):
        controls = [pair[0] for pair in pairs]
        candidates = [pair[1] for pair in pairs]
        differences = [candidate - control for control, candidate in pairs]
        interval = _paired_percentile_bootstrap(
            differences,
            confidence,
            resamples,
            _metric_seed(base_seed, metric_key),
        )
        summary[metric_key] = {
            "n_paired_queries": len(pairs),
            "standard_rag_mean": fmean(controls),
            "pirag_mean": fmean(candidates),
            "paired_mean_difference_pirag_minus_standard_rag": fmean(differences),
            "paired_percentile_bootstrap_interval": list(interval) if interval else None,
            "confidence_level": confidence,
            "bootstrap_resamples": resamples,
            "bootstrap_unit": "query",
        }

    primary_key = f"ndcg@{analysis_plan['primary_cutoff']}"
    primary_summary = summary[primary_key]
    primary_interval = primary_summary["paired_percentile_bootstrap_interval"]  # type: ignore[index]
    supports_direction = bool(primary_interval and primary_interval[0] > 0.0)

    assessor_provenance = [
        {
            "assessor_id": row["assessor_id"],
            "expertise_category": row["expertise_category"],
            "provenance_record_id": row["provenance_record_id"],
            "judgment_completed_at_utc": row["judgment_completed_at_utc"],
        }
        for row in bundle.assessor_rows
    ]
    return {
        "schema_version": "agribrain-independent-retrieval-results-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "study_id": metadata["study_id"],
        "evidence_scope": EVIDENCE_SCOPE,
        "downstream_ari_in_scope": False,
        "comparison": metadata["comparison"],
        "validated_input_sha256": bundle.observed_sha256,
        "design": {
            "query_count": len(queries),
            "assessor_count": len(assessors),
            "run_depth": metadata["study_design"]["run_depth"],
            "judgment_scale": metadata["assessment"]["judgment_scale"],
            "query_aggregation": analysis_plan["query_aggregation"],
            "assessor_provenance": assessor_provenance,
        },
        "primary_inference": {
            "metric": primary_key,
            "contrast": "piRAG_minus_Standard_RAG",
            "interval_method": "paired_query_percentile_bootstrap",
            "interval_excludes_zero_in_declared_direction": supports_direction,
            "claim_scope": (
                "retrieval ranking on this fixed query set, judged document pool, "
                "run depth, and assessor cohort only"
            ),
            "not_evidence_for": "downstream ARI or operational performance",
        },
        "metric_summary": summary,
        "per_query": per_query,
        "limitations": [
            "The validator checks recorded attestations and provenance identifiers; it cannot independently prove assessor independence or blinding.",
            "Recall and average precision are relative to the explicitly judged pooled documents, not every document in the corpus.",
            "The percentile bootstrap treats fixed queries as the paired resampling unit and averages assessor-specific metrics within query; it does not separately model assessor sampling.",
            "Only the predeclared nDCG cutoff is inferential. All other cutoffs and metrics are descriptive.",
            "These results do not test or support downstream Adaptive Resilience Index (ARI) claims.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze independently judged Standard-RAG versus piRAG rankings."
    )
    parser.add_argument("bundle", type=Path, help="validated evaluation bundle directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    validated = validate_bundle(args.bundle)
    result = analyze_bundle(validated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "study_id": result["study_id"],
                "evidence_scope": EVIDENCE_SCOPE,
                "primary_inference": result["primary_inference"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
