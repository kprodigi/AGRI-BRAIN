"""Blockchain-to-piRAG feedback loop.

Synthesizes piRAG-ingestible documents from blocks of routing decisions,
creating a feedback loop where past decision patterns inform future
retrievals. Called periodically during simulation (every 24 timesteps /
6 hours).
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

_log = logging.getLogger(__name__)


def synthesize_decision_document(
    decisions: List[Dict[str, Any]],
    scenario: str,
    hour_range: tuple[float, float],
) -> Dict[str, Any]:
    """Synthesize a piRAG-ingestible document from routing decisions.

    Parameters
    ----------
    decisions : list of decision dicts with keys: action, role, slca, carbon_kg, waste.
    scenario : current scenario name.
    hour_range : (start_hour, end_hour) of the decision block.

    Returns
    -------
    Dict with id, text, and metadata for piRAG ingestion.
    """
    if not decisions:
        return {"id": "empty_block", "text": "", "metadata": {}}

    action_counts = Counter(d.get("action", "unknown") for d in decisions)
    total = len(decisions)
    action_dist = {a: round(c / total, 2) for a, c in action_counts.items()}

    mean_slca = sum(d.get("slca", 0.0) for d in decisions) / total
    total_carbon = sum(d.get("carbon_kg", 0.0) for d in decisions)
    mean_waste = sum(d.get("waste", 0.0) for d in decisions) / total

    # Assess performance
    if mean_slca > 0.70 and mean_waste < 0.05:
        assessment = "strong social and waste performance"
    elif mean_slca > 0.50:
        assessment = "moderate social performance with room for waste reduction"
    else:
        assessment = "below target social performance requiring policy review"

    doc_id = f"decisions_{scenario}_{hour_range[0]:.0f}_{hour_range[1]:.0f}"
    text = (
        f"Decision history for {scenario} scenario, hours {hour_range[0]:.1f} to {hour_range[1]:.1f}. "
        f"Action distribution: {action_dist}. "
        f"Mean SLCA score: {mean_slca:.3f}. "
        f"Total carbon emissions: {total_carbon:.1f} kg. "
        f"Mean waste rate: {mean_waste:.4f}. "
        f"Performance assessment: {assessment}. "
        f"Total decisions in block: {total}."
    )

    return {
        "id": doc_id,
        "text": text,
        "metadata": {
            "source": "decision_feedback",
            "scenario": scenario,
            "hour_start": hour_range[0],
            "hour_end": hour_range[1],
            "n_decisions": total,
        },
    }


def ingest_decision_history(
    pipeline: Any,
    decisions: List[Dict[str, Any]],
    scenario: str,
    block_size: int = 24,
) -> int:
    """Ingest blocks of decisions into the piRAG knowledge base.

    Parameters
    ----------
    pipeline : PiRAGPipeline instance.
    decisions : full list of decision dicts from the episode so far.
    scenario : current scenario name.
    block_size : number of decisions per block (24 = 6 hours at 15-min steps).

    Returns
    -------
    Number of documents ingested.
    """
    if pipeline is None or not decisions:
        return 0

    docs_ingested = 0
    seen_ids = set()
    for start in range(0, len(decisions), block_size):
        block = decisions[start:start + block_size]
        if len(block) < block_size // 2:
            continue

        hour_start = block[0].get("hour", start * 0.25)
        hour_end = block[-1].get("hour", (start + len(block)) * 0.25)

        doc = synthesize_decision_document(block, scenario, (hour_start, hour_end))
        if doc["text"] and doc["id"] not in seen_ids:
            # Guard against injecting very low-information blocks.
            if "Total decisions in block: 0" in doc["text"]:
                continue
            try:
                pipeline.ingest([doc])
                docs_ingested += 1
                seen_ids.add(doc["id"])
            except Exception as _exc:
                _log.debug("dynamic knowledge ingest skipped for doc %s: %s", doc.get("id", "?"), _exc)

    return docs_ingested
