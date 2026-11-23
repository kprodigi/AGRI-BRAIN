"""Revised context design: constraint and eligibility features, acceptance policy.

The submitted design filled two context components from retrieval metadata
(fused-rank strength and a filename flag) and one from the recent share of
recovery decisions. Rank strength was close to one on every decision, so it
acted as a fixed bias; the flag followed the role schedule. The revised design
replaces the three components with quantities that the route outcome depends on
and that the decision side cannot observe locally:

    psi_0  operating-envelope severity      envelope tool (unchanged)
    psi_1  forecast urgency                 forecast tool (unchanged)
    psi_2  redistribution shortfall         capacity tool
    psi_3  redistribution ineligibility     retrieved acceptance policy
    psi_4  recovery shortfall               capacity tool

The acceptance criteria of the two destinations are stated in one corpus
document. The decision side learns them by retrieving that document; the values
it reads are the ones it then applies. Document, weights and thresholds are
synthetic design values of the benchmark.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .context_builder import DEFAULT_GUARD_CONSTRAINTS
from .context_to_logits import URGENCY_MAP
from .guards.feasibility_guard import within_ranges
from .guards.retrieval_guard import MIN_TOP_CITATION_SCORE, retrieval_quality_ok
from .guards.unit_guard import units_consistent

# Context weights of the revised design (3 actions x 5 components). Columns 0,
# 1 and 4 keep the submitted values. The two redistribution-constraint columns
# mirror the recovery column with redistribution and recovery exchanged, so no
# new magnitude is introduced.
THETA_CONTEXT_REVISED: np.ndarray = np.array([
    # psi_0     psi_1     psi_2     psi_3     psi_4
    [-0.40,    -0.30,    +0.12,    +0.12,    +0.12],   # cold chain
    [+0.30,    +0.25,    -0.20,    -0.20,    +0.08],   # local redistribution
    [+0.15,    +0.10,    +0.08,    +0.08,    -0.20],   # recovery
], dtype=np.float64)

TOOL_FEATURE_INDICES: Tuple[int, ...] = (0, 1, 2, 4)
RETRIEVAL_FEATURE_INDICES: Tuple[int, ...] = (3,)
FEATURE_NAMES: Tuple[str, ...] = (
    "operating_envelope_severity",
    "forecast_urgency",
    "redistribution_shortfall",
    "redistribution_ineligibility",
    "recovery_shortfall",
)

ACCEPTANCE_POLICY_DOC_ID = "destination_acceptance_policy"
ACCEPTANCE_POLICY_QUERY = (
    "destination acceptance policy: redistribution maximum modeled risk and "
    "recovery minimum modeled risk for a dispatch"
)
_KEY_REDISTRIBUTION_MAX = "redistribution_maximum_modeled_risk"
_KEY_REDISTRIBUTION_MIN = "redistribution_minimum_modeled_risk"
_KEY_RECOVERY_MIN = "recovery_minimum_modeled_risk"


def render_acceptance_policy(
    rho_at: float, rho_cut: float, strict_redistribution: bool = False,
) -> str:
    """Text of the acceptance-policy corpus document for the stated criteria."""

    lines = [
        "DESTINATION ACCEPTANCE POLICY FOR THE SYNTHETIC BENCHMARK",
        "Status: Constructed benchmark input; it states the acceptance criteria "
        "that the two modeled destinations apply. It is not a food-safety rule, "
        "a donation standard or a legal requirement.",
        "Source scope: Constructed institutional-retrieval input for the "
        "synthetic AGRI-BRAIN benchmark. It is not law, certification, field "
        "evidence, measured performance, or operational advice.",
        "Document ID: KB-DAP-001",
        "",
        "Destination acceptance policy. A dispatch routed to local "
        "redistribution or to recovery is taken by the destination only when "
        "the product meets that destination's criterion on modeled spoilage "
        "risk.",
        "",
        "Local redistribution outlets accept a dispatch while the product is "
        "marketable.",
        f"{_KEY_REDISTRIBUTION_MAX} = {float(rho_cut):.4f}",
    ]
    if strict_redistribution:
        lines += [
            "Local redistribution outlets accept only product that is at risk.",
            f"{_KEY_REDISTRIBUTION_MIN} = {float(rho_at):.4f}",
        ]
    lines += [
        "",
        "The recovery facility accepts a dispatch when the product is at risk.",
        f"{_KEY_RECOVERY_MIN} = {float(rho_at):.4f}",
        "",
        "A dispatch that a destination does not accept continues by cold chain "
        "after the diverted leg. Acceptance is separate from capacity: a "
        "destination also needs room for the dispatch.",
    ]
    return "\n".join(lines)


def _read_value(text: str, key: str) -> Optional[float]:
    match = re.search(rf"^{re.escape(key)}\s*=\s*([0-9]*\.?[0-9]+)\s*$", text, re.M)
    if match is None:
        return None
    value = float(match.group(1))
    return value if 0.0 <= value <= 1.0 else None


def parse_acceptance_policy(text: str) -> Optional[Dict[str, Any]]:
    """Read the acceptance criteria stated in the policy document.

    Returns ``None`` unless both required values are present and consistent.
    """

    rho_cut = _read_value(text, _KEY_REDISTRIBUTION_MAX)
    rho_at = _read_value(text, _KEY_RECOVERY_MIN)
    if rho_cut is None or rho_at is None or not rho_at < rho_cut:
        return None
    redistribution_min = _read_value(text, _KEY_REDISTRIBUTION_MIN)
    return {
        "rho_at": float(rho_at),
        "rho_cut": float(rho_cut),
        "strict_redistribution": redistribution_min is not None,
    }


def unavailable_acceptance_policy(reason: str) -> Dict[str, Any]:
    """Record for a decision at which the acceptance policy is not known."""

    return {
        "available": False,
        "reason": str(reason),
        "rho_at": None,
        "rho_cut": None,
        "strict_redistribution": False,
        "doc_id": "",
        "passage_sha256": "",
        "fused_score": 0.0,
        "guard_breakdown": {},
        "query": "",
    }


def published_criteria(policy: Any) -> Dict[str, Any]:
    """Criteria stated in the acceptance document of a configuration.

    Unless a different published value is declared, these are the criteria the
    destinations apply.
    """

    published_at = getattr(policy, "published_rho_at", None)
    published_cut = getattr(policy, "published_rho_cut", None)
    return {
        "rho_at": float(
            getattr(policy, "acceptance_rho_at", 0.10)
            if published_at is None else published_at
        ),
        "rho_cut": float(
            getattr(policy, "acceptance_rho_cut", 0.50)
            if published_cut is None else published_cut
        ),
        "strict_redistribution": bool(
            getattr(policy, "acceptance_strict_redistribution", False)
        ),
    }


def retrieve_acceptance_policy(pipeline: Any) -> Dict[str, Any]:
    """Retrieve the acceptance policy and read its criteria.

    The policy is usable when the top-ranked passage states both criteria and
    clears the same three checks as every other retrieval: the fused-rank
    floor, the unit check and the numerical-range check. The passage is
    recognized by its content; its identifier is recorded and not used.
    """

    if pipeline is None:
        return unavailable_acceptance_policy("retrieval_pipeline_unavailable")
    response = pipeline.ask(ACCEPTANCE_POLICY_QUERY, k=4, anchor_on_chain=False)
    citations = list(getattr(response, "citations", []) or [])
    if not citations:
        return unavailable_acceptance_policy("no_passage_retrieved")
    top = citations[0]
    passage = str(top.passage)
    fused_score = float(top.fused_score)
    retrieval_ok = retrieval_quality_ok(citations, fused_score)
    unit_ok = units_consistent(passage) if passage else True
    range_ok = within_ranges(passage, DEFAULT_GUARD_CONSTRAINTS) if passage else True
    record = unavailable_acceptance_policy("")
    record.update({
        "doc_id": str(top.doc_id),
        "passage_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        "fused_score": fused_score,
        "fused_score_floor_exclusive": float(MIN_TOP_CITATION_SCORE),
        "guard_breakdown": {
            "retrieval": bool(retrieval_ok),
            "unit": bool(unit_ok),
            "feasibility": bool(range_ok),
        },
        "query": ACCEPTANCE_POLICY_QUERY,
        "ranked_doc_ids": [str(c.doc_id) for c in citations],
    })
    criteria = parse_acceptance_policy(passage)
    if criteria is None:
        record["reason"] = "criteria_not_in_top_passage"
        return record
    if not (retrieval_ok and unit_ok and range_ok):
        record["reason"] = "retrieval_gate_closed"
        return record
    record.update(criteria)
    record["available"] = True
    record["reason"] = "retrieved"
    return record


def redistribution_ineligible(rho: float, policy: Mapping[str, Any]) -> bool:
    """Whether the retrieved criteria put ``rho`` outside the redistribution band."""

    if not policy.get("available", False):
        return False
    rho = float(rho)
    if rho > float(policy["rho_cut"]):
        return True
    return bool(policy.get("strict_redistribution", False)
                and rho <= float(policy["rho_at"]))


def extract_revised_context_features(
    mcp_results: Optional[Mapping[str, Any]],
    acceptance_policy: Optional[Mapping[str, Any]],
    obs: Any,
) -> np.ndarray:
    """Five-component context vector of the revised design, each in [0, 1].

    A missing, failed or skipped tool result leaves its component at zero, as
    in the submitted design. The eligibility component is zero unless the
    acceptance policy was retrieved.
    """

    psi = np.zeros(5, dtype=np.float64)
    mcp_results = mcp_results or {}

    compliance = mcp_results.get("check_compliance") or {}
    if not compliance.get("compliant", True):
        violations = compliance.get("violations", [])
        if any(v.get("severity") == "critical" for v in violations):
            psi[0] = 1.0
        elif violations:
            psi[0] = 0.5

    forecast = mcp_results.get("spoilage_forecast") or {}
    psi[1] = URGENCY_MAP.get(forecast.get("urgency", ""), 0.0)

    capacity = mcp_results.get("destination_capacity") or {}
    if capacity.get("_status") == "ok":
        psi[2] = float(np.clip(capacity.get("redistribution_shortfall", 0.0), 0.0, 1.0))
        psi[4] = float(np.clip(capacity.get("recovery_shortfall", 0.0), 0.0, 1.0))

    if acceptance_policy and redistribution_ineligible(
        float(getattr(obs, "rho", 0.0)), acceptance_policy,
    ):
        psi[3] = 1.0
    return psi


def capacity_report(mcp_results: Optional[Mapping[str, Any]]) -> Optional[Dict[str, float]]:
    """Capacity report carried by the tool results, or ``None`` if unread."""

    capacity = (mcp_results or {}).get("destination_capacity") or {}
    if capacity.get("_status") != "ok":
        return None
    return {
        "dispatch_size": float(capacity["dispatch_size"]),
        "redistribution_capacity": float(capacity["redistribution_capacity"]),
        "recovery_capacity": float(capacity["recovery_capacity"]),
    }


__all__ = [
    "THETA_CONTEXT_REVISED", "TOOL_FEATURE_INDICES", "RETRIEVAL_FEATURE_INDICES",
    "FEATURE_NAMES", "ACCEPTANCE_POLICY_DOC_ID", "ACCEPTANCE_POLICY_QUERY",
    "render_acceptance_policy", "parse_acceptance_policy", "published_criteria",
    "unavailable_acceptance_policy", "retrieve_acceptance_policy",
    "redistribution_ineligible", "extract_revised_context_features",
    "capacity_report",
]
