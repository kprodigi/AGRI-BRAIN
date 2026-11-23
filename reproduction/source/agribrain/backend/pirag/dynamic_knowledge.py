"""Optional decision-history ingestion diagnostic.

This module can summarize blocks of past routing records into documents that
an institutional-retrieval pipeline may ingest.  The feature is disabled by
default because feeding a system's own outputs back into retrieval can
self-amplify earlier choices.  It is not part of the locked publication
protocol or a learning mechanism.

The diagnostic normally uses caller-supplied in-memory records.  A caller may
explicitly request records from a configured EVM event log; that optional
source adapter has only been exercised with local development infrastructure
and is not evidence of a deployed blockchain feedback system.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


# DecisionLogger ABI fragment — only the event we read.
_DECISION_LOGGED_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "id", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "ts", "type": "uint256"},
            {"indexed": False, "internalType": "string",  "name": "agent", "type": "string"},
            {"indexed": False, "internalType": "string",  "name": "role", "type": "string"},
            {"indexed": False, "internalType": "string",  "name": "action", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "slca_milli", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "carbon_milli", "type": "uint256"},
            {"indexed": False, "internalType": "string",  "name": "note", "type": "string"},
        ],
        "name": "DecisionLogged",
        "type": "event",
    },
]


def _read_decisions_from_chain(n: int) -> Optional[List[Dict[str, Any]]]:
    """Try to read the last ``n`` ``DecisionLogged`` events.

    Returns ``None`` when the chain is not configured (no RPC, no
    contract address, or web3 unavailable). Returns ``[]`` when the configured
    endpoint responds but no matching event is found. This adapter is optional
    local research tooling; a successful read is not deployment validation.
    """
    try:
        from web3 import Web3
    except ImportError:
        return None

    rpc = os.environ.get("CHAIN_RPC", "")
    addr = os.environ.get("DECISION_LOGGER_ADDR", "")
    # Reuse an explicitly configured address from the local runtime when the
    # environment variables are absent.
    if not addr:
        try:
            from src.routers.governance import CHAIN as _CHAIN
            addr = (_CHAIN.get("addresses") or {}).get("DecisionLogger", "")
            if not rpc:
                rpc = _CHAIN.get("rpc") or ""
        except Exception:  # noqa: BLE001
            pass
    if not rpc or not addr:
        return None

    try:
        w3 = Web3(Web3.HTTPProvider(rpc))
        if not w3.is_connected():
            return None
        contract = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=_DECISION_LOGGED_ABI)
        # Bound the optional diagnostic query to avoid an unbounded log scan.
        latest = w3.eth.block_number
        from_block = max(0, latest - 5_000)
        events = contract.events.DecisionLogged.get_logs(from_block=from_block)
        records: List[Dict[str, Any]] = []
        for ev in events[-int(max(1, n)):]:
            args = ev["args"]
            records.append({
                "ts": int(args["ts"]),
                "agent": str(args["agent"]),
                "role": str(args["role"]),
                "action": str(args["action"]),
                "slca": float(args["slca_milli"]) / 1000.0,
                "carbon_kg": float(args["carbon_milli"]) / 1000.0,
                "note": str(args.get("note", "")),
                "tx_hash": ev["transactionHash"].hex(),
                "block": ev["blockNumber"],
                "_source": "on_chain",
            })
        return records
    except Exception as exc:  # noqa: BLE001
        _log.debug("on-chain decision read failed: %s", exc)
        return None


def synthesize_decision_document(
    decisions: List[Dict[str, Any]],
    scenario: str,
    hour_range: tuple[float, float],
) -> Dict[str, Any]:
    """Synthesize a piR-ingestible document from routing decisions.

    Parameters
    ----------
    decisions : list of decision dicts with keys: action, role, slca, carbon_kg, waste.
    scenario : current scenario name.
    hour_range : (start_hour, end_hour) of the decision block.

    Returns
    -------
    Dict with id, text, and metadata for piR ingestion.
    """
    if not decisions:
        return {"id": "empty_block", "text": "", "metadata": {}}

    action_counts = Counter(d.get("action", "unknown") for d in decisions)
    total = len(decisions)
    action_dist = {a: round(c / total, 2) for a, c in action_counts.items()}

    mean_slca = sum(d.get("slca", 0.0) for d in decisions) / total
    total_carbon = sum(d.get("carbon_kg", 0.0) for d in decisions)
    mean_waste = sum(d.get("waste", 0.0) for d in decisions) / total

    # Descriptive labels for this synthetic author-declared proxy only.
    if mean_slca > 0.70 and mean_waste < 0.05:
        assessment = "higher social-performance proxy with lower modeled waste"
    elif mean_slca > 0.50:
        assessment = "mid-range social-performance proxy"
    else:
        assessment = "lower social-performance proxy in this recorded block"

    doc_id = f"decisions_{scenario}_{hour_range[0]:.0f}_{hour_range[1]:.0f}"
    text = (
        f"Decision history for {scenario} scenario, hours {hour_range[0]:.1f} to {hour_range[1]:.1f}. "
        f"Action distribution: {action_dist}. "
        f"Mean author-declared social-performance proxy: {mean_slca:.3f}. "
        f"Sum of the modeled transport-emissions indicator: {total_carbon:.1f} kg CO2-eq. "
        f"Mean modeled waste fraction per routing opportunity: {mean_waste:.4f}. "
        f"Descriptive block label: {assessment}. "
        f"Total decisions in block: {total}."
    )

    return {
        "id": doc_id,
        "text": text,
        "metadata": {
            # ``source`` retains its historical value for schema compatibility.
            "source": "decision_feedback",
            "source_label": "decision_history_diagnostic",
            "source_is_legacy_alias": True,
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
    *,
    prefer_chain: bool = False,
    chain_window: int = 96,
) -> int:
    """Optionally ingest decision-history summaries into retrieval.

    Parameters
    ----------
    pipeline : PiRPipeline instance.
    decisions : in-memory decision history from the coordinator. This is the
        default source for the explicitly enabled diagnostic.
    scenario : current scenario name.
    block_size : number of decisions per block (24 = 6 hours at 15-min steps).
    prefer_chain : when True, explicitly try the optional event-log adapter
        first and fall back to ``decisions`` if it is unavailable. Defaults
        to False; no chain read is implied by ordinary ingestion.
    chain_window : number of most-recent on-chain ``DecisionLogged``
        events to fetch when ``prefer_chain`` is enabled. Mirrors the
        coordinator's 24-step block size by default (4 blocks).

    Returns
    -------
    Number of documents ingested.
    """
    if pipeline is None:
        return 0

    source = "memory"
    effective_decisions = decisions
    if prefer_chain:
        chain_records = _read_decisions_from_chain(chain_window)
        if chain_records is not None and chain_records:
            effective_decisions = chain_records
            source = "on_chain"

    if not effective_decisions:
        return 0

    docs_ingested = 0
    seen_ids = set()
    for start in range(0, len(effective_decisions), block_size):
        block = effective_decisions[start:start + block_size]
        if len(block) < block_size // 2:
            continue

        hour_start = block[0].get("hour", start * 0.25)
        hour_end = block[-1].get("hour", (start + len(block)) * 0.25)

        doc = synthesize_decision_document(block, scenario, (hour_start, hour_end))
        # Record the source used by this diagnostic.
        if doc.get("metadata"):
            doc["metadata"]["source_kind"] = source
            doc["metadata"]["source"] = "decision_feedback"
            doc["metadata"]["source_label"] = "decision_history_diagnostic"
            doc["metadata"]["source_is_legacy_alias"] = True
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
