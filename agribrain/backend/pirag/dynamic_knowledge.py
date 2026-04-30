"""Blockchain-to-piRAG feedback loop.

Synthesizes piRAG-ingestible documents from blocks of routing decisions,
creating a feedback loop where past decision patterns inform future
retrievals. Called periodically during simulation (every 24 timesteps /
6 hours).

When a permissioned EVM is configured (CHAIN_RPC + a deployed
DecisionLogger address) the loop reads the most recent
``DecisionLogged`` events directly from the chain, satisfying the
Section 3.7 framing of a true *blockchain*-to-retrieval feedback
loop. When no chain is configured the loop falls back to the
caller-supplied in-memory decision history so the simulator and
benchmark suites still exercise the synthesis path without needing
Hardhat running.
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
    """Pull the last ``n`` ``DecisionLogged`` events from the chain.

    Returns ``None`` when the chain is not configured (no RPC, no
    contract address, web3 not installed). Returns ``[]`` when the
    chain is reachable but the contract has not emitted any events
    yet — distinct from the unconfigured case so the caller can
    decide whether to fall back to in-memory history.
    """
    try:
        from web3 import Web3
    except ImportError:
        return None

    rpc = os.environ.get("CHAIN_RPC", "")
    addr = os.environ.get("DECISION_LOGGER_ADDR", "")
    # Allow the governance router's auto-loaded addresses to be the
    # source of truth when env vars are absent.
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
        # Look back over at most 5_000 blocks; in dev that covers ~1 day on
        # the local Hardhat node and is well below the eth_getLogs limit
        # most providers enforce.
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
    *,
    prefer_chain: bool = True,
    chain_window: int = 96,
) -> int:
    """Ingest blocks of decisions into the piRAG knowledge base.

    Parameters
    ----------
    pipeline : PiRAGPipeline instance.
    decisions : in-memory decision history from the coordinator. Used as
        the source of truth when no chain is configured (typical
        simulator runs) and as the fallback when a chain is
        configured but unreachable.
    scenario : current scenario name.
    block_size : number of decisions per block (24 = 6 hours at 15-min steps).
    prefer_chain : when True, attempt to read the latest decisions from
        the on-chain ``DecisionLogger`` first. Falls back to
        ``decisions`` when chain not configured or unreachable. Default
        True so the §3.7 "blockchain-to-retrieval feedback loop"
        framing is honoured by default whenever a chain is up.
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
        # Tag the document with its source so it is auditable
        # whether the feedback loop was reading the chain or the
        # in-memory fallback at synthesis time.
        if doc.get("metadata"):
            doc["metadata"]["source_kind"] = source
            doc["metadata"]["source"] = "decision_feedback"
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
