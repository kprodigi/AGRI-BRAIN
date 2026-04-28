#!/usr/bin/env python3
"""Verify decision-ledger Merkle roots, with optional on-chain proof.

Two verification modes:

1. **Off-chain self-consistency**: re-hash every record in a
   ``decision_ledger/<mode>__<scenario>.jsonl`` file, build the Merkle
   root from the canonical leaves, and assert it matches the header's
   recorded ``merkle_root`` field. Catches a ledger that was edited
   after writing, or a ledger whose leaves do not actually compose
   into the claimed root.

2. **On-chain anchoring proof** (optional, requires `web3` and a chain
   RPC): given a transaction hash returned by ``submit_onchain``,
   read the ``EpisodeLogged(root, …)`` event from the receipt and
   confirm the root recorded on-chain matches the local Merkle root.
   Closes the loop the audit flagged as "Merkle roots are produced
   but never verified anywhere".

Usage::

    # Off-chain only (default; no chain config needed)
    python mvp/simulation/analysis/verify_anchored_root.py

    # With on-chain anchoring proof for a single ledger. Substitute
    # your own transaction hash from the EpisodeLogged emit (read it
    # from the deploy log or `cast tx` against your RPC).
    python mvp/simulation/analysis/verify_anchored_root.py \\
        --ledger mvp/simulation/results/decision_ledger/agribrain__heatwave.jsonl \\
        --tx <0x... transaction hash from EpisodeLogged emit> \\
        --rpc http://127.0.0.1:8545

Exit codes:
- 0 when every checked ledger's Merkle root matches its leaves (and
  on-chain root, if --tx is provided).
- 1 on any mismatch or read error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_LEDGER_DIR = REPO_ROOT / "mvp" / "simulation" / "results" / "decision_ledger"


def _canonical_leaf(record: dict) -> str:
    """Canonicalise + SHA-256 a record exactly the way DecisionLedger does."""
    # Strip the recorder-side fields the writer added (`_leaf` is the
    # leaf hash itself; including it would create a chicken-and-egg).
    record_for_hash = {k: v for k, v in record.items() if k != "_leaf"}
    blob = json.dumps(record_for_hash, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _merkle_root(leaves: List[str]) -> str:
    """Binary Merkle root over hex leaf hashes; mirrors DecisionLedger.merkle_root."""
    if not leaves:
        return "0" * 64
    layer = [bytes.fromhex(h) for h in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer = layer + [layer[-1]]
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].hex()


def verify_ledger_offchain(path: Path) -> tuple[bool, str]:
    """Re-hash a single ledger JSONL and confirm the recorded root.

    Returns (ok, message).
    """
    if not path.exists():
        return False, f"missing: {path}"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return False, f"read error: {exc}"
    if not lines:
        return False, f"empty file: {path}"

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return False, f"bad header JSON: {exc}"
    if not header.get("_header"):
        return False, "first line is not a header"
    recorded_root = header.get("merkle_root", "")

    leaves: List[str] = []
    for i, ln in enumerate(lines[1:], start=2):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError as exc:
            return False, f"line {i}: bad JSON: {exc}"
        # Re-canonicalise and re-hash the record, then compare against
        # the leaf the writer recorded for it. Any divergence proves
        # the file was edited after writing.
        recomputed = _canonical_leaf(rec)
        recorded_leaf = rec.get("_leaf", "")
        if recorded_leaf and recorded_leaf != recomputed:
            return False, (
                f"line {i}: leaf mismatch (recorded={recorded_leaf[:16]}…, "
                f"recomputed={recomputed[:16]}…) — file was likely edited "
                "after the simulator wrote it"
            )
        leaves.append(recomputed)

    actual_root = _merkle_root(leaves)
    if actual_root != recorded_root:
        return False, (
            f"Merkle root mismatch: header says {recorded_root[:16]}…, "
            f"recomputed {actual_root[:16]}… from {len(leaves)} leaves"
        )
    return True, f"OK ({len(leaves)} leaves, root {actual_root[:16]}…)"


def verify_anchored_root_onchain(
    local_root_hex: str,
    tx_hash: str,
    rpc_url: str,
) -> tuple[bool, str]:
    """Read the EpisodeLogged event from a receipt and compare roots.

    Requires the ``web3`` package. The DecisionLogger contract emits
    ``EpisodeLogged(bytes32 root, …)`` from ``logEpisode``; the first
    indexed topic is the root. We ignore the rest of the event payload
    and just compare the recorded ``root`` against ``local_root_hex``.
    """
    try:
        from web3 import Web3
    except ImportError:
        return False, "web3 not installed; cannot verify on-chain root"

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        return False, f"cannot reach RPC at {rpc_url}"

    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
    except Exception as exc:
        return False, f"receipt fetch failed: {exc}"
    if receipt is None or receipt.get("status") != 1:
        return False, f"transaction {tx_hash} did not succeed"

    # EpisodeLogged: keccak256("EpisodeLogged(bytes32,uint256,string,string,uint256,uint256,string)")
    target_topic_root_hex = "0x" + (local_root_hex if not local_root_hex.startswith("0x") else local_root_hex[2:]).lower()
    for log in receipt.get("logs", []):
        topics = log.get("topics", [])
        if not topics:
            continue
        # Indexed bytes32 root is topics[1] when the event is the
        # canonical EpisodeLogged signature; topics[0] is the event
        # signature hash.
        if len(topics) >= 2:
            on_chain_root = topics[1].hex() if hasattr(topics[1], "hex") else str(topics[1])
            if on_chain_root.lower() == target_topic_root_hex.lower():
                return True, f"on-chain root matches: {on_chain_root}"
    return False, (
        f"no EpisodeLogged event with root {target_topic_root_hex} found "
        f"in tx {tx_hash}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Path to a single decision_ledger/*.jsonl file. "
             "When omitted, every *.jsonl under the default ledger "
             "directory is verified.",
    )
    p.add_argument(
        "--tx",
        default=None,
        help="Optional Ethereum transaction hash. When set, the script "
             "fetches the receipt via --rpc and checks the on-chain "
             "EpisodeLogged event's root against the local Merkle root. "
             "Requires --ledger (single-file mode) and the web3 package.",
    )
    p.add_argument(
        "--rpc",
        default="http://127.0.0.1:8545",
        help="HTTP RPC endpoint for on-chain verification (default: %(default)s).",
    )
    args = p.parse_args()

    if args.ledger is not None:
        targets = [args.ledger]
    else:
        if not DEFAULT_LEDGER_DIR.exists():
            print(f"FAIL: default ledger directory missing: {DEFAULT_LEDGER_DIR}")
            return 1
        targets = sorted(DEFAULT_LEDGER_DIR.glob("*.jsonl"))
        if not targets:
            print(f"FAIL: no *.jsonl files under {DEFAULT_LEDGER_DIR}")
            return 1

    errors = 0
    checked = 0
    for path in targets:
        ok, msg = verify_ledger_offchain(path)
        prefix = "OK  " if ok else "FAIL"
        print(f"{prefix}  {path.name}  {msg}")
        if not ok:
            errors += 1
        else:
            checked += 1

    if args.tx is not None:
        if args.ledger is None:
            print("FAIL: --tx requires --ledger (the local root to compare against)")
            return 1
        # Re-derive the local root from the named ledger.
        ledger_path = args.ledger
        try:
            header = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
            local_root = header.get("merkle_root", "")
        except (OSError, json.JSONDecodeError, IndexError) as exc:
            print(f"FAIL: cannot read local root for on-chain compare: {exc}")
            return 1
        ok, msg = verify_anchored_root_onchain(local_root, args.tx, args.rpc)
        prefix = "OK  " if ok else "FAIL"
        print(f"{prefix}  on-chain proof  {msg}")
        if not ok:
            errors += 1

    print(
        f"\nverify_anchored_root: checked {checked} ledger(s), "
        f"errors {errors}"
    )
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
