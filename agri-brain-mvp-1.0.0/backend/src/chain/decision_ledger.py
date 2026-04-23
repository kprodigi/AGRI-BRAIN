"""Per-episode decision ledger with Merkle anchoring.

Every routing decision the simulator emits is appended to a
:class:`DecisionLedger`. The ledger canonicalises each record, computes
a SHA-256 leaf hash, and produces a single 32-byte Merkle root over
the full episode. The root can be committed on-chain via
``log_episode_onchain`` so any individual decision is verifiable via
inclusion proof while only one transaction is paid per episode.

This module is the single per-step write point used by both the
HPC simulator and the FastAPI ``/decide`` endpoint, so the paper's
"on-chain auditability of every decision" claim is backed by code in
the simulation loop, not just the production endpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _canonical_bytes(record: Dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merkle_root_hex(leaves: List[str]) -> str:
    """Binary Merkle root over hex leaf hashes.

    Empty input -> 32 zero bytes. Odd-length layers duplicate the last
    leaf (Bitcoin-style) so the root depth is always log2 of a padded
    power-of-two layer.
    """
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


class DecisionLedger:
    """Append-only ledger of decisions for a single episode."""

    def __init__(self, episode_metadata: Optional[Dict[str, Any]] = None) -> None:
        self._records: List[Dict[str, Any]] = []
        self._leaves: List[str] = []
        self.metadata: Dict[str, Any] = dict(episode_metadata or {})

    def __len__(self) -> int:
        return len(self._records)

    def append(self, record: Dict[str, Any]) -> str:
        """Append a decision record. Returns the leaf hash (hex)."""
        leaf = _sha256_hex(_canonical_bytes(record))
        self._records.append(dict(record))
        self._leaves.append(leaf)
        return leaf

    def merkle_root(self) -> str:
        return merkle_root_hex(self._leaves)

    def write_jsonl(self, path: Path) -> Path:
        """Write the ledger to a JSONL file with a header line carrying the
        Merkle root and episode metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "_header": True,
            "merkle_root": self.merkle_root(),
            "n_records": len(self._records),
            "metadata": self.metadata,
        }
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(header, sort_keys=True, default=str) + "\n")
            for record, leaf in zip(self._records, self._leaves):
                f.write(json.dumps({**record, "_leaf": leaf}, sort_keys=True, default=str) + "\n")
        return path

    def submit_onchain(self, chain_cfg: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit the Merkle root on-chain via DecisionLogger.logEpisode.

        Returns the transaction hash on success, ``None`` when the chain
        is not configured or the submission failed. Errors are swallowed
        so simulation loops never block on chain availability.
        """
        if not chain_cfg:
            return None
        try:
            from .eth import log_episode_onchain
            return log_episode_onchain(
                root_hex=self.merkle_root(),
                metadata={**self.metadata, "n_records": len(self._records)},
                chain_cfg=chain_cfg,
            )
        except Exception:
            return None
