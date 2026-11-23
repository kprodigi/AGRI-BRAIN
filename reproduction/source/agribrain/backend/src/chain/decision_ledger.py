"""Per-episode decision ledger with Merkle anchoring.

Every routing decision the simulator emits is appended to a
:class:`DecisionLedger`. The ledger canonicalises each record, computes
a SHA-256 leaf hash, and produces a single 32-byte Merkle root over
the full episode. The root can be committed on-chain via
``log_episode_onchain`` so any individual decision is verifiable via
inclusion proof while only one transaction is paid per episode.

This module is the single per-step write point used by both the HPC simulator
and the FastAPI ``/decide`` endpoint.  Every local decision record contributes
to an off-chain Merkle root.  Only that episode root is eligible for optional
on-chain anchoring, and only when chain submission is explicitly configured.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


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


@dataclass(frozen=True)
class DecisionLedgerArchiveReceipt:
    """Literal identity and Merkle binding for a compressed ledger archive."""

    path: Path
    literal_sha256: str
    literal_bytes: int
    merkle_root: str
    n_records: int


def _json_line(value: Dict[str, Any]) -> str:
    """Preserve the historical DecisionLedger JSONL line representation."""

    return json.dumps(value, sort_keys=True, default=str)


def _jsonl_lines(payload: Dict[str, Any]) -> List[str]:
    return [
        _json_line(payload["header"]),
        *[_json_line(record) for record in payload["records"]],
    ]


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Flush one same-directory temporary file and atomically promote it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"ledger target is not a regular file: {path}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _deterministic_gzip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=output,
        compresslevel=9,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return output.getvalue()


def validate_evidence_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate leaf hashes and the Merkle root, returning a defensive copy."""

    if not isinstance(payload, dict):
        raise ValueError("decision-ledger evidence payload is not an object")
    header = payload.get("header")
    records = payload.get("records")
    if not isinstance(header, dict) or header.get("_header") is not True:
        raise ValueError("decision-ledger evidence header is missing")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("decision-ledger evidence records are invalid")
    if header.get("n_records") != len(records):
        raise ValueError("decision-ledger evidence record count mismatch")
    leaves: List[str] = []
    for index, stored in enumerate(records):
        record = dict(stored)
        leaf = record.pop("_leaf", None)
        if not isinstance(leaf, str) or len(leaf) != 64:
            raise ValueError(f"decision-ledger record {index} has no valid leaf hash")
        try:
            int(leaf, 16)
        except ValueError as exc:
            raise ValueError(
                f"decision-ledger record {index} has a non-hex leaf hash"
            ) from exc
        if _sha256_hex(_canonical_bytes(record)) != leaf:
            raise ValueError(f"decision-ledger record {index} leaf hash mismatch")
        leaves.append(leaf)
    if header.get("merkle_root") != merkle_root_hex(leaves):
        raise ValueError("decision-ledger Merkle root mismatch")
    return deepcopy({"header": header, "records": records})


def read_jsonl_gzip(
    path: Path,
    *,
    expected_literal_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a compressed ledger and verify its literal hash and Merkle binding."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"ledger archive is not a regular non-symlink file: {source}")
    literal = source.read_bytes()
    observed_sha256 = _sha256_hex(literal)
    if (
        expected_literal_sha256 is not None
        and observed_sha256 != expected_literal_sha256
    ):
        raise ValueError("decision-ledger archive SHA-256 mismatch")
    try:
        raw = gzip.decompress(literal).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise ValueError("invalid compressed decision-ledger archive") from exc
    lines = raw.splitlines()
    if not lines or raw[-1:] != "\n" or any(not line for line in lines):
        raise ValueError("compressed decision-ledger JSONL framing is invalid")
    try:
        values = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError("compressed decision-ledger JSONL is invalid") from exc
    return validate_evidence_payload({"header": values[0], "records": values[1:]})


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
        # Snapshot nested values before hashing so later caller mutation cannot
        # make the JSONL payload diverge from the Merkle leaf it claims.
        snapshot = deepcopy(record)
        leaf = _sha256_hex(_canonical_bytes(snapshot))
        self._records.append(snapshot)
        self._leaves.append(leaf)
        return leaf

    def recent_records(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return defensive copies of the latest records in this episode."""
        limit = max(0, int(n))
        if limit == 0:
            return []
        return deepcopy(self._records[-limit:])

    def merkle_root(self) -> str:
        return merkle_root_hex(self._leaves)

    def evidence_payload(self) -> Dict[str, Any]:
        """Return a defensive, Merkle-bound snapshot of this episode ledger."""

        if len(self._records) != len(self._leaves):
            raise RuntimeError("decision-ledger record and leaf counts diverged")
        payload = {
            "header": {
                "_header": True,
                "merkle_root": self.merkle_root(),
                "n_records": len(self._records),
                "metadata": deepcopy(self.metadata),
            },
            "records": [
                {**deepcopy(record), "_leaf": leaf}
                for record, leaf in zip(
                    self._records, self._leaves, strict=True,
                )
            ],
        }
        return validate_evidence_payload(payload)

    def write_jsonl(self, path: Path) -> Path:
        """Write the ledger to a JSONL file with a header line carrying the
        Merkle root and episode metadata. Promotion is crash-safe and atomic;
        the historical platform-native newline bytes and JSON schema are
        unchanged.
        """
        path = Path(path)
        lines = _jsonl_lines(self.evidence_payload())
        encoded = (os.linesep.join(lines) + os.linesep).encode("utf-8")
        _atomic_write_bytes(path, encoded)
        return path

    def write_jsonl_gzip(self, path: Path) -> DecisionLedgerArchiveReceipt:
        """Write lossless, deterministic gzip JSONL for adaptation evidence.

        Numeric values are serialized by the existing DecisionLedger JSON
        representation without rounding.  The gzip header carries no filename
        and uses ``mtime=0``; uncompressed JSONL uses a fixed LF delimiter.
        """

        target = Path(path)
        snapshot = self.evidence_payload()
        raw = ("\n".join(_jsonl_lines(snapshot)) + "\n").encode("utf-8")
        literal = _deterministic_gzip(raw)
        _atomic_write_bytes(target, literal)
        return DecisionLedgerArchiveReceipt(
            path=target,
            literal_sha256=_sha256_hex(literal),
            literal_bytes=len(literal),
            merkle_root=str(snapshot["header"]["merkle_root"]),
            n_records=int(snapshot["header"]["n_records"]),
        )

    def submit_onchain(self, chain_cfg: Optional[Dict[str, Any]]) -> Optional[str]:
        """Submit the Merkle root on-chain via DecisionLogger.logEpisode.

        Returns the transaction hash on success, ``None`` only when the
        chain is not configured (``chain_cfg`` is empty / falsy).

        On configured-but-failing submissions (RPC unreachable, ABI
        mismatch, transaction reverted, receipt status != 1) this
        method now logs the error at WARN and re-raises by default,
        so operators do not silently believe an anchoring happened
        when it did not. Set ``CHAIN_BEST_EFFORT=true`` to restore the
        previous "swallow everything and return None" behaviour for
        long-running simulation loops where anchoring is best-effort.
        """
        if not chain_cfg:
            return None

        import logging as _logging
        import os as _os
        _logger = _logging.getLogger(__name__)
        best_effort = _os.environ.get("CHAIN_BEST_EFFORT", "false").lower() == "true"

        try:
            from .eth import log_episode_onchain
            return log_episode_onchain(
                root_hex=self.merkle_root(),
                metadata={**self.metadata, "n_records": len(self._records)},
                chain_cfg=chain_cfg,
            )
        except Exception as exc:
            if best_effort:
                _logger.warning(
                    "DecisionLedger.submit_onchain failed (best-effort): %s",
                    exc,
                )
                return None
            _logger.error(
                "DecisionLedger.submit_onchain failed: %s. Re-raising; "
                "set CHAIN_BEST_EFFORT=true to swallow.",
                exc,
            )
            raise


_ACTIVE_EPISODE_LEDGER: ContextVar[Optional[DecisionLedger]] = ContextVar(
    "agribrain_active_episode_ledger", default=None
)
_ACTIVE_LEDGER_OUTPUT_DIR: ContextVar[Optional[Path]] = ContextVar(
    "agribrain_active_ledger_output_dir", default=None
)


@contextmanager
def decision_ledger_episode_scope(
    ledger: DecisionLedger,
) -> Iterator[DecisionLedger]:
    """Expose exactly one episode ledger to in-process context tools.

    ContextVar isolation makes nested async/task contexts safe and, unlike a
    module global or newest-file lookup, cannot mix concurrent Slurm arms.
    """
    token = _ACTIVE_EPISODE_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE_EPISODE_LEDGER.reset(token)


def get_active_episode_ledger() -> Optional[DecisionLedger]:
    """Return the ledger in the current execution context, if one is active."""
    return _ACTIVE_EPISODE_LEDGER.get()


@contextmanager
def decision_ledger_output_scope(path: Path) -> Iterator[Path]:
    """Select an audit-output directory without mutating process environment."""
    output_dir = Path(path).resolve()
    token = _ACTIVE_LEDGER_OUTPUT_DIR.set(output_dir)
    try:
        yield output_dir
    finally:
        _ACTIVE_LEDGER_OUTPUT_DIR.reset(token)


def get_active_decision_ledger_output_dir() -> Optional[Path]:
    """Return the current context's audit-output directory, if declared."""
    return _ACTIVE_LEDGER_OUTPUT_DIR.get()
