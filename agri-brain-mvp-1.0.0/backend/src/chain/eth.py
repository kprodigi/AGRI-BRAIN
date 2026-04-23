# backend/src/chain/eth.py
from __future__ import annotations
from typing import Optional
from web3 import Web3
from eth_account import Account

# Minimal ABI for our single function + event
ABI = [
    {"anonymous": False, "inputs": [
        {"indexed": True,  "internalType": "bytes32", "name": "id", "type": "bytes32"},
        {"indexed": False, "internalType": "uint256","name": "ts", "type": "uint256"},
        {"indexed": False, "internalType": "string", "name": "agent", "type": "string"},
        {"indexed": False, "internalType": "string", "name": "role", "type": "string"},
        {"indexed": False, "internalType": "string", "name": "action", "type": "string"},
        {"indexed": False, "internalType": "uint256","name": "slca_milli", "type": "uint256"},
        {"indexed": False, "internalType": "uint256","name": "carbon_milli", "type": "uint256"},
        {"indexed": False, "internalType": "string", "name": "note", "type": "string"}
    ], "name": "DecisionLogged", "type": "event"},
    {"inputs": [
        {"internalType":"uint256","name":"ts","type":"uint256"},
        {"internalType":"string","name":"agent","type":"string"},
        {"internalType":"string","name":"role","type":"string"},
        {"internalType":"string","name":"action","type":"string"},
        {"internalType":"uint256","name":"slca_milli","type":"uint256"},
        {"internalType":"uint256","name":"carbon_milli","type":"uint256"},
        {"internalType":"string","name":"note","type":"string"}
    ], "name":"logDecision", "outputs":[{"internalType":"bytes32","name":"id","type":"bytes32"}],
     "stateMutability":"nonpayable","type":"function"},
    {"anonymous": False, "inputs": [
        {"indexed": True,  "internalType": "bytes32", "name": "root", "type": "bytes32"},
        {"indexed": False, "internalType": "uint256","name": "ts", "type": "uint256"},
        {"indexed": False, "internalType": "string", "name": "mode", "type": "string"},
        {"indexed": False, "internalType": "string", "name": "scenario", "type": "string"},
        {"indexed": False, "internalType": "uint256","name": "seed", "type": "uint256"},
        {"indexed": False, "internalType": "uint256","name": "n_records", "type": "uint256"},
        {"indexed": False, "internalType": "string", "name": "note", "type": "string"}
    ], "name": "EpisodeLogged", "type": "event"},
    {"inputs": [
        {"internalType":"bytes32","name":"root","type":"bytes32"},
        {"internalType":"uint256","name":"ts","type":"uint256"},
        {"internalType":"string","name":"mode","type":"string"},
        {"internalType":"string","name":"scenario","type":"string"},
        {"internalType":"uint256","name":"seed","type":"uint256"},
        {"internalType":"uint256","name":"n_records","type":"uint256"},
        {"internalType":"string","name":"note","type":"string"}
    ], "name":"logEpisode", "outputs":[],
     "stateMutability":"nonpayable","type":"function"}
]

def _checksum(addr: str) -> str:
    return Web3.to_checksum_address(addr)

def _client(cfg: dict):
    w3 = Web3(Web3.HTTPProvider(cfg["rpc"]))
    acct = Account.from_key(cfg["private_key"]) if cfg.get("private_key") else None
    return w3, acct

def _contract(w3: Web3, addr: str):
    return w3.eth.contract(address=_checksum(addr), abi=ABI)


def _fee_params(w3: Web3) -> tuple[int, int]:
    """Use dynamic EIP-1559 fees when available; fallback to conservative defaults."""
    try:
        latest = w3.eth.get_block("latest")
        base_fee = int(latest.get("baseFeePerGas") or 0)
    except Exception:
        base_fee = 0
    try:
        priority = int(w3.eth.max_priority_fee)
    except Exception:
        priority = int(w3.to_wei("1", "gwei"))
    if base_fee > 0:
        max_fee = int(base_fee * 2 + priority)
    else:
        max_fee = int(w3.to_wei("2", "gwei"))
    return max_fee, priority


def log_decision_onchain(memo: dict, chain_cfg: dict) -> Optional[str]:
    """Send tx to DecisionLogger if configured. Returns tx hash hex or None."""
    if not chain_cfg: return None
    addrs = chain_cfg.get("addresses") or {}
    dl = addrs.get("DecisionLogger")
    if not (chain_cfg.get("rpc") and dl and chain_cfg.get("private_key")):
        return None  # not configured; silently no-op

    w3, acct = _client(chain_cfg)
    c = _contract(w3, dl)
    max_fee, priority_fee = _fee_params(w3)
    tx = c.functions.logDecision(
        int(memo.get("ts", 0)),
        str(memo.get("agent", "")),
        str(memo.get("role", "")),
        str(memo.get("action", "")),
        int(round(float(memo.get("slca_score", 0)) * 1000)),
        int(round(float(memo.get("carbon_kg", 0)) * 1000)),
        str(memo.get("note", "")),
    ).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 300000,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
        "chainId": int(chain_cfg.get("chain_id", 31337)),
    })
    signed = acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    rcpt = w3.eth.wait_for_transaction_receipt(txh)
    if int(rcpt.get("status", 1)) != 1:
        raise RuntimeError("DecisionLogger transaction reverted")
    tx_hash = rcpt.get("transactionHash")
    return tx_hash.hex() if tx_hash is not None else None

def log_episode_onchain(root_hex: str, metadata: dict, chain_cfg: dict) -> Optional[str]:
    """Anchor a per-episode Merkle root via DecisionLogger.logEpisode.

    ``root_hex`` is the SHA-256 binary-Merkle root produced by
    :class:`backend.src.chain.decision_ledger.DecisionLedger`. ``metadata``
    must carry ``mode``, ``scenario``, ``seed``, and ``n_records``; the
    optional ``note`` is a free-form string.
    """
    if not chain_cfg:
        return None
    addrs = chain_cfg.get("addresses") or {}
    dl = addrs.get("DecisionLogger")
    if not (chain_cfg.get("rpc") and dl and chain_cfg.get("private_key")):
        return None

    w3, acct = _client(chain_cfg)
    c = _contract(w3, dl)
    max_fee, priority_fee = _fee_params(w3)

    root_clean = root_hex[2:] if root_hex.startswith("0x") else root_hex
    root_bytes = bytes.fromhex(root_clean)
    if len(root_bytes) != 32:
        raise ValueError(f"merkle root must be 32 bytes, got {len(root_bytes)}")

    tx = c.functions.logEpisode(
        root_bytes,
        int(metadata.get("ts", 0)),
        str(metadata.get("mode", "")),
        str(metadata.get("scenario", "")),
        int(metadata.get("seed", 0)),
        int(metadata.get("n_records", 0)),
        str(metadata.get("note", "")),
    ).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 200000,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
        "chainId": int(chain_cfg.get("chain_id", 31337)),
    })
    signed = acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    rcpt = w3.eth.wait_for_transaction_receipt(txh)
    if int(rcpt.get("status", 1)) != 1:
        raise RuntimeError("DecisionLogger.logEpisode transaction reverted")
    tx_hash = rcpt.get("transactionHash")
    return tx_hash.hex() if tx_hash is not None else None


def fetch_recent_decisions(chain_cfg: dict, from_block: int = 0, to_block: str | int = "latest"):
    """Return decoded DecisionLogged events (list of dicts)."""
    addrs = chain_cfg.get("addresses") or {}
    dl = addrs.get("DecisionLogger")
    if not (chain_cfg.get("rpc") and dl):
        return []
    w3, _ = _client(chain_cfg)
    c = _contract(w3, dl)
    # for dev: if from_block==0, try a reasonable recent window
    if from_block == 0:
        latest = w3.eth.block_number
        from_block = max(0, latest - 5000)
    logs = c.events.DecisionLogged().get_logs(fromBlock=from_block, toBlock=to_block)
    out = []
    for ev in logs:
        args = ev["args"]
        out.append({
            "id": args["id"].hex(),
            "ts": int(args["ts"]),
            "agent": args["agent"],
            "role": args["role"],
            "action": args["action"],
            "slca_score": float(args["slca_milli"]) / 1000.0,
            "carbon_kg": float(args["carbon_milli"]) / 1000.0,
            "reason": args["note"],
            "tx_hash": ev["transactionHash"].hex(),
            "block": ev["blockNumber"],
        })
    return out
