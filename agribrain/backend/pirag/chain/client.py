
import os
from typing import Optional
from src.settings import SETTINGS
try:
    from web3 import Web3
except Exception:
    Web3 = None

# ProvenanceRegistry ABI — matches ProvenanceRegistry.sol anchor() function
ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
            {"internalType": "string", "name": "decisionId", "type": "string"},
        ],
        "name": "anchor",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "decisionId", "type": "string"},
            {"indexed": True, "internalType": "address", "name": "submitter", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "name": "ProvenanceAnchored",
        "type": "event",
    },
]

CONTRACT_ADDRESS = os.getenv("PROVENANCE_ADDR", "")


def _get_chain_cfg() -> dict:
    """Read chain config from governance module (canonical source)."""
    try:
        from src.routers.governance import CHAIN
        return dict(CHAIN)
    except ImportError:
        return {}


def anchor_root(root_hex: str, policy_uri: str = "") -> Optional[str]:
    if Web3 is None:
        return None
    cfg = _get_chain_cfg()
    addr = CONTRACT_ADDRESS or (cfg.get("addresses") or {}).get("ProvenanceRegistry", "")
    if not addr:
        return None
    rpc = cfg.get("rpc") or os.getenv("CHAIN_RPC", "http://localhost:8545")
    privkey = cfg.get("private_key") or os.getenv("CHAIN_PRIVKEY", "")
    if SETTINGS.chain_require_privkey and not privkey:
        return None
    if not privkey:
        return None
    w3 = Web3(Web3.HTTPProvider(rpc))
    acct = w3.eth.account.from_key(privkey)
    contract = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ABI)
    normalized_root = root_hex[2:] if root_hex.startswith("0x") else root_hex
    tx = contract.functions.anchor(bytes.fromhex(normalized_root), policy_uri).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 500000,
        "maxFeePerGas": w3.to_wei("2", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
        "chainId": int(cfg.get("chain_id", w3.eth.chain_id)),
    })
    signed = acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    rcpt = w3.eth.wait_for_transaction_receipt(txh, timeout=30)
    return rcpt.transactionHash.hex()
