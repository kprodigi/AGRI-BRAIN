"""
Python wrappers for the AGRI-BRAIN governance smart contracts.

Provides callable integration for the four governance contracts
that complement the DecisionLogger:

    SLCARewards    — reward/slash tokens based on SLCA performance
    AgriDAO        — propose/vote/execute governance proposals
    PolicyStore    — on-chain key-value store for policy parameters
    AgentRegistry  — register agents and toggle active status

Each wrapper follows the same pattern as eth.py: build a transaction,
sign with the configured private key, and wait for the receipt.
All functions are best-effort (return None when chain is not configured).
"""
from __future__ import annotations

import logging
from typing import Optional

from web3 import Web3

from .eth import _client, _checksum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal ABIs (only the functions we call)
# ---------------------------------------------------------------------------

SLCA_REWARDS_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "reward",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "from", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "slash",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "to", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "Rewarded",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "from", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "Slashed",
        "type": "event",
    },
]

AGRI_DAO_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "text", "type": "string"},
        ],
        "name": "propose",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "id", "type": "uint256"},
            {"internalType": "bool", "name": "support", "type": "bool"},
        ],
        "name": "vote",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "id", "type": "uint256"},
        ],
        "name": "execute",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "id", "type": "uint256"},
            {"indexed": False, "internalType": "address", "name": "proposer", "type": "address"},
            {"indexed": False, "internalType": "string", "name": "text", "type": "string"},
        ],
        "name": "Proposed",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "id", "type": "uint256"},
            {"indexed": False, "internalType": "address", "name": "voter", "type": "address"},
            {"indexed": False, "internalType": "bool", "name": "support", "type": "bool"},
        ],
        "name": "Voted",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "id", "type": "uint256"},
        ],
        "name": "Executed",
        "type": "event",
    },
]

POLICY_STORE_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "key", "type": "bytes32"},
            {"internalType": "uint256", "name": "value", "type": "uint256"},
        ],
        "name": "setPolicy",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "key", "type": "bytes32"},
        ],
        "name": "getPolicy",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "key", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "oldValue", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "newValue", "type": "uint256"},
        ],
        "name": "PolicyChanged",
        "type": "event",
    },
]

AGENT_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "id", "type": "bytes32"},
            {"internalType": "string", "name": "role", "type": "string"},
            {"internalType": "string", "name": "meta", "type": "string"},
        ],
        "name": "register",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bool", "name": "on", "type": "bool"},
        ],
        "name": "setActive",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "agent", "type": "address"},
            {"indexed": False, "internalType": "bytes32", "name": "id", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "role", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "meta", "type": "string"},
        ],
        "name": "Registered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "agent", "type": "address"},
            {"indexed": False, "internalType": "bool", "name": "active", "type": "bool"},
        ],
        "name": "Status",
        "type": "event",
    },
]

# ---------------------------------------------------------------------------
# Contract name -> (ABI, addresses key)
# ---------------------------------------------------------------------------
_CONTRACTS = {
    "SLCARewards":    (SLCA_REWARDS_ABI, "SLCARewards"),
    "AgriDAO":        (AGRI_DAO_ABI, "AgriDAO"),
    "PolicyStore":    (POLICY_STORE_ABI, "PolicyStore"),
    "AgentRegistry":  (AGENT_REGISTRY_ABI, "AgentRegistry"),
}


def _get_contract(chain_cfg: dict, name: str):
    """Return (w3, acct, contract) or None if not configured."""
    if not chain_cfg or not chain_cfg.get("rpc") or not chain_cfg.get("private_key"):
        return None
    addrs = chain_cfg.get("addresses") or {}
    addr = addrs.get(name)
    if not addr:
        return None
    abi, _ = _CONTRACTS[name]
    w3, acct = _client(chain_cfg)
    contract = w3.eth.contract(address=_checksum(addr), abi=abi)
    return w3, acct, contract


def _send_tx(w3, acct, tx_fn, chain_cfg: dict) -> Optional[str]:
    """Build, sign, and send a transaction. Returns tx hash hex or None."""
    tx = tx_fn.build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 300_000,
        "maxFeePerGas": w3.to_wei("2", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
        "chainId": int(chain_cfg.get("chain_id", 31337)),
    })
    signed = acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    rcpt = w3.eth.wait_for_transaction_receipt(txh)
    return rcpt.transactionHash.hex()


# ---------------------------------------------------------------------------
# SLCARewards
# ---------------------------------------------------------------------------

def slca_reward(to_address: str, amount: int, chain_cfg: dict) -> Optional[str]:
    """Reward an agent with SLCA tokens. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "SLCARewards")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.reward(_checksum(to_address), amount), chain_cfg)


def slca_slash(from_address: str, amount: int, chain_cfg: dict) -> Optional[str]:
    """Slash SLCA tokens from an agent. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "SLCARewards")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.slash(_checksum(from_address), amount), chain_cfg)


# ---------------------------------------------------------------------------
# AgriDAO
# ---------------------------------------------------------------------------

def dao_propose(text: str, chain_cfg: dict) -> Optional[str]:
    """Submit a governance proposal. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgriDAO")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.propose(text), chain_cfg)


def dao_vote(proposal_id: int, support: bool, chain_cfg: dict) -> Optional[str]:
    """Vote on a governance proposal. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgriDAO")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.vote(proposal_id, support), chain_cfg)


def dao_execute(proposal_id: int, chain_cfg: dict) -> Optional[str]:
    """Execute an approved governance proposal. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgriDAO")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.execute(proposal_id), chain_cfg)


# ---------------------------------------------------------------------------
# PolicyStore
# ---------------------------------------------------------------------------

def _policy_key(name: str) -> bytes:
    """Convert a policy parameter name to a bytes32 key."""
    return Web3.keccak(text=name)


def policy_store_set(key_name: str, value: int, chain_cfg: dict) -> Optional[str]:
    """Store a policy parameter on-chain. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "PolicyStore")
    if result is None:
        return None
    w3, acct, contract = result
    key = _policy_key(key_name)
    return _send_tx(w3, acct, contract.functions.setPolicy(key, value), chain_cfg)


def policy_store_get(key_name: str, chain_cfg: dict) -> Optional[int]:
    """Read a policy parameter from on-chain storage. Returns value or None."""
    result = _get_contract(chain_cfg, "PolicyStore")
    if result is None:
        return None
    _, _, contract = result
    key = _policy_key(key_name)
    return contract.functions.getPolicy(key).call()


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------

def agent_register(agent_id: str, role: str, meta: str, chain_cfg: dict) -> Optional[str]:
    """Register an agent on-chain. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgentRegistry")
    if result is None:
        return None
    w3, acct, contract = result
    id_bytes = Web3.keccak(text=agent_id)
    return _send_tx(w3, acct, contract.functions.register(id_bytes, role, meta), chain_cfg)


def agent_set_active(active: bool, chain_cfg: dict) -> Optional[str]:
    """Toggle an agent's active status on-chain. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgentRegistry")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.setActive(active), chain_cfg)
