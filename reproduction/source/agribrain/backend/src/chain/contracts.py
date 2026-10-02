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
            {"internalType": "string", "name": "description", "type": "string"},
            {"internalType": "bytes32", "name": "policyKey", "type": "bytes32"},
            {"internalType": "uint256", "name": "policyValue", "type": "uint256"},
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
        "inputs": [{"internalType": "uint256", "name": "id", "type": "uint256"}],
        "name": "finalize",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "id", "type": "uint256"}],
        "name": "queue",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "id", "type": "uint256"}],
        "name": "execute",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "id", "type": "uint256"},
            {"indexed": False, "internalType": "address", "name": "proposer", "type": "address"},
            {"indexed": False, "internalType": "string", "name": "description", "type": "string"},
            {"indexed": False, "internalType": "bytes32", "name": "policyKey", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "policyValue", "type": "uint256"},
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
    # Matrix-shaped policy parameters (Theta, Theta_context). Values are
    # milli-scaled int256 (multiply float entries by 1000 before sending).
    # See PolicyStore.sol for the full schema.
    {
        "inputs": [
            {"internalType": "bytes32", "name": "key", "type": "bytes32"},
            {"internalType": "uint256", "name": "rows", "type": "uint256"},
            {"internalType": "uint256", "name": "cols", "type": "uint256"},
            {"internalType": "int256[]", "name": "valuesMilli", "type": "int256[]"},
        ],
        "name": "setPolicyMatrix",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "key", "type": "bytes32"},
        ],
        "name": "getPolicyMatrix",
        "outputs": [
            {"internalType": "uint256", "name": "rows", "type": "uint256"},
            {"internalType": "uint256", "name": "cols", "type": "uint256"},
            {"internalType": "int256[]", "name": "valuesMilli", "type": "int256[]"},
            {"internalType": "uint256", "name": "version", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "key", "type": "bytes32"},
            {"indexed": True, "internalType": "uint256", "name": "version", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "rows", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "cols", "type": "uint256"},
        ],
        "name": "PolicyMatrixChanged",
        "type": "event",
    },
]

AGENT_REGISTRY_ABI = [
    # ownerRegister: bootstrap path used during deployment. Only the
    # deployer address (contract owner) can call it. Solidity signature:
    #   function ownerRegister(address account, bytes32 id, string role, string meta)
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "bytes32", "name": "id", "type": "bytes32"},
            {"internalType": "string", "name": "role", "type": "string"},
            {"internalType": "string", "name": "meta", "type": "string"},
        ],
        "name": "ownerRegister",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    # sponsorRegister: production path. An existing active agent with
    # an admin role admits a new agent. Solidity signature:
    #   function sponsorRegister(address account, bytes32 id, string role, string meta)
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "bytes32", "name": "id", "type": "bytes32"},
            {"internalType": "string", "name": "role", "type": "string"},
            {"internalType": "string", "name": "meta", "type": "string"},
        ],
        "name": "sponsorRegister",
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


def _get_contract_readonly(chain_cfg: dict, name: str):
    """Return (w3, contract) for read-only calls — no private key needed."""
    if not chain_cfg or not chain_cfg.get("rpc"):
        return None
    addrs = chain_cfg.get("addresses") or {}
    addr = addrs.get(name)
    if not addr:
        return None
    abi, _ = _CONTRACTS[name]
    w3 = Web3(Web3.HTTPProvider(chain_cfg["rpc"]))
    contract = w3.eth.contract(address=_checksum(addr), abi=abi)
    return w3, contract


def _send_tx(w3, acct, tx_fn, chain_cfg: dict) -> Optional[str]:
    """Build, sign, and send a transaction. Returns tx hash hex or None."""
    max_fee, priority_fee = _fee_params(w3)
    tx = tx_fn.build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "gas": 300_000,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
        "chainId": int(chain_cfg.get("chain_id", 31337)),
    })
    signed = acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.raw_transaction)
    rcpt = w3.eth.wait_for_transaction_receipt(txh)
    if int(rcpt.get("status", 1)) != 1:
        raise RuntimeError("On-chain transaction reverted")
    tx_hash = rcpt.get("transactionHash")
    return tx_hash.hex() if tx_hash is not None else None


# Fee-params resolution lives in eth.py with structured WARN logging
# on RPC-level failures. The pre-2026-05 duplicate copy below used
# bare ``except Exception`` and silent fallback -- the exact regression
# the chain/README.md change-log called out as fixed. Re-import the
# canonical implementation so a future fix in one place is not
# silently shadowed by a stale duplicate.
from .eth import _fee_params  # noqa: F401  (re-exported for callers)


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

def dao_propose(description: str, policy_key: bytes, policy_value: int,
                chain_cfg: dict) -> Optional[str]:
    """Submit a governance proposal. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgriDAO")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct,
                    contract.functions.propose(description, policy_key, policy_value),
                    chain_cfg)


def dao_vote(proposal_id: int, support: bool, chain_cfg: dict) -> Optional[str]:
    """Vote on a governance proposal. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgriDAO")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.vote(proposal_id, support), chain_cfg)


def dao_finalize(proposal_id: int, chain_cfg: dict) -> Optional[str]:
    """Finalize voting outcome for a governance proposal."""
    result = _get_contract(chain_cfg, "AgriDAO")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.finalize(proposal_id), chain_cfg)


def dao_queue(proposal_id: int, chain_cfg: dict) -> Optional[str]:
    """Queue a succeeded governance proposal for execution."""
    result = _get_contract(chain_cfg, "AgriDAO")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.queue(proposal_id), chain_cfg)


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
    """Read a policy parameter from on-chain storage. Returns value or None.

    Uses read-only contract access — no private key required.
    """
    result = _get_contract_readonly(chain_cfg, "PolicyStore")
    if result is None:
        return None
    _, contract = result
    key = _policy_key(key_name)
    return contract.functions.getPolicy(key).call()


# ---------------------------------------------------------------------------
# PolicyStore — matrix-shaped parameters (Theta, Theta_context)
# ---------------------------------------------------------------------------

def _to_milli_int_list(matrix) -> list[int]:
    """Flatten a 2-D float matrix row-major and milli-scale to int.

    Accepts a numpy ndarray, a list-of-lists, or any iterable of
    iterables. The on-chain contract stores int256 values scaled by
    1000, so 0.50 -> 500, -0.80 -> -800, +1.234 -> 1234. Anything that
    cannot be coerced to a Python int after scaling raises ValueError.
    """
    out: list[int] = []
    for row in matrix:
        for cell in row:
            scaled = round(float(cell) * 1000.0)
            out.append(int(scaled))
    return out


def policy_store_set_matrix(
    key_name: str,
    matrix,
    chain_cfg: dict,
) -> Optional[str]:
    """Anchor a matrix-shaped policy parameter (Theta, Theta_context, ...) on-chain.

    Parameters
    ----------
    key_name :
        Logical name of the matrix; will be hashed via keccak256
        ("THETA", "THETA_CONTEXT", ...).
    matrix :
        2-D iterable of floats. Row-major. Cell values must satisfy
        the per-key max-abs bound declared in PolicyStore.sol.
    chain_cfg :
        Standard chain config dict (rpc, addresses, private_key).

    Returns
    -------
    tx hash on success, None when the chain is not configured.
    """
    result = _get_contract(chain_cfg, "PolicyStore")
    if result is None:
        return None
    w3, acct, contract = result
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        raise ValueError("policy_store_set_matrix: empty matrix")
    flat = _to_milli_int_list(matrix)
    if len(flat) != rows * cols:
        raise ValueError(f"policy_store_set_matrix: ragged matrix {rows}x{cols} vs {len(flat)} cells")
    key = _policy_key(key_name)
    return _send_tx(
        w3, acct,
        contract.functions.setPolicyMatrix(key, rows, cols, flat),
        chain_cfg,
    )


def policy_store_get_matrix(
    key_name: str,
    chain_cfg: dict,
) -> Optional[dict]:
    """Read a matrix-shaped policy parameter from chain.

    Returns a dict ``{rows, cols, values, version}`` where ``values``
    is a row-major list of floats (milli-scale already undone). Returns
    None when the chain is not configured.
    """
    result = _get_contract_readonly(chain_cfg, "PolicyStore")
    if result is None:
        return None
    _, contract = result
    key = _policy_key(key_name)
    rows, cols, vals_milli, version = contract.functions.getPolicyMatrix(key).call()
    return {
        "rows": int(rows),
        "cols": int(cols),
        "values": [float(v) / 1000.0 for v in vals_milli],
        "version": int(version),
    }


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------

def agent_register(
    account: str,
    agent_id: str,
    role: str,
    meta: str,
    chain_cfg: dict,
    *,
    method: str = "sponsorRegister",
) -> Optional[str]:
    """Register an agent on-chain. Returns tx hash or None.

    The Solidity contract exposes two registration paths:

    - ``sponsorRegister(address, bytes32, string, string)`` — production
      path; the calling EOA must already be a registered agent with an
      admin-tier role (``adminRole[role] == True``). This is the
      default and what app integrations should use post-bootstrap.
    - ``ownerRegister(address, bytes32, string, string)`` — bootstrap
      path; only the contract owner (the deployer EOA stored on
      construction) can call it. Used during initial cooperative
      seeding before any sponsor with admin role exists. Pass
      ``method="ownerRegister"`` to invoke this path.

    The previous version of this wrapper called a ``register(bytes32,
    string, string)`` 3-arg function that does not exist on the
    deployed AgentRegistry.sol -- any chain submission would have
    reverted with "function selector not found". Fixed in 2026-05.
    """
    # Validate the method first so a typo fails loudly even when the
    # chain is not configured (i.e. _get_contract would return None
    # for an unrelated reason). This makes the contract guarantee
    # uniform across "chain configured" and "chain not configured"
    # call paths.
    if method not in ("sponsorRegister", "ownerRegister"):
        raise ValueError(
            f"agent_register: method must be 'sponsorRegister' or "
            f"'ownerRegister', got {method!r}"
        )
    result = _get_contract(chain_cfg, "AgentRegistry")
    if result is None:
        return None
    w3, acct, contract = result
    id_bytes = Web3.keccak(text=agent_id)
    fn = getattr(contract.functions, method)
    return _send_tx(w3, acct, fn(account, id_bytes, role, meta), chain_cfg)


def agent_set_active(active: bool, chain_cfg: dict) -> Optional[str]:
    """Toggle an agent's active status on-chain. Returns tx hash or None."""
    result = _get_contract(chain_cfg, "AgentRegistry")
    if result is None:
        return None
    w3, acct, contract = result
    return _send_tx(w3, acct, contract.functions.setActive(active), chain_cfg)
