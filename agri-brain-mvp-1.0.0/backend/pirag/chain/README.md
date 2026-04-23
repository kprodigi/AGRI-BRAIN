# pirag.chain

Thin Python client for the on-chain ProvenanceRegistry.

The canonical Solidity source is under
``agri-brain-mvp-1.0.0/contracts/hardhat/contracts/ProvenanceRegistry.sol``.
This package only holds the Python wrapper; do not add a ``.sol``
source here. The ABI baked into ``client.py`` matches the Hardhat
contract (``anchor(bytes32 merkleRoot, string decisionId)``,
``ProvenanceAnchored(merkleRoot, decisionId, submitter, timestamp)``).

A second Solidity file used to live here that encoded a different
contract (allowlist + rate limit, ``RootAnchored`` event). It was never
deployed by any script and was deleted because its presence confused
operators into deploying the wrong bytecode.

Environment variables read by ``client.py``:

- ``CHAIN_RPC`` or ``ANCHOR_RPC_URL`` — HTTP RPC endpoint
- ``CHAIN_PROVENANCE_ADDRESS`` or ``ANCHOR_REGISTRY_ADDR`` — deployed registry address
- ``CHAIN_PRIVKEY`` or ``ANCHOR_ACCOUNT_KEY`` — signing key (fallback when POST /chain/config has not supplied one)
