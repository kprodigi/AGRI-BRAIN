# Optional EVM research prototypes

This directory contains Solidity prototypes and a Hardhat test suite. They are
optional development tooling: the locked AGRI-BRAIN publication benchmark uses
local Merkle commitments and sets `CHAIN_SUBMIT=0`.

## Verified scope

The repository verifies compilation and contract behavior only in the local
Hardhat test environment. The tests exercise role grants and revocation,
proposal lifecycle logic, bounded policy storage, decision-record events, local
Merkle-root storage, and the reward/slash bookkeeping API.

No public, consortium, or production network was used to validate these
contracts. The presence of a configurable `permissioned` Hardhat network entry
is a template, not evidence of compatibility with Besu, Quorum, Geth, or any
particular deployment. This code has not received a production security audit
and makes no regulatory, legal, interoperability, immutability, or operational
readiness claim.

## Local test

From `agribrain/contracts/hardhat`:

```bash
npm ci
npx hardhat test
```

The local configuration exposes:

- `AgentRegistry`: prototype address-to-role records;
- `AgriDAO`: prototype proposal and voting lifecycle;
- `PolicyStore`: bounded scalar and matrix storage;
- `DecisionLogger`: optional decision and episode-root event records;
- `ProvenanceRegistry`: optional Merkle-root records; and
- `SLCARewards`: prototype integer reward/slash bookkeeping.

## Optional local anchoring demonstration

`agribrain/backend/src/chain/decision_ledger.py` can submit an episode Merkle
root only when `CHAIN_SUBMIT=1` and a matching local chain configuration is
provided. This is separate from publication evidence. A transaction receipt
shows only that the configured development chain accepted a call; it does not
verify the underlying scientific result, expose Merkle inclusion paths, or
establish real-world deployment.

Keep `CHAIN_SUBMIT=0` for the publication benchmark. Any experiment that enables
submission must be identified separately as an optional local demonstration.
