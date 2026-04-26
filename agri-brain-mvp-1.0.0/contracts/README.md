# AGRI-BRAIN smart contracts

This directory contains the Solidity smart contracts and Hardhat test
suite for the AGRI-BRAIN audit-trail prototype.

## Honest scope

**The published runs do not deploy these contracts to a permissioned
EVM.** The only configured network is `localhost: 127.0.0.1:8545`
(see `hardhat/hardhat.config.cjs`); the deployment artefacts live at
`hardhat/deployed-addresses.localhost.json`. There is no Besu QBFT or
IBFT-2.0 genesis configuration in this repo. The README's "permissioned
EVM audit trail" wording has been corrected to reflect what is
actually deployed: **off-chain Merkle audit ledger with optional
on-chain anchoring on a localhost Hardhat chain**.

The contracts are research code intended to demonstrate the design.
They are not audited for production deployment and the prototype
warnings in their NatSpec headers should be respected.

## Access-control posture (post 2026-04 audit)

| Contract              | Pre-2026-04 access     | Current access                                    |
|-----------------------|------------------------|---------------------------------------------------|
| `AgentRegistry`       | role registry (real)   | unchanged                                         |
| `AgriDAO`             | proposer / voter roles | unchanged; hand-rolled `_LocalReentrancyGuard` ok |
| `DecisionLogger`      | single-key Ownable     | single-key Ownable, prototype warning             |
| `PolicyStore`         | single-key Ownable     | single-key Ownable, prototype warning             |
| `ProvenanceRegistry`  | single-key Ownable     | single-key Ownable, prototype warning             |
| `SLCARewards`         | single-key Ownable     | **role-based** (ADMIN/REWARDER/SLASHER) — fixed   |

`SLCARewards` was the most exposed (single key could mint unbounded).
The 2026-04 hardening replaces the hand-rolled Ownable with a minimal
role-based access control pattern (ADMIN_ROLE manages role grants;
REWARDER_ROLE can mint; SLASHER_ROLE can deduct) and exposes
`grantRole`/`revokeRole`/`hasRole` so the on-chain governance
contract (AgriDAO) can be granted REWARDER_ROLE in production. Tests
in `test/SLCARewards.test.js` cover both the new role flow and the
legacy reward/slash paths.

The remaining Ownable contracts retain that pattern in the prototype;
their NatSpec headers carry the warning. Replacing them with full
OpenZeppelin `AccessControl` requires adding OZ as a dependency and
re-running the test matrix; that is queued as a separate hardening
task.

## On-chain anchoring posture

`backend/src/chain/decision_ledger.py` produces a per-episode Merkle
root. The publish path is gated by `CHAIN_SUBMIT=1` and the chain
config is supplied via `CHAIN_CFG_JSON`. The 2026-04 audit found
that `submit_onchain` previously swallowed every exception silently —
operators believed an anchoring had happened when it had not. The
fixed implementation **logs at WARN/ERROR and re-raises by default**;
set `CHAIN_BEST_EFFORT=true` to restore the previous swallow-and-
return-None behaviour for long-running simulation loops where
anchoring is best-effort. The simulator (`mvp/simulation/generate_results.py`)
defaults to `CHAIN_BEST_EFFORT=true` for HPC runs but records the
outcome in `decision_ledger_tx_status` so reviewers can count how
many submissions actually landed.

There is no verifying reader for anchored roots. A future hardening
should add a `verify_anchored_root.py` that reads the on-chain root
and compares against the local ledger.

## Production checklist (not satisfied today)

- [ ] Hyperledger Besu QBFT / IBFT-2.0 genesis config committed.
- [ ] OpenZeppelin `AccessControl` for the four Ownable contracts.
- [ ] OpenZeppelin `ReentrancyGuard` import in `AgriDAO.sol`
      (hand-rolled `_LocalReentrancyGuard` is functionally equivalent
      but reviewers expect the canonical import).
- [ ] Per-tx anchoring receipt verifier
      (`scripts/verify_anchored_root.py`).
- [ ] CI workflow that compiles and runs Solidity tests against a
      Besu network, not just localhost Hardhat.

When the manuscript text says "permissioned EVM audit trail",
substitute "off-chain Merkle audit ledger with optional anchoring on
a localhost Hardhat chain (production deployment to Besu QBFT is
flagged as future work)" until the checklist above is satisfied.
