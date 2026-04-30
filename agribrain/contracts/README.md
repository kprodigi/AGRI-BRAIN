# AGRI-BRAIN smart contracts

This directory contains the Solidity smart contracts and Hardhat test
suite for the AGRI-BRAIN audit-trail prototype.

## Scope

These contracts target a **permissioned EVM consortium** — the same
deployment posture as the manuscript's §3.15 / §4.13 framing. The
Hardhat config now ships three networks:

* `localhost` — single-node Hardhat for the dev quickstart in
  `HOW_TO_RUN.md` §7. Used by the test matrix.
* `hardhat`   — in-process VM for the unit tests.
* `permissioned` — template that picks up
  `PERMISSIONED_RPC_URL`, `PERMISSIONED_CHAIN_ID`,
  `PERMISSIONED_PRIVKEYS`, `EXTRA_LOGGERS`, and `EXTRA_ANCHORERS`
  from the environment. Deploys cleanly against Hyperledger Besu
  QBFT / IBFT-2.0, Quorum, or a Geth Clique consortium. The deploy
  script (`scripts/deploy.js`) grants `LOGGER_ROLE` /
  `ANCHORER_ROLE` to every address in the EXTRA_* lists in the same
  transaction batch as deployment, so the consortium validator set
  is on-chain from block 1.

The contracts are research code; production deployments should run
the Slither analysis and the rest of the production checklist below.

## Access-control posture (post 2026-04 hardening)

| Contract              | Access model                                                        |
|-----------------------|---------------------------------------------------------------------|
| `AgentRegistry`       | role registry (proposer / voter membership)                         |
| `AgriDAO`             | proposer / voter roles + Pending → Active lifecycle (VOTING_DELAY)  |
| `DecisionLogger`      | **role-based** (ADMIN_ROLE / LOGGER_ROLE) with `setAuthorized` shim |
| `PolicyStore`         | owner + AgriDAO authorized writer                                   |
| `ProvenanceRegistry`  | **role-based** (ADMIN_ROLE / ANCHORER_ROLE) with legacy `onlyOwner` |
| `SLCARewards`         | **role-based** (ADMIN/REWARDER/SLASHER)                             |

The 2026-04 cleanup's "single-key Ownable" finding for `DecisionLogger`
and `ProvenanceRegistry` is closed: both contracts now expose
`grantRole`/`revokeRole`/`hasRole` mirrored on the SLCARewards
pattern. The deployer is granted ADMIN + functional roles at
construction so existing operational scripts keep working; additional
keys (per-agent service accounts, the AgriDAO contract) are granted
through `EXTRA_LOGGERS` / `EXTRA_ANCHORERS` at deploy time. The
legacy `setAuthorized(addr, allowed)` and `onlyOwner` surfaces are
retained as thin shims over the role layer so that backend chain
wrappers and existing scripts work unchanged while every state
change is visible on-chain through `RoleGranted` / `RoleRevoked`
events.

Tests:

* `test/DecisionLogger.test.js` covers role grants, revocation, the
  `setAuthorized` shim, and unauthorized-caller revert.
* `test/ProvenanceRegistry.test.js` covers ADMIN-granted ANCHORER
  delegation and the immutable append-only audit trail.
* `test/SLCARewards.test.js` covers ADMIN-mediated role grants and
  the reward / slash paths.

## On-chain anchoring posture

`backend/src/chain/decision_ledger.py` produces a per-episode Merkle
root. The publish path is gated by `CHAIN_SUBMIT=1` and the chain
config is supplied via `CHAIN_CFG_JSON`. The 2026-04 cleanup found
that `submit_onchain` previously swallowed every exception silently —
operators believed an anchoring had happened when it had not. The
fixed implementation **logs at WARN/ERROR and re-raises by default**;
set `CHAIN_BEST_EFFORT=true` to restore the previous swallow-and-
return-None behaviour for long-running simulation loops where
anchoring is best-effort. The simulator (`mvp/simulation/generate_results.py`)
defaults to `CHAIN_BEST_EFFORT=true` for HPC runs but records the
outcome in `decision_ledger_tx_status` so the count of how many
submissions actually landed is auditable.

There is no verifying reader for anchored roots. A future hardening
should add a `verify_anchored_root.py` that reads the on-chain root
and compares against the local ledger.

## Production checklist

- [x] Permissioned EVM network entry in `hardhat.config.cjs`
      (`permissioned`); the deployer accepts validator keys via
      `PERMISSIONED_PRIVKEYS` and grants role permissions via
      `EXTRA_LOGGERS` / `EXTRA_ANCHORERS`.
- [x] Role-based access control on `DecisionLogger`,
      `ProvenanceRegistry`, and `SLCARewards`.
- [x] Per-tx anchoring receipt verifier
      (`mvp/simulation/analysis/verify_anchored_root.py`).
- [ ] OpenZeppelin `ReentrancyGuard` import in `AgriDAO.sol`. The
      hand-rolled `_LocalReentrancyGuard` is functionally equivalent
      and Slither passes, but the canonical import is preferable.
      Replacing it requires adding OpenZeppelin as an npm
      dependency and is queued for a follow-up.
- [ ] CI workflow that compiles and runs Solidity tests against a
      live Besu network (the current CI runs against localhost
      Hardhat only).

When operating against a permissioned chain, set the env vars listed
in `hardhat.config.cjs` and run:

```bash
EXTRA_LOGGERS=0xagent1,0xagent2 \
EXTRA_ANCHORERS=0xexplainerService \
PERMISSIONED_RPC_URL=https://validator-1.consortium.example \
PERMISSIONED_CHAIN_ID=2025 \
PERMISSIONED_PRIVKEYS=0x... \
npx hardhat run scripts/deploy.js --network permissioned
```
