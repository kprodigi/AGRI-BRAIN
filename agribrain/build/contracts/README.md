# Compiled-contract ABIs (fixtures, NOT build outputs)

The `*.json` files in this directory are pinned ABI fixtures consumed
by the FastAPI backend's chain client (`agribrain/backend/src/chain/contracts.py`)
and the simulator's `chain_query` MCP tool. They are committed
deliberately so:

* The Python client can decode `DecisionLogged` / `EpisodeLogged`
  events without invoking `npx hardhat compile` first.
* The artifact-manifest SHA-256 hashes for these JSONs are stable
  across CI runs (Hardhat's `artifacts/cache/` produces non-deterministic
  paths and timestamps; `agribrain/build/contracts/` is the curated
  subset).

## Path hygiene

| Path                                                | Tracked?      | Purpose |
|-----------------------------------------------------|---------------|---------|
| `agribrain/build/contracts/*.json`                  | yes (here)    | Pinned ABI fixtures (this directory). |
| `agribrain/contracts/hardhat/artifacts/`            | gitignored    | Raw `npx hardhat compile` output (transient). |
| `agribrain/contracts/hardhat/cache/`                | gitignored    | Hardhat's solidity cache. |
| `agribrain/backend/runtime/chain/deployed-addresses.localhost.json` | yes | Hardhat deterministic local addresses (fixture). |

## When to regenerate

Only when the Solidity source under
`agribrain/contracts/hardhat/contracts/` materially changes (new
function, event signature change, storage layout change). Procedure:

```bash
cd agribrain/contracts/hardhat
npm install
npx hardhat compile
# Copy the relevant ABI JSONs out of artifacts/contracts/<Name>.sol/<Name>.json
# (just the ABI portion, not the full Hardhat artifact wrapper) into
# agribrain/build/contracts/<Name>.json.
```

After regenerating, rerun the artifact-manifest builder so the SHA-256
hashes reflect the new ABIs:

```bash
python mvp/simulation/analysis/build_artifact_manifest.py
```

Bump the package version and follow `docs/RELEASE.md` if the change
is backward-incompatible.
