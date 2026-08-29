# Chain runtime fixtures

The two `deployed-addresses.*.json` files in this directory are the
**Hardhat deterministic localhost-network addresses** for the six
deployed contracts (`AgentRegistry`, `PolicyStore`, `AgriDAO`,
`DecisionLogger`, `SLCARewards`, `ProvenanceRegistry`). They are
committed as fixtures so the FastAPI backend can autoload addresses
on startup without first running `npx hardhat run scripts/deploy.js`.

## Why these addresses are stable

Hardhat's `npx hardhat node` derives all accounts and contract
addresses deterministically from a fixed mnemonic. Address-0 deploys
the first contract, address-1 the second, etc. The published
addresses below match what *any* default Hardhat node will produce
for the deploy order in `agribrain/contracts/hardhat/scripts/deploy.js`:

* `AgentRegistry`     -> `0x5FbDB2315678afecb367f032d93F642f64180aa3`
* `PolicyStore`       -> `0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512`
* `AgriDAO`           -> `0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0`
* `DecisionLogger`    -> `0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9`
* `SLCARewards`       -> `0x5FC8d32690cc91D4c39d9d3abcBD16989F875707`
* `ProvenanceRegistry`-> `0x0165878A594ca255338adfa4d48449f69242Eb8F`

## When this gets overwritten

Running `bash agribrain/contracts/hardhat/scripts/start_localhost_chain.sh`
or `npx hardhat run scripts/deploy.js --network localhost` rewrites
both files. If the deploy order in `deploy.js` does not change, the
addresses are byte-identical and `git status` shows no diff. If the
deploy order *does* change, the new addresses must be committed in
the same change as the deploy script edit.

## Production posture

These addresses are NEVER used in production. Production contracts
are deployed to a real chain and their addresses are written by the
production deploy script into a separate runtime config (env var or
secret store), never into a tracked file. The presence of this
directory in the repo is a localhost-developer convenience only.
