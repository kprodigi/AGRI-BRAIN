#!/usr/bin/env bash
# AGRI-BRAIN: localhost EVM bring-up.
#
# Brings up a Hardhat node, deploys all six smart contracts, writes the
# addresses where the backend auto-loads them, and prints the
# CHAIN_PRIVKEY to export. After this script exits the node is still
# running in the background (PID printed at the end); kill it with
# `kill <PID>` when you're done.
#
# Usage (from the repo root):
#   bash agribrain/contracts/hardhat/scripts/start_localhost_chain.sh
#
# Prereqs: Node.js 18+, npm 9+. Idempotent: re-running redeploys.
set -euo pipefail

HARDHAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="${HARDHAT_DIR}/.run"
mkdir -p "${LOG_DIR}"
NODE_LOG="${LOG_DIR}/hardhat-node.log"

cd "${HARDHAT_DIR}"

if [[ ! -d node_modules ]]; then
  echo "[start_localhost_chain] installing Hardhat dependencies..."
  npm install --silent
fi

# Start the node if nothing is already listening on 8545.
if ! curl -s -o /dev/null -w "%{http_code}" \
        --max-time 1 \
        --data '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}' \
        -H 'Content-Type: application/json' \
        http://127.0.0.1:8545 | grep -q "^200$"; then
  echo "[start_localhost_chain] starting hardhat node on 127.0.0.1:8545 (logs: ${NODE_LOG})..."
  nohup npx hardhat node > "${NODE_LOG}" 2>&1 &
  NODE_PID=$!
  # Wait up to 20s for the JSON-RPC port to come up.
  for _ in $(seq 1 40); do
    sleep 0.5
    if curl -s -o /dev/null -w "%{http_code}" \
          --max-time 1 \
          --data '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}' \
          -H 'Content-Type: application/json' \
          http://127.0.0.1:8545 | grep -q "^200$"; then
      break
    fi
  done
else
  echo "[start_localhost_chain] hardhat node already running on 127.0.0.1:8545"
  NODE_PID="(existing)"
fi

echo "[start_localhost_chain] deploying contracts..."
npx hardhat run scripts/deploy.js --network localhost

# Hardhat's deterministic accounts. Account 0 is the deployer.
DEPLOYER_PRIVKEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

cat <<EOF

[start_localhost_chain] DONE.

Addresses written to:
  agribrain/contracts/hardhat/deployed-addresses.localhost.json
  agribrain/backend/runtime/chain/deployed-addresses.localhost.json

To anchor decisions on chain, export the deployer key and (re)start
the FastAPI backend:

  export CHAIN_PRIVKEY=${DEPLOYER_PRIVKEY}
  export PROVENANCE_ADDR=\$(jq -r .ProvenanceRegistry agribrain/backend/runtime/chain/deployed-addresses.localhost.json)
  python -m uvicorn src.app:API --host 127.0.0.1 --port 8100 --app-dir agribrain/backend

Then verify a fresh decision is anchored:

  curl -s -X POST http://127.0.0.1:8100/decide \\
       -H 'Content-Type: application/json' \\
       -d '{"agent_id":"farm","role":"farm"}' \\
       | jq .memo.tx_hash
  # -> "0x..."  (a real transaction hash; not "0x0" and not null)

Hardhat node PID: ${NODE_PID}
Stop it later with: kill ${NODE_PID}
EOF
