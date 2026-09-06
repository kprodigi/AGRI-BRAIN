#!/bin/bash
# SLURM job-array task: run one seed of the AgriBrain benchmark.
# Submitted by hpc/hpc_run.sh; expects RUN_TAG to be exported by sbatch.
#
# Output: mvp/simulation/results/benchmark_seeds/${RUN_TAG}/seed_${SEED}.json
# One file per array task, isolated per run by the hash-tagged subdirectory.
#SBATCH --job-name=agribrain-seed
#SBATCH --array=0-19
# The 18-hour request is a conservative scheduling limit, not a runtime claim;
# the publication environment records the actual run platform and timestamps.
#SBATCH --time=18:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/seed_%A_%a.out
#SBATCH --error=logs/seed_%A_%a.err

set -euo pipefail

if [ -z "${AGRIBRAIN_SOURCE_SNAPSHOT:-}" ]; then
    echo "BLOCK: AGRIBRAIN_SOURCE_SNAPSHOT not exported. Submit via hpc/hpc_run.sh."
    exit 1
fi
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
if [ "$(pwd -P)" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: seed worker is not executing from the declared source snapshot."
    exit 1
fi

# Compute nodes on the target cluster expose Git through a module rather than
# the default batch PATH.  Load it before the fail-closed source identity gate.
source hpc/ensure_git_available.sh

if [ -z "${RUN_TAG:-}" ]; then
    echo "BLOCK: RUN_TAG not exported. Submit via hpc/hpc_run.sh."
    exit 1
fi
for required in AGRIBRAIN_SOURCE_SNAPSHOT_MODE AGRIBRAIN_SOURCE_TREE_SHA256 \
    SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID; do
    if [ -z "${!required:-}" ]; then
        echo "BLOCK: ${required} is missing; submit via hpc/hpc_run.sh."
        exit 1
    fi
done
if [ "${AGRIBRAIN_VENV:-}" != ".publication_venvs/${RUN_TAG}" ]; then
    echo "BLOCK: AGRIBRAIN_VENV is missing or does not match RUN_TAG."
    exit 1
fi
if [ ! -f "$AGRIBRAIN_VENV/bin/activate" ]; then
    echo "BLOCK: run-scoped venv not found: ${AGRIBRAIN_VENV}"
    exit 1
fi
source "$AGRIBRAIN_VENV/bin/activate"

# Belt-and-suspenders: even if --export skipped DETERMINISTIC_MODE, force
# stochastic. Aborts visibly if something upstream tried to set it true.
if [ "${DETERMINISTIC_MODE:-false}" = "true" ]; then
    echo "BLOCK: DETERMINISTIC_MODE=true reached the seed task. Stochastic seeds expected."
    exit 1
fi
export DETERMINISTIC_MODE=false

# Confirmatory posture.  Re-apply the complete canonical environment inside
# the worker so a direct/manual sbatch cannot bypass the login-node preflight.
source hpc/publication_env.sh
python hpc/validate_publication_env.py
python hpc/validate_source_checkout.py --allow-run-artifacts
python hpc/validate_source_snapshot.py
python hpc/capture_publication_environment.py --validate-only
python hpc/validate_pinn_artifacts.py

# Map array index to the canonical 20-seed list.
SEEDS=(42 1337 2024 7 99 101 202 303 404 505 \
       606 707 808 909 1010 1111 1212 1313 1414 1515)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

OUT_DIR="mvp/simulation/results/benchmark_seeds/${RUN_TAG}"
mkdir -p "$OUT_DIR"

# Route the per-step DecisionLedger JSONL files to a seed-specific root so
# the 20 Slurm array tasks cannot overwrite one another on a shared default
# path. Each seed keeps the exact 55 final ledgers (11 modes x 5 scenarios),
# 150 compressed adaptation ledgers, and 205 lossless episode archives.
#
# A fresh in-memory ledger supplies within-episode history and shadows all
# prior files. The 55 mode/scenario JSONLs in this seed-specific directory are
# the final-episode audit outputs used for decision-level attribution.
# Each seed task writes under
# mvp/simulation/results/benchmark_seeds/${RUN_TAG}/decision_ledger_${SEED}/,
# hpc/hpc_publish.sh rolls these into one archive, and
# mvp/simulation/benchmarks/aggregate_channel_attribution.py computes the
# cross-seed per-decision channel-attribution statistics that the
# manuscript §5.8 / Fig 14 reports.
LEDGER_DIR="$OUT_DIR/decision_ledger_${SEED}"
mkdir -p "$LEDGER_DIR"
export DECISION_LEDGER_DIR="$LEDGER_DIR"

echo "[seed=${SEED} tag=${RUN_TAG}] starting at $(date)"
echo "  metric envelope -> ${OUT_DIR}/seed_${SEED}.json"
echo "  decision ledger -> ${LEDGER_DIR}/"

# Pre-flight invariants check. Costs <1 s and avoids 2-6 h of wasted
# compute producing numbers from a stale code path.
python -c "
import sys
sys.path.insert(0, 'agribrain/backend')
from pirag.mcp.registry import get_default_registry
from pirag.context_to_logits import THETA_CONTEXT
names = {t['name'] for t in get_default_registry().list_tools()}
expected = {
    'calculator', 'chain_query', 'check_compliance', 'context_features',
    'convert_units', 'demand_query', 'explain', 'footprint_query',
    'pirag_query', 'policy_oracle', 'slca_lookup', 'spoilage_forecast',
    'yield_query',
}
assert names == expected, f'BLOCK: MCP registry drift: {sorted(names ^ expected)}'
from src.models.action_selection import THETA
assert THETA_CONTEXT.shape == (3, 5), 'BLOCK: THETA_CONTEXT shape not (3,5)'
assert THETA.shape == (3, 10), 'BLOCK: THETA shape not (3,10)'
print('Pre-flight invariants OK')
"

RUNTIME_RECEIPT="$OUT_DIR/runtime_receipts/seed_${SEED}__job_${SLURM_JOB_ID}__restart_${SLURM_RESTART_COUNT:-0}.json"
python hpc/run_with_resource_receipt.py \
    --output "$RUNTIME_RECEIPT" \
    --label "core_seed_${SEED}" \
    -- python mvp/simulation/benchmarks/run_single_seed.py \
        "$SEED" --output-dir "$OUT_DIR"

echo "[seed=${SEED}] validating complete 205-episode evidence inventory"
python hpc/validate_complete_episode_evidence.py \
    --ledger-root "$LEDGER_DIR" \
    --expected-groups 55 \
    --expected-episodes 205 \
    --expected-adaptation-ledgers 150 \
    --expected-final-ledgers 55 \
    --manifest "$LEDGER_DIR/complete_episode_evidence_manifest.json"

python hpc/validate_source_checkout.py --allow-run-artifacts
python hpc/validate_source_snapshot.py
echo "[seed=${SEED}] complete at $(date)"
