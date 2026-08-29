#!/bin/bash
# SLURM scenario-array task for the 20-seed H3 stress suite.
# Submitted by hpc/hpc_run.sh after the primary seed array succeeds; one
# scenario per task.  Each task reuses the frozen primary nominal endpoints.
#SBATCH --job-name=agribrain-stress
#SBATCH --array=0-4
#SBATCH --time=18:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stress_%A_%a.out
#SBATCH --error=logs/stress_%A_%a.err

set -euo pipefail
if [ -z "${AGRIBRAIN_SOURCE_SNAPSHOT:-}" ]; then
    echo "BLOCK: AGRIBRAIN_SOURCE_SNAPSHOT not exported. Submit via hpc/hpc_run.sh."
    exit 1
fi
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
if [ "$(pwd -P)" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: stress worker is not executing from the declared source snapshot."
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

SCENARIOS=(heatwave overproduction cyber_outage adaptive_pricing baseline)
SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
OUT_ROOT="mvp/simulation/results/stress_runs/${RUN_TAG}/${SCENARIO}"
PRIMARY_ROOT="mvp/simulation/results/benchmark_seeds/${RUN_TAG}"

source hpc/publication_env.sh
export STRESS_SCENARIOS="$SCENARIO"
export STRESS_N_SEEDS=20
export STRESS_LEARNING_EPISODES=4
export STRESS_MAX_ROWS=0
export STRESS_OUTPUT_DIR="$OUT_ROOT"
# Retain the 500 stressed episode-3 ledgers in a non-hidden, run-scoped
# canonical tree.  Nominal H3 endpoints continue to reuse the 100 AGRI-BRAIN
# ledgers produced by the primary seed array; they are not rerun or copied
# into this tree.
export STRESS_LEDGER_ROOT="mvp/simulation/results/decision_ledger_h3/${RUN_TAG}"
export STRESS_PRIMARY_SEEDS_DIR="$PRIMARY_ROOT"
python hpc/validate_publication_env.py
python hpc/validate_source_checkout.py --allow-run-artifacts
python hpc/validate_source_snapshot.py
python hpc/capture_publication_environment.py --validate-only
python hpc/validate_pinn_artifacts.py
if [ ! -d "$PRIMARY_ROOT" ]; then
    echo "BLOCK: primary benchmark directory is missing: ${PRIMARY_ROOT}"
    exit 1
fi

echo "[stress scenario=${SCENARIO} tag=${RUN_TAG}] starting at $(date)"
RUNTIME_RECEIPT="${STRESS_LEDGER_ROOT}/${SCENARIO}/runtime_receipts/job_${SLURM_JOB_ID}__restart_${SLURM_RESTART_COUNT:-0}.json"
python hpc/run_with_resource_receipt.py \
    --output "$RUNTIME_RECEIPT" \
    --label "h3_scenario_${SCENARIO}" \
    -- python mvp/simulation/benchmarks/run_stress_suite.py
echo "[stress scenario=${SCENARIO}] validating complete 400-episode evidence inventory"
python hpc/validate_complete_episode_evidence.py \
    --ledger-root "${STRESS_LEDGER_ROOT}/${SCENARIO}" \
    --expected-groups 100 \
    --expected-episodes 400 \
    --expected-adaptation-ledgers 300 \
    --expected-final-ledgers 100 \
    --manifest "${STRESS_LEDGER_ROOT}/${SCENARIO}/complete_episode_evidence_manifest.json"
python hpc/validate_source_checkout.py --allow-run-artifacts
python hpc/validate_source_snapshot.py
echo "[stress scenario=${SCENARIO}] complete at $(date)"
