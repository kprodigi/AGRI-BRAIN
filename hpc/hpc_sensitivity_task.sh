#!/bin/bash
# Execute one hash-bound structural-sensitivity manifest task.
# Submitted only by hpc/hpc_sensitivity_run.sh.
#SBATCH --job-name=agribrain-structural
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail
if [ -z "${AGRIBRAIN_SOURCE_SNAPSHOT:-}" ]; then
    echo "BLOCK: source snapshot is missing; submit through hpc/hpc_sensitivity_run.sh."
    exit 1
fi
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
if [ "$(pwd -P)" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: structural worker is outside the declared source snapshot."
    exit 1
fi
source hpc/ensure_git_available.sh

for required in RUN_TAG AGRIBRAIN_VENV AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT \
    AGRIBRAIN_SENSITIVITY_ROOT SENSITIVITY_RUN_DIR SENSITIVITY_RUN_PLAN \
    SENSITIVITY_TASK_OFFSET SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID SLURM_JOB_ID \
    AGRIBRAIN_SOURCE_SNAPSHOT_MODE AGRIBRAIN_SOURCE_TREE_SHA256; do
    if [ -z "${!required:-}" ]; then
        echo "BLOCK: ${required} is missing; submit through hpc/hpc_sensitivity_run.sh."
        exit 1
    fi
done
case "$SENSITIVITY_TASK_OFFSET" in *[!0-9]*|'') echo "BLOCK: invalid task offset."; exit 1;; esac
case "$SLURM_ARRAY_TASK_ID" in *[!0-9]*|'') echo "BLOCK: invalid array task id."; exit 1;; esac
TASK_INDEX=$((SENSITIVITY_TASK_OFFSET + SLURM_ARRAY_TASK_ID))
if [ "$TASK_INDEX" -lt 0 ] || [ "$TASK_INDEX" -ge 3000 ]; then
    echo "BLOCK: computed structural task index ${TASK_INDEX} is outside 0..2999."
    exit 1
fi
if [ "${AGRIBRAIN_VENV:-}" != ".publication_venvs/${RUN_TAG}" ]; then
    echo "BLOCK: run-scoped venv does not match RUN_TAG."
    exit 1
fi
if [ ! -f "$AGRIBRAIN_VENV/bin/activate" ]; then
    echo "BLOCK: run-scoped venv is missing: ${AGRIBRAIN_VENV}"
    exit 1
fi
source "$AGRIBRAIN_VENV/bin/activate"
source hpc/publication_env.sh

python hpc/validate_publication_env.py
python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
python hpc/capture_publication_environment.py --validate-only
python hpc/validate_pinn_artifacts.py
python hpc/validate_structural_sensitivity_hpc.py

echo "[structural task=${TASK_INDEX} tag=${RUN_TAG}] starting at $(date)"
RUNTIME_RECEIPT="${SENSITIVITY_RUN_DIR}/runtime_receipts/task_${TASK_INDEX}/job_${SLURM_JOB_ID}__restart_${SLURM_RESTART_COUNT:-0}.json"
python hpc/run_with_resource_receipt.py \
    --output "$RUNTIME_RECEIPT" \
    --label "structural_task_${TASK_INDEX}" \
    -- python -m mvp.simulation.sensitivity.run_structural_sensitivity run-task \
        --run-plan "$SENSITIVITY_RUN_PLAN" \
        --task-index "$TASK_INDEX" \
        --resume
python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
echo "[structural task=${TASK_INDEX}] complete at $(date)"
