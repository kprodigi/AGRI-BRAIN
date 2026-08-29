#!/bin/bash
# Validate, analyse, manifest-bind, and archive a complete structural run.
#SBATCH --job-name=agribrain-structural-publish
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
if [ -z "${AGRIBRAIN_SOURCE_SNAPSHOT:-}" ]; then
    echo "BLOCK: source snapshot is missing; submit through hpc/hpc_sensitivity_run.sh."
    exit 1
fi
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
if [ "$(pwd -P)" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: structural publisher is outside the declared source snapshot."
    exit 1
fi
source hpc/ensure_git_available.sh

for required in RUN_TAG AGRIBRAIN_VENV AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT \
    AGRIBRAIN_SENSITIVITY_ROOT SENSITIVITY_RUN_DIR SENSITIVITY_RUN_PLAN \
    AGRIBRAIN_SOURCE_SNAPSHOT_MODE AGRIBRAIN_SOURCE_TREE_SHA256 SLURM_JOB_ID; do
    if [ -z "${!required:-}" ]; then
        echo "BLOCK: ${required} is missing; submit through hpc/hpc_sensitivity_run.sh."
        exit 1
    fi
done
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
python hpc/validate_structural_sensitivity_hpc.py

STATUS_PATH="${SENSITIVITY_RUN_DIR}/completion_status.json"
ANALYSIS_PATH="${SENSITIVITY_RUN_DIR}/structural_sensitivity_analysis.json"
ENVIRONMENT_PATH="${SENSITIVITY_RUN_DIR}/publication_environment.json"
SCHEDULER_ACCOUNTING_PATH="${SENSITIVITY_RUN_DIR}/slurm_simulation_accounting.json"
MANIFEST_PATH="${SENSITIVITY_RUN_DIR}/structural_sensitivity_artifact_manifest.json"
ARCHIVE_PATH="${SENSITIVITY_RUN_DIR}/structural_sensitivity_evidence_${RUN_TAG}.tar.gz"
RECEIPT_PATH="${SENSITIVITY_RUN_DIR}/structural_sensitivity_archive_receipt.json"
STRUCTURAL_TABLE_PATH="${SENSITIVITY_RUN_DIR}/structural_sensitivity_summary.csv"
STRUCTURAL_PNG_PATH="${SENSITIVITY_RUN_DIR}/structural_sensitivity_summary.png"
STRUCTURAL_PDF_PATH="${SENSITIVITY_RUN_DIR}/structural_sensitivity_summary.pdf"
STRUCTURAL_PUBLICATION_RECEIPT="${SENSITIVITY_RUN_DIR}/structural_sensitivity_publication_receipt.json"
for output in "$STATUS_PATH" "$ANALYSIS_PATH" "$ENVIRONMENT_PATH" \
    "$SCHEDULER_ACCOUNTING_PATH" "$MANIFEST_PATH" "$ARCHIVE_PATH" \
    "$RECEIPT_PATH" "$STRUCTURAL_TABLE_PATH" "$STRUCTURAL_PNG_PATH" \
    "$STRUCTURAL_PDF_PATH" "$STRUCTURAL_PUBLICATION_RECEIPT"; do
    if [ -e "$output" ]; then
        echo "BLOCK: refusing to overwrite existing final evidence: ${output}"
        exit 1
    fi
done
if [ ! -s "${SENSITIVITY_RUN_DIR}/slurm_submission.json" ]; then
    echo "BLOCK: Slurm submission receipt is missing."
    exit 1
fi

echo "=== Capture post-job Slurm accounting for all 3,000 simulation workers ==="
python hpc/capture_slurm_accounting.py \
    --submission-receipt "${SENSITIVITY_RUN_DIR}/slurm_submission.json" \
    --output "$SCHEDULER_ACCOUNTING_PATH" \
    --kind structural \
    --run-tag "$RUN_TAG" \
    --source-commit "$AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT" \
    --source-tree-sha256 "$AGRIBRAIN_SOURCE_TREE_SHA256"

echo "=== Hash-check all 3,000 structural task outputs ==="
STATUS_TMP="${STATUS_PATH}.tmp.${SLURM_JOB_ID:-manual}"
trap 'rm -f -- "$STATUS_TMP"' EXIT
python -m mvp.simulation.sensitivity.run_structural_sensitivity status \
    --run-plan "$SENSITIVITY_RUN_PLAN" \
    --submission-receipt "${SENSITIVITY_RUN_DIR}/slurm_submission.json" \
    > "$STATUS_TMP"
mv "$STATUS_TMP" "$STATUS_PATH"
trap - EXIT

echo "=== Compute the declared structural analysis ==="
python -m mvp.simulation.sensitivity.run_structural_sensitivity analyze \
    --run-plan "$SENSITIVITY_RUN_PLAN" \
    --output "$ANALYSIS_PATH"

echo "=== Export the provenance-bound structural table and figure ==="
python -m mvp.simulation.sensitivity.publish_structural_sensitivity \
    "$ANALYSIS_PATH" "$SENSITIVITY_RUN_DIR"

echo "=== Capture the locked run-scoped Python environment ==="
python hpc/capture_publication_environment.py --output "$ENVIRONMENT_PATH"

echo "=== Revalidate, manifest-bind, and byte-verify the external archive ==="
python -m mvp.simulation.sensitivity.finalize_structural_sensitivity \
    --run-plan "$SENSITIVITY_RUN_PLAN" \
    --status "$STATUS_PATH" \
    --analysis "$ANALYSIS_PATH" \
    --environment "$ENVIRONMENT_PATH" \
    --scheduler-accounting "$SCHEDULER_ACCOUNTING_PATH" \
    --manifest "$MANIFEST_PATH" \
    --archive "$ARCHIVE_PATH" \
    --receipt "$RECEIPT_PATH"

python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
echo "Structural analysis: ${ANALYSIS_PATH}"
echo "Verified archive:    ${ARCHIVE_PATH}"
echo "Archive receipt:     ${RECEIPT_PATH}"
echo "Core publication results directory was not used."
