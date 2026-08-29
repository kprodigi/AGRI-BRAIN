#!/bin/bash
# Publication-only recovery for an already completed core simulation run.
#
# This script never invokes a simulation worker.  It is submitted held by
# hpc/publication_recovery_run.sh, authorized for that exact Slurm job id, and
# only then released.  The original raw outputs and submission receipt remain
# outside this clean publication-repair worktree and are verified byte for byte
# before a private copy is staged for deterministic aggregation.
#SBATCH --job-name=agribrain-core-recovery
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

for required in RUN_TAG SLURM_JOB_ID AGRIBRAIN_SOURCE_SNAPSHOT AGRIBRAIN_VENV \
    AGRIBRAIN_SIMULATION_COMMIT AGRIBRAIN_PUBLICATION_CODE_COMMIT \
    AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256 \
    AGRIBRAIN_PUBLICATION_SOURCE_TREE_SHA256 \
    AGRIBRAIN_ORIGINAL_CORE_RECEIPT AGRIBRAIN_EXTERNAL_RECOVERY_RECEIPT \
    AGRIBRAIN_EXTERNAL_RAW_MANIFEST AGRIBRAIN_RAW_SEEDS_DIR \
    AGRIBRAIN_RAW_STRESS_DIR AGRIBRAIN_RAW_H3_LEDGER_DIR; do
    if [ -z "${!required:-}" ]; then
        echo "BLOCK: core recovery input ${required} is missing."
        exit 1
    fi
done

cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
if [ "$(pwd -P)" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: core recovery is outside its publication-repair snapshot."
    exit 1
fi
if [ "$AGRIBRAIN_VENV" != ".publication_venvs/${RUN_TAG}" ] \
    || [ ! -f "$AGRIBRAIN_VENV/bin/activate" ]; then
    echo "BLOCK: core recovery run-scoped venv is missing or mismatched."
    exit 1
fi
source "$AGRIBRAIN_VENV/bin/activate"

if [ "$AGRIBRAIN_SIMULATION_COMMIT" = "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" ]; then
    echo "BLOCK: recovery requires distinct simulation and publication commits."
    exit 1
fi
export AGRIBRAIN_GIT_COMMIT="$AGRIBRAIN_SIMULATION_COMMIT"
export AGRIBRAIN_SOURCE_TREE_SHA256="$AGRIBRAIN_PUBLICATION_SOURCE_TREE_SHA256"
export DETERMINISTIC_MODE=false

python hpc/publication_recovery_receipt.py validate \
    --receipt "$AGRIBRAIN_EXTERNAL_RECOVERY_RECEIPT" \
    --original-submission-receipt "$AGRIBRAIN_ORIGINAL_CORE_RECEIPT" \
    --kind core \
    --run-tag "$RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" \
    --recovery-publisher-slurm-job-id "$SLURM_JOB_ID"

# First mandatory live-input gate: this immediately precedes staging and no
# aggregation code has run yet.
python hpc/preserved_raw_manifest.py validate \
    --manifest "$AGRIBRAIN_EXTERNAL_RAW_MANIFEST" \
    --kind core \
    --run-tag "$RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --simulation-source-tree-sha256 "$AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256" \
    --input-root "benchmark_seed_outputs=${AGRIBRAIN_RAW_SEEDS_DIR}" \
    --input-root "stress_outputs=${AGRIBRAIN_RAW_STRESS_DIR}" \
    --input-root "h3_decision_ledgers=${AGRIBRAIN_RAW_H3_LEDGER_DIR}" \
    --input-file "core_submission_receipt.json=${AGRIBRAIN_ORIGINAL_CORE_RECEIPT}"

RESULTS_DIR="mvp/simulation/results"
SEEDS_DIR="${RESULTS_DIR}/benchmark_seeds/${RUN_TAG}"
STRESS_DIR="${RESULTS_DIR}/stress_runs/${RUN_TAG}"
H3_LEDGER_DIR="${RESULTS_DIR}/decision_ledger_h3/${RUN_TAG}"
CORE_RECEIPT="${RESULTS_DIR}/core_submission_receipts/${RUN_TAG}.json"
RECOVERY_RECEIPT="${RESULTS_DIR}/publication_recovery_receipts/${RUN_TAG}.json"
RAW_MANIFEST="${RESULTS_DIR}/preserved_raw_manifests/${RUN_TAG}.json"

for destination in "$SEEDS_DIR" "$STRESS_DIR" "$H3_LEDGER_DIR" \
    "$CORE_RECEIPT" "$RECOVERY_RECEIPT" "$RAW_MANIFEST"; do
    if [ -e "$destination" ] || [ -L "$destination" ]; then
        echo "BLOCK: refusing to overwrite staged recovery evidence: ${destination}"
        exit 1
    fi
done

mkdir -p "${RESULTS_DIR}/benchmark_seeds" \
    "${RESULTS_DIR}/stress_runs" \
    "${RESULTS_DIR}/decision_ledger_h3" \
    "${RESULTS_DIR}/core_submission_receipts" \
    "${RESULTS_DIR}/publication_recovery_receipts" \
    "${RESULTS_DIR}/preserved_raw_manifests"
mkdir "$SEEDS_DIR" "$STRESS_DIR" "$H3_LEDGER_DIR"
cp -a -- "$AGRIBRAIN_RAW_SEEDS_DIR/." "$SEEDS_DIR/"
cp -a -- "$AGRIBRAIN_RAW_STRESS_DIR/." "$STRESS_DIR/"
cp -a -- "$AGRIBRAIN_RAW_H3_LEDGER_DIR/." "$H3_LEDGER_DIR/"
python - \
    "$AGRIBRAIN_ORIGINAL_CORE_RECEIPT" "$CORE_RECEIPT" \
    "$AGRIBRAIN_EXTERNAL_RECOVERY_RECEIPT" "$RECOVERY_RECEIPT" \
    "$AGRIBRAIN_EXTERNAL_RAW_MANIFEST" "$RAW_MANIFEST" <<'PY'
import sys
from pathlib import Path

pairs = list(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
for source_raw, target_raw in pairs:
    source, target = Path(source_raw), Path(target_raw)
    if source.is_symlink() or target.is_symlink() or target.exists():
        raise SystemExit("BLOCK: unsafe or existing staged recovery evidence")
for source_raw, target_raw in pairs:
    source, target = Path(source_raw), Path(target_raw)
    payload = source.read_bytes()
    with target.open("xb") as handle:
        handle.write(payload)
    if target.read_bytes() != payload:
        raise SystemExit("BLOCK: staged recovery-evidence copy mismatch")
PY

export CORE_SUBMISSION_RECEIPT="$CORE_RECEIPT"
export AGRIBRAIN_RECOVERY_RECEIPT="$RECOVERY_RECEIPT"
export AGRIBRAIN_PRESERVED_RAW_MANIFEST="$RAW_MANIFEST"

# From this point forward, bind every gate to the private staged bytes that
# the deterministic producers actually consume.  This catches copy damage or
# any later mutation in the recovery worktree; the earlier gate covered the
# preserved originals.
export AGRIBRAIN_RAW_SEEDS_DIR="$SEEDS_DIR"
export AGRIBRAIN_RAW_STRESS_DIR="$STRESS_DIR"
export AGRIBRAIN_RAW_H3_LEDGER_DIR="$H3_LEDGER_DIR"
python hpc/preserved_raw_manifest.py validate \
    --manifest "$AGRIBRAIN_PRESERVED_RAW_MANIFEST" \
    --kind core \
    --run-tag "$RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --simulation-source-tree-sha256 "$AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256" \
    --input-root "benchmark_seed_outputs=${AGRIBRAIN_RAW_SEEDS_DIR}" \
    --input-root "stress_outputs=${AGRIBRAIN_RAW_STRESS_DIR}" \
    --input-root "h3_decision_ledgers=${AGRIBRAIN_RAW_H3_LEDGER_DIR}" \
    --input-file "core_submission_receipt.json=${CORE_SUBMISSION_RECEIPT}"

# hpc_publish.sh performs the complete normal semantic pipeline, but its
# recovery branch revalidates the authorization and live raw bindings before
# aggregation and again immediately before archive/receipt commitment.
exec bash hpc/hpc_publish.sh
