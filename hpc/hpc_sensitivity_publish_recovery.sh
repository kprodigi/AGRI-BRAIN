#!/bin/bash
# Deterministically publish preserved structural results after a failed publisher.
#
# This script never runs a structural worker.  It is authorized only by a
# self-hashed recovery receipt created after this exact Slurm job was submitted
# with --hold.  Recovery stdout/stderr must be routed outside the raw run because
# the preserved manifest binds the complete original logs/ directory.
#SBATCH --job-name=agribrain-structural-recovery
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail

for required in AGRIBRAIN_SOURCE_SNAPSHOT AGRIBRAIN_SOURCE_SNAPSHOT_MODE \
    AGRIBRAIN_SOURCE_TREE_SHA256 AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256 \
    AGRIBRAIN_SIMULATION_COMMIT AGRIBRAIN_PUBLICATION_CODE_COMMIT \
    AGRIBRAIN_RECOVERY_RECEIPT AGRIBRAIN_PRESERVED_RAW_MANIFEST \
    AGRIBRAIN_RECOVERY_LOG_DIR AGRIBRAIN_VENV \
    AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT AGRIBRAIN_SENSITIVITY_ROOT \
    SENSITIVITY_RUN_DIR SENSITIVITY_RUN_PLAN RUN_TAG SLURM_JOB_ID; do
    if [ -z "${!required:-}" ]; then
        echo "BLOCK: ${required} is required for structural publication recovery."
        exit 1
    fi
done

cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
if [ "$(pwd -P)" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: recovery publisher is outside the publication-repair snapshot."
    exit 1
fi
source hpc/ensure_git_available.sh

if [ "$AGRIBRAIN_SIMULATION_COMMIT" != "$AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT" ]; then
    echo "BLOCK: structural simulation commit variables disagree."
    exit 1
fi
if [ "$AGRIBRAIN_SIMULATION_COMMIT" = "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" ]; then
    echo "BLOCK: recovery requires distinct simulation and publication commits."
    exit 1
fi
if [ "${AGRIBRAIN_VENV:-}" != ".publication_venvs/${RUN_TAG}" ]; then
    echo "BLOCK: run-scoped venv does not match RUN_TAG."
    exit 1
fi
if [ ! -f "$AGRIBRAIN_VENV/bin/activate" ]; then
    echo "BLOCK: recovery run-scoped venv is missing: ${AGRIBRAIN_VENV}"
    exit 1
fi
source "$AGRIBRAIN_VENV/bin/activate"
source hpc/publication_env.sh

# Prove that Slurm itself routed this job's live stdout/stderr away from every
# raw-manifest-bound directory.  Checking only a caller-supplied variable would
# not protect against an incorrect sbatch --output/--error argument.
python - "$SLURM_JOB_ID" "$AGRIBRAIN_RECOVERY_LOG_DIR" "$SENSITIVITY_RUN_DIR" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

job_id, raw_log_root, raw_run_root = sys.argv[1:]
log_root = Path(raw_log_root)
run_root = Path(raw_run_root)
if not log_root.is_absolute() or not run_root.is_absolute():
    raise SystemExit("BLOCK: recovery log and structural run paths must be absolute")
log_root = log_root.resolve(strict=True)
run_root = run_root.resolve(strict=True)
try:
    log_root.relative_to(run_root)
except ValueError:
    pass
else:
    raise SystemExit("BLOCK: recovery log directory is inside preserved raw inputs")
record = subprocess.run(
    ["scontrol", "show", "job", "-o", job_id],
    check=True,
    capture_output=True,
    text=True,
).stdout
for field in ("StdOut", "StdErr"):
    match = re.search(rf"(?:^|\s){field}=(\S+)", record)
    if match is None:
        raise SystemExit(f"BLOCK: Slurm job record lacks {field}")
    path = Path(match.group(1))
    if not path.is_absolute():
        raise SystemExit(f"BLOCK: Slurm {field} is not an absolute path")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(log_root)
    except ValueError as exc:
        raise SystemExit(
            f"BLOCK: Slurm {field} is outside AGRIBRAIN_RECOVERY_LOG_DIR"
        ) from exc
    try:
        resolved.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise SystemExit(f"BLOCK: Slurm {field} would mutate preserved raw inputs")
PY

export AGRIBRAIN_GIT_COMMIT="$AGRIBRAIN_PUBLICATION_CODE_COMMIT"
python hpc/validate_publication_env.py
python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
python hpc/capture_publication_environment.py --validate-only
python hpc/validate_pinn_artifacts.py
AGRIBRAIN_GIT_COMMIT="$AGRIBRAIN_SIMULATION_COMMIT" \
    python hpc/validate_structural_sensitivity_hpc.py

ORIGINAL_SUBMISSION="${SENSITIVITY_RUN_DIR}/slurm_submission.json"
CANONICAL_RECOVERY_RECEIPT="${SENSITIVITY_RUN_DIR}/publication_recovery_receipts/${RUN_TAG}.json"
CANONICAL_RAW_MANIFEST="${SENSITIVITY_RUN_DIR}/preserved_raw_manifests/${RUN_TAG}.json"
python - "$AGRIBRAIN_RECOVERY_RECEIPT" "$CANONICAL_RECOVERY_RECEIPT" \
    "$AGRIBRAIN_PRESERVED_RAW_MANIFEST" "$CANONICAL_RAW_MANIFEST" <<'PY'
import sys
from pathlib import Path

for actual_raw, expected_raw in zip(sys.argv[1::2], sys.argv[2::2], strict=True):
    actual = Path(actual_raw)
    expected = Path(expected_raw)
    if actual.is_symlink() or expected.is_symlink():
        raise SystemExit("BLOCK: canonical recovery evidence must not be symlinked")
    if actual.resolve(strict=True) != expected.resolve(strict=True):
        raise SystemExit("BLOCK: recovery evidence is outside its canonical run path")
PY

python hpc/publication_recovery_receipt.py validate \
    --receipt "$AGRIBRAIN_RECOVERY_RECEIPT" \
    --original-submission-receipt "$ORIGINAL_SUBMISSION" \
    --kind structural \
    --run-tag "$RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" \
    --recovery-publisher-slurm-job-id "$SLURM_JOB_ID"

validate_preserved_raw_outputs() {
    python hpc/preserved_raw_manifest.py validate \
        --manifest "$AGRIBRAIN_PRESERVED_RAW_MANIFEST" \
        --kind structural \
        --run-tag "$RUN_TAG" \
        --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
        --simulation-source-tree-sha256 "$AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256" \
        --input-root "logs=${SENSITIVITY_RUN_DIR}/logs" \
        --input-root "runtime_receipts=${SENSITIVITY_RUN_DIR}/runtime_receipts" \
        --input-root "tasks=${SENSITIVITY_RUN_DIR}/tasks" \
        --input-file "episode_accounting.json=${SENSITIVITY_RUN_DIR}/episode_accounting.json" \
        --input-file "experiment_protocol.json=${SENSITIVITY_RUN_DIR}/experiment_protocol.json" \
        --input-file "lhs_design.csv=${SENSITIVITY_RUN_DIR}/lhs_design.csv" \
        --input-file "lhs_design.json=${SENSITIVITY_RUN_DIR}/lhs_design.json" \
        --input-file "parameter_registry.json=${SENSITIVITY_RUN_DIR}/parameter_registry.json" \
        --input-file "run_plan.json=${SENSITIVITY_RUN_DIR}/run_plan.json" \
        --input-file "slurm_submission.json=${SENSITIVITY_RUN_DIR}/slurm_submission.json" \
        --input-file "task_manifest.json=${SENSITIVITY_RUN_DIR}/task_manifest.json" \
        --input-file "task_manifest.jsonl=${SENSITIVITY_RUN_DIR}/task_manifest.jsonl"
}

# First immutable-input check: no analysis, rendering, aggregation, or archive
# operation has run in this recovery job yet.
validate_preserved_raw_outputs

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
    if [ -e "$output" ] || [ -L "$output" ]; then
        echo "BLOCK: refusing to overwrite existing structural evidence: ${output}"
        exit 1
    fi
done

echo "=== Capture accounting for the preserved 3,000 simulation tasks ==="
python hpc/capture_slurm_accounting.py \
    --submission-receipt "$ORIGINAL_SUBMISSION" \
    --output "$SCHEDULER_ACCOUNTING_PATH" \
    --kind structural \
    --run-tag "$RUN_TAG" \
    --source-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --source-tree-sha256 "$AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256" \
    --attempts 12 \
    --retry-seconds 5 \
    --max-retry-seconds 120 \
    --query-timeout-seconds 60

echo "=== Hash-check the preserved 3,000-task panel ==="
STATUS_TMP="${STATUS_PATH}.tmp.${SLURM_JOB_ID}"
trap 'rm -f -- "$STATUS_TMP"' EXIT
python -m mvp.simulation.sensitivity.run_structural_sensitivity status \
    --run-plan "$SENSITIVITY_RUN_PLAN" \
    --submission-receipt "$ORIGINAL_SUBMISSION" \
    > "$STATUS_TMP"
mv "$STATUS_TMP" "$STATUS_PATH"
trap - EXIT

echo "=== Recompute deterministic structural analysis and publication exports ==="
python -m mvp.simulation.sensitivity.run_structural_sensitivity analyze \
    --run-plan "$SENSITIVITY_RUN_PLAN" \
    --output "$ANALYSIS_PATH"
python -m mvp.simulation.sensitivity.publish_structural_sensitivity \
    "$ANALYSIS_PATH" "$SENSITIVITY_RUN_DIR"
python hpc/capture_publication_environment.py --output "$ENVIRONMENT_PATH"

# Second explicit live check is immediately before the final semantic
# manifest/archive boundary.  The finalizer repeats it internally before write.
validate_preserved_raw_outputs

python -m mvp.simulation.sensitivity.finalize_structural_sensitivity \
    --run-plan "$SENSITIVITY_RUN_PLAN" \
    --status "$STATUS_PATH" \
    --analysis "$ANALYSIS_PATH" \
    --environment "$ENVIRONMENT_PATH" \
    --scheduler-accounting "$SCHEDULER_ACCOUNTING_PATH" \
    --manifest "$MANIFEST_PATH" \
    --archive "$ARCHIVE_PATH" \
    --receipt "$RECEIPT_PATH" \
    --recovery-receipt "$AGRIBRAIN_RECOVERY_RECEIPT" \
    --preserved-raw-manifest "$AGRIBRAIN_PRESERVED_RAW_MANIFEST" \
    --publication-commit "$AGRIBRAIN_PUBLICATION_CODE_COMMIT"

python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
echo "Structural publication-only recovery complete; no simulation was rerun."
echo "Verified archive: ${ARCHIVE_PATH}"
echo "Archive receipt:  ${RECEIPT_PATH}"
