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
RECOVERY_ATTEMPT_ROOT="${SENSITIVITY_RUN_DIR}/publication_recovery_attempts/${SLURM_JOB_ID}"
AGRIBRAIN_RECOVERY_RECEIPT="${RECOVERY_ATTEMPT_ROOT}/publication_recovery_receipts/${RUN_TAG}.json"
AGRIBRAIN_PRESERVED_RAW_MANIFEST="${RECOVERY_ATTEMPT_ROOT}/preserved_raw_manifests/${RUN_TAG}.json"
export AGRIBRAIN_RECOVERY_RECEIPT AGRIBRAIN_PRESERVED_RAW_MANIFEST
python - "$SENSITIVITY_RUN_DIR" "$RECOVERY_ATTEMPT_ROOT" "$SLURM_JOB_ID" \
    "$AGRIBRAIN_RECOVERY_RECEIPT" "$AGRIBRAIN_PRESERVED_RAW_MANIFEST" \
    "$RUN_TAG" <<'PY'
import os
import re
import sys
from pathlib import Path

run_root = Path(os.path.abspath(sys.argv[1]))
attempt_root = Path(os.path.abspath(sys.argv[2]))
job_id, receipt_raw, manifest_raw, run_tag = sys.argv[3:]
if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
    raise SystemExit("BLOCK: recovery publisher Slurm job ID is invalid")
expected_attempt = run_root / "publication_recovery_attempts" / job_id
if attempt_root != expected_attempt:
    raise SystemExit("BLOCK: recovery attempt root is not job-ID-scoped")
cursor = attempt_root
while True:
    if cursor.is_symlink():
        raise SystemExit(
            f"BLOCK: recovery attempt path has a symlink component: {cursor}"
        )
    if cursor == run_root:
        break
    if cursor == cursor.parent:
        raise SystemExit("BLOCK: recovery attempt root escapes the structural run")
    cursor = cursor.parent
if not attempt_root.is_dir():
    raise SystemExit("BLOCK: recovery attempt root is missing")
expected_files = (
    (
        Path(receipt_raw),
        attempt_root / "publication_recovery_receipts" / f"{run_tag}.json",
        "publication-recovery receipt",
    ),
    (
        Path(manifest_raw),
        attempt_root / "preserved_raw_manifests" / f"{run_tag}.json",
        "preserved raw-output manifest",
    ),
)
for actual, expected, label in expected_files:
    actual = Path(os.path.abspath(actual))
    if actual != expected or actual.is_symlink() or not actual.is_file():
        raise SystemExit(f"BLOCK: {label} is not canonical for this recovery attempt")
    parent = actual.parent
    if parent.is_symlink() or not parent.is_dir():
        raise SystemExit(f"BLOCK: {label} parent is unsafe")
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

STATUS_PATH="${RECOVERY_ATTEMPT_ROOT}/completion_status.json"
ANALYSIS_PATH="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_analysis.json"
ENVIRONMENT_PATH="${RECOVERY_ATTEMPT_ROOT}/publication_environment.json"
SCHEDULER_ACCOUNTING_PATH="${RECOVERY_ATTEMPT_ROOT}/slurm_simulation_accounting.json"
MANIFEST_PATH="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_artifact_manifest.json"
ARCHIVE_PATH="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_evidence_${RUN_TAG}.tar.gz"
RECEIPT_PATH="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_archive_receipt.json"
STRUCTURAL_TABLE_PATH="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_summary.csv"
STRUCTURAL_PNG_PATH="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_summary.png"
STRUCTURAL_PDF_PATH="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_summary.pdf"
STRUCTURAL_PUBLICATION_RECEIPT="${RECOVERY_ATTEMPT_ROOT}/structural_sensitivity_publication_receipt.json"
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
    "$ANALYSIS_PATH" "$RECOVERY_ATTEMPT_ROOT"
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
    --recovery-attempt-root "$RECOVERY_ATTEMPT_ROOT" \
    --recovery-receipt "$AGRIBRAIN_RECOVERY_RECEIPT" \
    --preserved-raw-manifest "$AGRIBRAIN_PRESERVED_RAW_MANIFEST" \
    --publication-commit "$AGRIBRAIN_PUBLICATION_CODE_COMMIT"

python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
echo "Structural publication-only recovery complete; no simulation was rerun."
echo "Verified archive: ${ARCHIVE_PATH}"
echo "Archive receipt:  ${RECEIPT_PATH}"
