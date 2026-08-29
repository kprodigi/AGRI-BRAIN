#!/bin/bash
# Publication-only dependent finalizer for a dual recovered core/structural run.
# This job executes no simulation. It validates both publication archives plus
# the lossless core run-evidence archive and atomically promotes one combined
# receipt/READY directory.

#SBATCH --job-name=agribrain-recovery-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

for required in SLURM_JOB_ID AGRIBRAIN_CORE_RECOVERY_JOB_ID \
    AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID AGRIBRAIN_FULL_EVIDENCE_DIR \
    AGRIBRAIN_CORE_ARCHIVE AGRIBRAIN_CORE_ARCHIVE_RECEIPT \
    AGRIBRAIN_CORE_ARCHIVE_READY AGRIBRAIN_CORE_COMPLETE_ARCHIVE \
    AGRIBRAIN_CORE_COMPLETE_RECEIPT AGRIBRAIN_CORE_COMPLETE_READY \
    AGRIBRAIN_STRUCTURAL_ARCHIVE AGRIBRAIN_STRUCTURAL_ARCHIVE_RECEIPT \
    AGRIBRAIN_SIMULATION_COMMIT AGRIBRAIN_PUBLICATION_CODE_COMMIT \
    AGRIBRAIN_FINALIZER_AUTHORIZATION AGRIBRAIN_GIT_COMMIT \
    AGRIBRAIN_SOURCE_SNAPSHOT AGRIBRAIN_VENV RUN_TAG; do
    if [ -z "${!required:-}" ]; then
        echo "BLOCK: combined recovery finalizer requires ${required}."
        exit 1
    fi
done
if [ "$PWD" != "$AGRIBRAIN_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: finalizer current directory differs lexically from source snapshot."
    exit 1
fi
case "$AGRIBRAIN_VENV" in
    /*) VENV_ABSOLUTE="$AGRIBRAIN_VENV" ;;
    *) VENV_ABSOLUTE="${PWD}/${AGRIBRAIN_VENV}" ;;
esac
for job_id in "$SLURM_JOB_ID" "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" \
    "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID"; do
    if [[ ! "$job_id" =~ ^[1-9][0-9]*$ ]]; then
        echo "BLOCK: combined recovery finalizer received an invalid Slurm job id."
        exit 1
    fi
done
if [ "$SLURM_JOB_ID" = "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" ] \
    || [ "$SLURM_JOB_ID" = "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID" ] \
    || [ "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" = \
        "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID" ]; then
    echo "BLOCK: recovery publishers and finalizer must be distinct Slurm jobs."
    exit 1
fi
if [ "$AGRIBRAIN_SIMULATION_COMMIT" = "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" ]; then
    echo "BLOCK: recovered combined evidence requires distinct simulation/publication commits."
    exit 1
fi
if [ "$AGRIBRAIN_GIT_COMMIT" != "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" ]; then
    echo "BLOCK: finalizer checkout commit differs from publication code commit."
    exit 1
fi
case "$AGRIBRAIN_FULL_EVIDENCE_DIR" in
    /*) ;;
    *) echo "BLOCK: AGRIBRAIN_FULL_EVIDENCE_DIR must be absolute."; exit 1;;
esac
case "$AGRIBRAIN_FINALIZER_AUTHORIZATION" in
    /*) ;;
    *) echo "BLOCK: finalizer authorization path must be absolute."; exit 1;;
esac
validate_lexical_inputs() {
    local path_python="${1:-python}"
    "$path_python" hpc/validate_lexical_path.py \
        --require-directory "$AGRIBRAIN_SOURCE_SNAPSHOT" \
        --require-directory "$PWD" \
        --require-directory "$VENV_ABSOLUTE" \
        --require-file "$VENV_ABSOLUTE/bin/activate" \
        --require-file "$AGRIBRAIN_CORE_ARCHIVE" \
        --require-file "$AGRIBRAIN_CORE_ARCHIVE_RECEIPT" \
        --require-file "$AGRIBRAIN_CORE_ARCHIVE_READY" \
        --require-file "$AGRIBRAIN_CORE_COMPLETE_ARCHIVE" \
        --require-file "$AGRIBRAIN_CORE_COMPLETE_RECEIPT" \
        --require-file "$AGRIBRAIN_CORE_COMPLETE_READY" \
        --require-file "$AGRIBRAIN_STRUCTURAL_ARCHIVE" \
        --require-file "$AGRIBRAIN_STRUCTURAL_ARCHIVE_RECEIPT" \
        --require-file "$AGRIBRAIN_FINALIZER_AUTHORIZATION" \
        --require-absent "$AGRIBRAIN_FULL_EVIDENCE_DIR"
}
BOOTSTRAP_PYTHON="${AGRIBRAIN_PYTHON_BIN:-python3.11}"
if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
    echo "BLOCK: required bootstrap Python is unavailable: ${BOOTSTRAP_PYTHON}"
    exit 1
fi
validate_lexical_inputs "$BOOTSTRAP_PYTHON"

source hpc/ensure_git_available.sh
source hpc/publication_env.sh
source "${AGRIBRAIN_VENV}/bin/activate"
python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
python hpc/validate_publication_env.py
python hpc/capture_publication_environment.py --validate-only
python hpc/finalizer_submission_authorization.py validate \
    --receipt "$AGRIBRAIN_FINALIZER_AUTHORIZATION" \
    --finalizer-job-id "$SLURM_JOB_ID" \
    --core-publisher-job-id "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" \
    --structural-publisher-job-id "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID"

OUTPUT_PARENT="$(dirname "$AGRIBRAIN_FULL_EVIDENCE_DIR")"
OUTPUT_NAME="$(basename "$AGRIBRAIN_FULL_EVIDENCE_DIR")"
mkdir -p "$OUTPUT_PARENT"
python hpc/validate_lexical_path.py \
    --require-directory "$OUTPUT_PARENT" \
    --require-absent "$AGRIBRAIN_FULL_EVIDENCE_DIR"
STAGE="$(mktemp -d "${OUTPUT_PARENT}/.${OUTPUT_NAME}.stage.XXXXXX")"
python hpc/validate_lexical_path.py --require-directory "$STAGE"
RECEIPT_NAME="FULL_SUBMISSION_EVIDENCE_RECEIPT.json"
READY_NAME="READY.json"
AUTHORIZATION_NAME="FINALIZER_SUBMISSION_AUTHORIZATION.json"
ENVIRONMENT_NAME="FINALIZER_PUBLICATION_ENVIRONMENT.json"
python hpc/capture_publication_environment.py \
    --output "${STAGE}/${ENVIRONMENT_NAME}"
python - "$AGRIBRAIN_FINALIZER_AUTHORIZATION" \
    "${STAGE}/${AUTHORIZATION_NAME}" <<'PY'
import os
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
if not source.is_absolute() or source.is_symlink() or not source.is_file():
    raise SystemExit("BLOCK: finalizer authorization source is not a plain absolute file")
if any(parent.is_symlink() for parent in source.parents):
    raise SystemExit("BLOCK: finalizer authorization source has a symlinked parent")
if target.exists() or target.is_symlink():
    raise SystemExit("BLOCK: finalizer authorization staging target is occupied")
payload = source.read_bytes()
with target.open("xb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
if target.read_bytes() != payload:
    raise SystemExit("BLOCK: staged finalizer authorization differs from its source")
PY
python hpc/build_full_submission_evidence.py \
    --core-archive "$AGRIBRAIN_CORE_ARCHIVE" \
    --core-receipt "$AGRIBRAIN_CORE_ARCHIVE_RECEIPT" \
    --core-ready "$AGRIBRAIN_CORE_ARCHIVE_READY" \
    --core-complete-archive "$AGRIBRAIN_CORE_COMPLETE_ARCHIVE" \
    --core-complete-receipt "$AGRIBRAIN_CORE_COMPLETE_RECEIPT" \
    --core-complete-ready "$AGRIBRAIN_CORE_COMPLETE_READY" \
    --structural-archive "$AGRIBRAIN_STRUCTURAL_ARCHIVE" \
    --structural-receipt "$AGRIBRAIN_STRUCTURAL_ARCHIVE_RECEIPT" \
    --output "${STAGE}/${RECEIPT_NAME}"

python hpc/validate_full_submission_ready.py create \
    --directory "$STAGE" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" \
    --finalizer-job-id "$SLURM_JOB_ID" \
    --core-job-id "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" \
    --structural-job-id "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID" \
    --run-tag "$RUN_TAG"

python hpc/validate_source_checkout.py
python hpc/validate_source_snapshot.py
python hpc/validate_publication_env.py
python hpc/capture_publication_environment.py --validate-only
python hpc/finalizer_submission_authorization.py validate \
    --receipt "$AGRIBRAIN_FINALIZER_AUTHORIZATION" \
    --finalizer-job-id "$SLURM_JOB_ID" \
    --core-publisher-job-id "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" \
    --structural-publisher-job-id "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID"

# Recheck every consumed path immediately before promotion, then validate the
# exact four-file staging inventory and all literal/self-hash bindings.
validate_lexical_inputs
python hpc/validate_lexical_path.py \
    --require-directory "$STAGE" \
    --require-directory "$OUTPUT_PARENT" \
    --require-absent "$AGRIBRAIN_FULL_EVIDENCE_DIR"
python hpc/validate_full_submission_ready.py validate \
    --directory "$STAGE" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" \
    --finalizer-job-id "$SLURM_JOB_ID" \
    --core-job-id "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" \
    --structural-job-id "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID" \
    --run-tag "$RUN_TAG"

python - "$STAGE" "$AGRIBRAIN_FULL_EVIDENCE_DIR" <<'PY'
import os
import sys
from pathlib import Path

stage = Path(sys.argv[1]).absolute()
target = Path(sys.argv[2]).absolute()
if stage.is_symlink() or target.is_symlink() or target.exists():
    raise SystemExit("BLOCK: unsafe or occupied combined-evidence promotion path")
if stage.parent != target.parent:
    raise SystemExit("BLOCK: combined-evidence staging and target are not siblings")
os.rename(stage, target)
PY

python hpc/validate_lexical_path.py \
    --require-absent "$STAGE" \
    --require-directory "$AGRIBRAIN_FULL_EVIDENCE_DIR"
python hpc/validate_full_submission_ready.py validate \
    --directory "$AGRIBRAIN_FULL_EVIDENCE_DIR" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$AGRIBRAIN_PUBLICATION_CODE_COMMIT" \
    --finalizer-job-id "$SLURM_JOB_ID" \
    --core-job-id "$AGRIBRAIN_CORE_RECOVERY_JOB_ID" \
    --structural-job-id "$AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID" \
    --run-tag "$RUN_TAG"

echo "Combined recovered submission evidence is READY: ${AGRIBRAIN_FULL_EVIDENCE_DIR}"
echo "No simulation was submitted or rerun by finalizer job ${SLURM_JOB_ID}."
