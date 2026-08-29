#!/bin/bash
# Authorize and launch publication-only recovery for completed core and
# structural simulations whose original publishers failed.
#
# Run this on the Slurm login node from a CLEAN publication-repair commit.
# It performs no simulation submission. Two replacement publishers and their
# afterok combined-evidence finalizer are submitted held. The publishers are
# bound to self-hashed recovery receipts; all three jobs are released only
# after the complete two-run authorization has validated.

set -euo pipefail

for required in AGRIBRAIN_PARTITION AGRIBRAIN_RECOVERY_CONTROL_ROOT \
    AGRIBRAIN_CORE_RAW_SOURCE_SNAPSHOT AGRIBRAIN_STRUCTURAL_RUN_DIR \
    AGRIBRAIN_SIMULATION_COMMIT AGRIBRAIN_CORE_RUN_TAG \
    AGRIBRAIN_STRUCTURAL_RUN_TAG AGRIBRAIN_CORE_FAILED_PUBLISHER_JOB_ID \
    AGRIBRAIN_STRUCTURAL_FAILED_PUBLISHER_JOB_ID \
    AGRIBRAIN_CORE_FAILED_STDOUT AGRIBRAIN_CORE_FAILED_STDERR \
    AGRIBRAIN_STRUCTURAL_FAILED_STDOUT AGRIBRAIN_STRUCTURAL_FAILED_STDERR; do
    if [ -z "${!required:-}" ]; then
        echo "BLOCK: publication recovery requires ${required}."
        exit 1
    fi
done
for command in git sbatch scontrol sacct scancel; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "BLOCK: required command is unavailable: ${command}"
        exit 1
    fi
done

CORE_RECOVERY_JOB=""
STRUCTURAL_RECOVERY_JOB=""
FINALIZER_RECOVERY_JOB=""
STRUCTURAL_CANONICAL_RAW=""
STRUCTURAL_CANONICAL_RECEIPT=""
STRUCTURAL_RAW_MANIFEST=""
STRUCTURAL_RECOVERY_RECEIPT=""
STRUCTURAL_CANONICAL_RAW_CREATED=false
STRUCTURAL_CANONICAL_RECEIPT_CREATED=false
RELEASE_ATTEMPTED=false
require_user_held_job() {
    local job_id="$1"
    local record
    record="$(scontrol show job -o "$job_id")" || return 1
    case " $record " in
        *" JobId=${job_id} "*" JobState=PENDING "*" Reason=JobHeldUser "*) ;;
        *)
            echo "BLOCK: recovery job ${job_id} is not PENDING/User-held."
            return 1
            ;;
    esac
}
require_held_finalizer_dependency() {
    local record
    local dependency=""
    local field
    local normalized
    require_user_held_job "$FINALIZER_RECOVERY_JOB"
    record="$(scontrol show job -o "$FINALIZER_RECOVERY_JOB")" || return 1
    for field in $record; do
        case "$field" in Dependency=*) dependency="${field#Dependency=}";; esac
    done
    normalized="${dependency//\(unfulfilled\)/}"
    normalized="${normalized//\(satisfied\)/}"
    case "$normalized" in
        "afterok:${CORE_RECOVERY_JOB}:${STRUCTURAL_RECOVERY_JOB}"|\
        "afterok:${STRUCTURAL_RECOVERY_JOB}:${CORE_RECOVERY_JOB}"|\
        "afterok:${CORE_RECOVERY_JOB},afterok:${STRUCTURAL_RECOVERY_JOB}"|\
        "afterok:${STRUCTURAL_RECOVERY_JOB},afterok:${CORE_RECOVERY_JOB}") ;;
        *)
            echo "BLOCK: combined finalizer lacks the exact two-publisher afterok dependency."
            return 1
            ;;
    esac
}
require_not_user_held_job() {
    local job_id="$1"
    local record
    record="$(scontrol show job -o "$job_id")" || return 1
    case " $record " in
        *" JobId=${job_id} "*) ;;
        *)
            echo "BLOCK: cannot prove release state for recovery job ${job_id}."
            return 1
            ;;
    esac
    case " $record " in
        *" JobState=PENDING "*" Reason=JobHeldUser "*)
            echo "BLOCK: recovery job ${job_id} remained user-held after release."
            return 1
            ;;
        *)
            echo "Release transition observed for recovery job ${job_id}."
            ;;
    esac
}
cancel_held_recovery_jobs_on_failure() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        set +e
        local all_confirmed_held=true
        local job_id
        local record
        local state
        local reason
        for job_id in "$FINALIZER_RECOVERY_JOB" "$STRUCTURAL_RECOVERY_JOB" \
            "$CORE_RECOVERY_JOB"; do
            if [[ "$job_id" =~ ^[1-9][0-9]*$ ]]; then
                state="UNKNOWN"
                reason="UNKNOWN"
                if record="$(scontrol show job -o "$job_id" 2>/dev/null)"; then
                    for field in $record; do
                        case "$field" in
                            JobState=*) state="${field#JobState=}" ;;
                            Reason=*) reason="${field#Reason=}" ;;
                        esac
                    done
                fi
                echo "Setup-failure state: job=${job_id} state=${state} reason=${reason}." >&2
                if [ "$state" != PENDING ] || [ "$reason" != JobHeldUser ]; then
                    all_confirmed_held=false
                fi
                if scancel "$job_id"; then
                    echo "Cancellation requested for publication-only recovery job ${job_id}." >&2
                else
                    echo "WARNING: cancellation request failed for recovery job ${job_id}; inspect it manually." >&2
                fi
            fi
        done
        local preserve_canonical=false
        if [ "$RELEASE_ATTEMPTED" = true ] \
            || [ "$all_confirmed_held" != true ]; then
            preserve_canonical=true
            echo "Preserving canonical recovery evidence because release state is partial or uncertain." >&2
        fi
        if [ "$preserve_canonical" = false ] \
            && [ "$STRUCTURAL_CANONICAL_RECEIPT_CREATED" = true ]; then
            if [ -f "$STRUCTURAL_CANONICAL_RECEIPT" ] \
                && [ ! -L "$STRUCTURAL_CANONICAL_RECEIPT" ] \
                && cmp --silent -- "$STRUCTURAL_RECOVERY_RECEIPT" \
                    "$STRUCTURAL_CANONICAL_RECEIPT"; then
                rm -- "$STRUCTURAL_CANONICAL_RECEIPT"
            else
                echo "WARNING: canonical structural recovery receipt changed; remove/audit it manually." >&2
            fi
        fi
        if [ "$preserve_canonical" = false ] \
            && [ "$STRUCTURAL_CANONICAL_RAW_CREATED" = true ]; then
            if [ -f "$STRUCTURAL_CANONICAL_RAW" ] \
                && [ ! -L "$STRUCTURAL_CANONICAL_RAW" ] \
                && cmp --silent -- "$STRUCTURAL_RAW_MANIFEST" \
                    "$STRUCTURAL_CANONICAL_RAW"; then
                rm -- "$STRUCTURAL_CANONICAL_RAW"
            else
                echo "WARNING: canonical structural raw manifest changed; remove/audit it manually." >&2
            fi
        fi
    fi
    exit "$status"
}
trap cancel_held_recovery_jobs_on_failure EXIT
PUBLICATION_PYTHON="${AGRIBRAIN_PYTHON_BIN:-python3.11}"
if ! command -v "$PUBLICATION_PYTHON" >/dev/null 2>&1; then
    echo "BLOCK: Python 3.11 is required for recovery publication."
    exit 1
fi

REPAIR_REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "BLOCK: cannot resolve the publication-repair repository."
    exit 1
}
if [ "$(pwd -P)" != "$(cd "$REPAIR_REPO" && pwd -P)" ]; then
    echo "BLOCK: run hpc/publication_recovery_run.sh from the repository root."
    exit 1
fi
if [ -n "$(git status --porcelain=v1 --untracked-files=all)" ]; then
    echo "BLOCK: publication-repair checkout is dirty; commit and retest first."
    exit 1
fi

PUBLICATION_COMMIT="$(git rev-parse HEAD)"
if [ "$PUBLICATION_COMMIT" = "$AGRIBRAIN_SIMULATION_COMMIT" ]; then
    echo "BLOCK: publication recovery requires a distinct repair commit."
    exit 1
fi
case "$AGRIBRAIN_RECOVERY_CONTROL_ROOT" in
    /*) ;;
    *) echo "BLOCK: AGRIBRAIN_RECOVERY_CONTROL_ROOT must be absolute."; exit 1;;
esac
if [ -e "$AGRIBRAIN_RECOVERY_CONTROL_ROOT" ] \
    || [ -L "$AGRIBRAIN_RECOVERY_CONTROL_ROOT" ]; then
    echo "BLOCK: recovery control root already exists: ${AGRIBRAIN_RECOVERY_CONTROL_ROOT}"
    exit 1
fi

if [ -L "$AGRIBRAIN_CORE_RAW_SOURCE_SNAPSHOT" ] \
    || [ ! -d "$AGRIBRAIN_CORE_RAW_SOURCE_SNAPSHOT" ]; then
    echo "BLOCK: supplied core snapshot is missing or symlinked."
    exit 1
fi
if [ -L "$AGRIBRAIN_STRUCTURAL_RUN_DIR" ] \
    || [ ! -d "$AGRIBRAIN_STRUCTURAL_RUN_DIR" ]; then
    echo "BLOCK: supplied structural run is missing or symlinked."
    exit 1
fi
CORE_RAW_SNAPSHOT="$(cd "$AGRIBRAIN_CORE_RAW_SOURCE_SNAPSHOT" && pwd -P)"
STRUCTURAL_RUN_DIR="$(cd "$AGRIBRAIN_STRUCTURAL_RUN_DIR" && pwd -P)"
"$PUBLICATION_PYTHON" - "$AGRIBRAIN_RECOVERY_CONTROL_ROOT" "$REPAIR_REPO" \
    "$CORE_RAW_SNAPSHOT" "$STRUCTURAL_RUN_DIR" <<'PY'
import sys
from pathlib import Path

control = Path(sys.argv[1]).resolve()
for raw in sys.argv[2:]:
    protected = Path(raw).resolve(strict=True)
    try:
        control.relative_to(protected)
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"BLOCK: recovery control root must be outside protected path {protected}"
        )
PY
CORE_RESULTS="${CORE_RAW_SNAPSHOT}/mvp/simulation/results"
CORE_RAW_SEEDS="${CORE_RESULTS}/benchmark_seeds/${AGRIBRAIN_CORE_RUN_TAG}"
CORE_RAW_STRESS="${CORE_RESULTS}/stress_runs/${AGRIBRAIN_CORE_RUN_TAG}"
CORE_RAW_H3="${CORE_RESULTS}/decision_ledger_h3/${AGRIBRAIN_CORE_RUN_TAG}"
CORE_ORIGINAL_RECEIPT="${CORE_RESULTS}/core_submission_receipts/${AGRIBRAIN_CORE_RUN_TAG}.json"
STRUCTURAL_ORIGINAL_RECEIPT="${STRUCTURAL_RUN_DIR}/slurm_submission.json"
for path in "$CORE_RAW_SEEDS" "$CORE_RAW_STRESS" "$CORE_RAW_H3"; do
    if [ ! -d "$path" ] || [ -L "$path" ]; then
        echo "BLOCK: preserved core directory is missing or unsafe: ${path}"
        exit 1
    fi
done
for output in completion_status.json structural_sensitivity_analysis.json \
    publication_environment.json slurm_simulation_accounting.json \
    structural_sensitivity_artifact_manifest.json \
    "structural_sensitivity_evidence_${AGRIBRAIN_STRUCTURAL_RUN_TAG}.tar.gz" \
    structural_sensitivity_archive_receipt.json \
    structural_sensitivity_summary.csv structural_sensitivity_summary.png \
    structural_sensitivity_summary.pdf \
    structural_sensitivity_publication_receipt.json; do
    if [ -e "${STRUCTURAL_RUN_DIR}/${output}" ] \
        || [ -L "${STRUCTURAL_RUN_DIR}/${output}" ]; then
        echo "BLOCK: structural derived output already exists: ${STRUCTURAL_RUN_DIR}/${output}"
        exit 1
    fi
done
for path in "$CORE_ORIGINAL_RECEIPT" "$STRUCTURAL_ORIGINAL_RECEIPT" \
    "$AGRIBRAIN_CORE_FAILED_STDOUT" "$AGRIBRAIN_CORE_FAILED_STDERR" \
    "$AGRIBRAIN_STRUCTURAL_FAILED_STDOUT" \
    "$AGRIBRAIN_STRUCTURAL_FAILED_STDERR"; do
    if [ ! -f "$path" ] || [ -L "$path" ]; then
        echo "BLOCK: required recovery evidence is missing or unsafe: ${path}"
        exit 1
    fi
done

mkdir -p "$AGRIBRAIN_RECOVERY_CONTROL_ROOT"
CONTROL_ROOT="$(cd "$AGRIBRAIN_RECOVERY_CONTROL_ROOT" && pwd -P)"
mkdir -p "$CONTROL_ROOT/accounting" "$CONTROL_ROOT/raw_manifests/core" \
    "$CONTROL_ROOT/raw_manifests/structural" \
    "$CONTROL_ROOT/recovery_receipts/core" \
    "$CONTROL_ROOT/recovery_receipts/structural" \
    "$CONTROL_ROOT/recovery_receipts/finalizer" \
    "$CONTROL_ROOT/recovery_logs"

CORE_FAILED_ACCOUNTING="${CONTROL_ROOT}/accounting/${AGRIBRAIN_CORE_FAILED_PUBLISHER_JOB_ID}.json"
STRUCTURAL_FAILED_ACCOUNTING="${CONTROL_ROOT}/accounting/${AGRIBRAIN_STRUCTURAL_FAILED_PUBLISHER_JOB_ID}.json"
"$PUBLICATION_PYTHON" hpc/capture_failed_publisher_accounting.py \
    --job-id "$AGRIBRAIN_CORE_FAILED_PUBLISHER_JOB_ID" \
    --output "$CORE_FAILED_ACCOUNTING"
"$PUBLICATION_PYTHON" hpc/capture_failed_publisher_accounting.py \
    --job-id "$AGRIBRAIN_STRUCTURAL_FAILED_PUBLISHER_JOB_ID" \
    --output "$STRUCTURAL_FAILED_ACCOUNTING"

CORE_SIMULATION_TREE="$("$PUBLICATION_PYTHON" - "$CORE_ORIGINAL_RECEIPT" \
    "$AGRIBRAIN_CORE_RUN_TAG" "$AGRIBRAIN_SIMULATION_COMMIT" <<'PY'
import json, re, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if p.get("run_tag") != sys.argv[2] or p.get("source_commit") != sys.argv[3]:
    raise SystemExit("BLOCK: core receipt identity differs from requested recovery")
d = p.get("source_tree_sha256", "")
if re.fullmatch(r"[0-9a-f]{64}", str(d)) is None:
    raise SystemExit("BLOCK: core receipt source-tree digest is invalid")
print(d)
PY
)"
STRUCTURAL_SIMULATION_TREE="$("$PUBLICATION_PYTHON" - "$STRUCTURAL_ORIGINAL_RECEIPT" \
    "$AGRIBRAIN_STRUCTURAL_RUN_TAG" "$AGRIBRAIN_SIMULATION_COMMIT" <<'PY'
import json, re, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if p.get("run_tag") != sys.argv[2] or p.get("source_commit") != sys.argv[3]:
    raise SystemExit("BLOCK: structural receipt identity differs from requested recovery")
d = p.get("source_tree_sha256", "")
if re.fullmatch(r"[0-9a-f]{64}", str(d)) is None:
    raise SystemExit("BLOCK: structural receipt source-tree digest is invalid")
print(d)
PY
)"
if [ "$CORE_SIMULATION_TREE" != "$STRUCTURAL_SIMULATION_TREE" ]; then
    echo "BLOCK: core and structural receipts bind different simulation trees."
    exit 1
fi

CORE_RAW_MANIFEST="${CONTROL_ROOT}/raw_manifests/core/${AGRIBRAIN_CORE_RUN_TAG}.json"
STRUCTURAL_RAW_MANIFEST="${CONTROL_ROOT}/raw_manifests/structural/${AGRIBRAIN_STRUCTURAL_RUN_TAG}.json"
"$PUBLICATION_PYTHON" hpc/preserved_raw_manifest.py create \
    --manifest "$CORE_RAW_MANIFEST" --kind core \
    --run-tag "$AGRIBRAIN_CORE_RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --simulation-source-tree-sha256 "$CORE_SIMULATION_TREE" \
    --input-root "benchmark_seed_outputs=${CORE_RAW_SEEDS}" \
    --input-root "stress_outputs=${CORE_RAW_STRESS}" \
    --input-root "h3_decision_ledgers=${CORE_RAW_H3}" \
    --input-file "core_submission_receipt.json=${CORE_ORIGINAL_RECEIPT}"
"$PUBLICATION_PYTHON" hpc/preserved_raw_manifest.py create \
    --manifest "$STRUCTURAL_RAW_MANIFEST" --kind structural \
    --run-tag "$AGRIBRAIN_STRUCTURAL_RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --simulation-source-tree-sha256 "$STRUCTURAL_SIMULATION_TREE" \
    --input-root "logs=${STRUCTURAL_RUN_DIR}/logs" \
    --input-root "runtime_receipts=${STRUCTURAL_RUN_DIR}/runtime_receipts" \
    --input-root "tasks=${STRUCTURAL_RUN_DIR}/tasks" \
    --input-file "episode_accounting.json=${STRUCTURAL_RUN_DIR}/episode_accounting.json" \
    --input-file "experiment_protocol.json=${STRUCTURAL_RUN_DIR}/experiment_protocol.json" \
    --input-file "lhs_design.csv=${STRUCTURAL_RUN_DIR}/lhs_design.csv" \
    --input-file "lhs_design.json=${STRUCTURAL_RUN_DIR}/lhs_design.json" \
    --input-file "parameter_registry.json=${STRUCTURAL_RUN_DIR}/parameter_registry.json" \
    --input-file "run_plan.json=${STRUCTURAL_RUN_DIR}/run_plan.json" \
    --input-file "slurm_submission.json=${STRUCTURAL_ORIGINAL_RECEIPT}" \
    --input-file "task_manifest.json=${STRUCTURAL_RUN_DIR}/task_manifest.json" \
    --input-file "task_manifest.jsonl=${STRUCTURAL_RUN_DIR}/task_manifest.jsonl"

# Core publication writes repository-ignored results. Structural validation and
# final combined validation require clean trees throughout, so each job gets a
# separate detached worktree at the identical reviewed publication commit.
CORE_PUBLICATION_SNAPSHOT="${CONTROL_ROOT}/publication_source_core"
STRUCTURAL_PUBLICATION_SNAPSHOT="${CONTROL_ROOT}/publication_source_structural"
FINALIZER_PUBLICATION_SNAPSHOT="${CONTROL_ROOT}/publication_source_finalizer"
for snapshot in "$CORE_PUBLICATION_SNAPSHOT" \
    "$STRUCTURAL_PUBLICATION_SNAPSHOT" "$FINALIZER_PUBLICATION_SNAPSHOT"; do
    git worktree add --detach "$snapshot" "$PUBLICATION_COMMIT"
done
CORE_PUBLICATION_SNAPSHOT="$(cd "$CORE_PUBLICATION_SNAPSHOT" && pwd -P)"
STRUCTURAL_PUBLICATION_SNAPSHOT="$(cd "$STRUCTURAL_PUBLICATION_SNAPSHOT" && pwd -P)"
FINALIZER_PUBLICATION_SNAPSHOT="$(cd "$FINALIZER_PUBLICATION_SNAPSHOT" && pwd -P)"
prepare_recovery_venv() {
    local snapshot="$1"
    local run_tag="$2"
    local venv=".publication_venvs/${run_tag}"
    (
        cd "$snapshot"
        mkdir -p .publication_venvs
        if ! mkdir "$venv"; then
            echo "BLOCK: recovery venv path already exists: ${venv}"
            exit 1
        fi
        "$PUBLICATION_PYTHON" -m venv "$venv"
        source "$venv/bin/activate"
        python -m pip install -r agribrain/backend/requirements-lock.txt --quiet
        mkdir "$venv/backend-build-source"
        cp -a agribrain/backend/. "$venv/backend-build-source/"
        python -m pip install "$venv/backend-build-source" --no-deps --quiet
        python -m pip check
        source hpc/publication_env.sh
        RUN_TAG="$run_tag" AGRIBRAIN_VENV="$venv" \
            AGRIBRAIN_GIT_COMMIT="$PUBLICATION_COMMIT" \
            python hpc/capture_publication_environment.py --validate-only
    )
}
prepare_recovery_venv "$CORE_PUBLICATION_SNAPSHOT" "$AGRIBRAIN_CORE_RUN_TAG"
prepare_recovery_venv "$STRUCTURAL_PUBLICATION_SNAPSHOT" \
    "$AGRIBRAIN_STRUCTURAL_RUN_TAG"
prepare_recovery_venv "$FINALIZER_PUBLICATION_SNAPSHOT" \
    "finalizer_${AGRIBRAIN_CORE_RUN_TAG}"

snapshot_source_digest() {
    local snapshot="$1"
    (
        cd "$snapshot"
        while IFS= read -r -d '' tracked_path; do
            case "$tracked_path" in
                mvp/simulation/results/*) ;;
                *) chmod a-w -- "$tracked_path";;
            esac
        done < <(git ls-files -z)
        AGRIBRAIN_SOURCE_SNAPSHOT="$snapshot" \
            AGRIBRAIN_SOURCE_SNAPSHOT_MODE="detached_readonly_git_worktree_v1" \
            AGRIBRAIN_GIT_COMMIT="$PUBLICATION_COMMIT" \
            "$PUBLICATION_PYTHON" hpc/validate_source_snapshot.py --print-digest
    )
}
CORE_PUBLICATION_TREE_SHA256="$(snapshot_source_digest "$CORE_PUBLICATION_SNAPSHOT")"
STRUCTURAL_PUBLICATION_TREE_SHA256="$(
    snapshot_source_digest "$STRUCTURAL_PUBLICATION_SNAPSHOT"
)"
FINALIZER_PUBLICATION_TREE_SHA256="$(
    snapshot_source_digest "$FINALIZER_PUBLICATION_SNAPSHOT"
)"
if [ "$CORE_PUBLICATION_TREE_SHA256" != "$STRUCTURAL_PUBLICATION_TREE_SHA256" ] \
    || [ "$CORE_PUBLICATION_TREE_SHA256" != "$FINALIZER_PUBLICATION_TREE_SHA256" ]; then
    echo "BLOCK: publication worktrees do not have one identical source digest."
    exit 1
fi
PUBLICATION_TREE_SHA256="$CORE_PUBLICATION_TREE_SHA256"

CORE_RECOVERY_RECEIPT="${CONTROL_ROOT}/recovery_receipts/core/${AGRIBRAIN_CORE_RUN_TAG}.json"
STRUCTURAL_RECOVERY_RECEIPT="${CONTROL_ROOT}/recovery_receipts/structural/${AGRIBRAIN_STRUCTURAL_RUN_TAG}.json"
FINALIZER_AUTHORIZATION="${CONTROL_ROOT}/recovery_receipts/finalizer/${AGRIBRAIN_CORE_RUN_TAG}_${AGRIBRAIN_STRUCTURAL_RUN_TAG}.json"
STRUCTURAL_CANONICAL_RAW="${STRUCTURAL_RUN_DIR}/preserved_raw_manifests/${AGRIBRAIN_STRUCTURAL_RUN_TAG}.json"
STRUCTURAL_CANONICAL_RECEIPT="${STRUCTURAL_RUN_DIR}/publication_recovery_receipts/${AGRIBRAIN_STRUCTURAL_RUN_TAG}.json"
"$PUBLICATION_PYTHON" - "$STRUCTURAL_RUN_DIR" "$STRUCTURAL_CANONICAL_RAW" \
    "$STRUCTURAL_CANONICAL_RECEIPT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for raw_target in sys.argv[2:]:
    target = Path(raw_target)
    if not root.is_absolute() or not target.is_absolute():
        raise SystemExit("BLOCK: canonical structural evidence paths must be absolute")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SystemExit(
            "BLOCK: canonical structural evidence target escapes the run directory"
        ) from exc
    for component in (target, *target.parents):
        if component.is_symlink():
            raise SystemExit(
                f"BLOCK: canonical structural evidence has a symlink component: {component}"
            )
        if component.exists() and component != target and not component.is_dir():
            raise SystemExit(
                f"BLOCK: canonical structural evidence parent is not a directory: {component}"
            )
        if component == root:
            break
PY
for canonical in "$STRUCTURAL_CANONICAL_RAW" "$STRUCTURAL_CANONICAL_RECEIPT"; do
    if [ -e "$canonical" ] || [ -L "$canonical" ]; then
        echo "BLOCK: canonical structural recovery evidence already exists: ${canonical}"
        exit 1
    fi
done
CORE_BUNDLE="${CORE_PUBLICATION_SNAPSHOT}/publication_bundle_${AGRIBRAIN_CORE_RUN_TAG}"
CORE_ARCHIVE="${CORE_BUNDLE}/hpc_results_${AGRIBRAIN_CORE_RUN_TAG}.tar.gz"
CORE_ARCHIVE_RECEIPT="${CORE_BUNDLE}/publication_archive_receipt_${AGRIBRAIN_CORE_RUN_TAG}.json"
CORE_ARCHIVE_READY="${CORE_BUNDLE}/READY.json"
CORE_COMPLETE_BUNDLE="${CORE_PUBLICATION_SNAPSHOT}/mvp/simulation/results/complete_run_evidence/${AGRIBRAIN_CORE_RUN_TAG}"
CORE_COMPLETE_ARCHIVE="${CORE_COMPLETE_BUNDLE}/complete_run_evidence_${AGRIBRAIN_CORE_RUN_TAG}.tar.gz"
CORE_COMPLETE_RECEIPT="${CORE_COMPLETE_BUNDLE}/RECEIPT.json"
CORE_COMPLETE_READY="${CORE_COMPLETE_BUNDLE}/READY.json"
STRUCTURAL_ARCHIVE="${STRUCTURAL_RUN_DIR}/structural_sensitivity_evidence_${AGRIBRAIN_STRUCTURAL_RUN_TAG}.tar.gz"
STRUCTURAL_ARCHIVE_RECEIPT="${STRUCTURAL_RUN_DIR}/structural_sensitivity_archive_receipt.json"
FULL_EVIDENCE_DIR="${CONTROL_ROOT}/full_submission_evidence"
CORE_SUBMISSION="$(sbatch --parsable --hold \
    --partition="$AGRIBRAIN_PARTITION" --chdir="$CORE_PUBLICATION_SNAPSHOT" \
    --output="${CONTROL_ROOT}/recovery_logs/core_%j.out" \
    --error="${CONTROL_ROOT}/recovery_logs/core_%j.err" \
    --export=ALL,RUN_TAG="$AGRIBRAIN_CORE_RUN_TAG",AGRIBRAIN_SOURCE_SNAPSHOT="$CORE_PUBLICATION_SNAPSHOT",AGRIBRAIN_SOURCE_SNAPSHOT_MODE=detached_readonly_git_worktree_v1,AGRIBRAIN_VENV=".publication_venvs/${AGRIBRAIN_CORE_RUN_TAG}",AGRIBRAIN_SIMULATION_COMMIT="$AGRIBRAIN_SIMULATION_COMMIT",AGRIBRAIN_PUBLICATION_CODE_COMMIT="$PUBLICATION_COMMIT",AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256="$CORE_SIMULATION_TREE",AGRIBRAIN_PUBLICATION_SOURCE_TREE_SHA256="$PUBLICATION_TREE_SHA256",AGRIBRAIN_SOURCE_TREE_SHA256="$PUBLICATION_TREE_SHA256",AGRIBRAIN_ORIGINAL_CORE_RECEIPT="$CORE_ORIGINAL_RECEIPT",AGRIBRAIN_EXTERNAL_RECOVERY_RECEIPT="$CORE_RECOVERY_RECEIPT",AGRIBRAIN_EXTERNAL_RAW_MANIFEST="$CORE_RAW_MANIFEST",AGRIBRAIN_RAW_SEEDS_DIR="$CORE_RAW_SEEDS",AGRIBRAIN_RAW_STRESS_DIR="$CORE_RAW_STRESS",AGRIBRAIN_RAW_H3_LEDGER_DIR="$CORE_RAW_H3" \
    hpc/hpc_publish_recovery.sh)"
CORE_RECOVERY_JOB="${CORE_SUBMISSION%%;*}"
if [[ ! "$CORE_RECOVERY_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "BLOCK: sbatch returned an invalid core recovery job id."
    exit 1
fi
require_user_held_job "$CORE_RECOVERY_JOB"

STRUCTURAL_SUBMISSION="$(sbatch --parsable --hold \
    --partition="$AGRIBRAIN_PARTITION" --chdir="$STRUCTURAL_PUBLICATION_SNAPSHOT" \
    --output="${CONTROL_ROOT}/recovery_logs/structural_%j.out" \
    --error="${CONTROL_ROOT}/recovery_logs/structural_%j.err" \
    --export=ALL,RUN_TAG="$AGRIBRAIN_STRUCTURAL_RUN_TAG",AGRIBRAIN_SOURCE_SNAPSHOT="$STRUCTURAL_PUBLICATION_SNAPSHOT",AGRIBRAIN_SOURCE_SNAPSHOT_MODE=detached_readonly_git_worktree_v1,AGRIBRAIN_SOURCE_TREE_SHA256="$PUBLICATION_TREE_SHA256",AGRIBRAIN_SIMULATION_SOURCE_TREE_SHA256="$STRUCTURAL_SIMULATION_TREE",AGRIBRAIN_SIMULATION_COMMIT="$AGRIBRAIN_SIMULATION_COMMIT",AGRIBRAIN_PUBLICATION_CODE_COMMIT="$PUBLICATION_COMMIT",AGRIBRAIN_RECOVERY_RECEIPT="$STRUCTURAL_CANONICAL_RECEIPT",AGRIBRAIN_PRESERVED_RAW_MANIFEST="$STRUCTURAL_CANONICAL_RAW",AGRIBRAIN_RECOVERY_LOG_DIR="${CONTROL_ROOT}/recovery_logs",AGRIBRAIN_VENV=".publication_venvs/${AGRIBRAIN_STRUCTURAL_RUN_TAG}",AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT="$AGRIBRAIN_SIMULATION_COMMIT",AGRIBRAIN_SENSITIVITY_ROOT="$(dirname "$STRUCTURAL_RUN_DIR")",SENSITIVITY_RUN_DIR="$STRUCTURAL_RUN_DIR",SENSITIVITY_RUN_PLAN="${STRUCTURAL_RUN_DIR}/run_plan.json" \
    hpc/hpc_sensitivity_publish_recovery.sh)"
STRUCTURAL_RECOVERY_JOB="${STRUCTURAL_SUBMISSION%%;*}"
if [[ ! "$STRUCTURAL_RECOVERY_JOB" =~ ^[1-9][0-9]*$ ]] \
    || [ "$STRUCTURAL_RECOVERY_JOB" = "$CORE_RECOVERY_JOB" ]; then
    echo "BLOCK: sbatch returned an invalid or repeated structural recovery job id."
    exit 1
fi
require_user_held_job "$STRUCTURAL_RECOVERY_JOB"

FINALIZER_SUBMISSION="$(sbatch --parsable --hold \
    --partition="$AGRIBRAIN_PARTITION" --chdir="$FINALIZER_PUBLICATION_SNAPSHOT" \
    --dependency="afterok:${CORE_RECOVERY_JOB}:${STRUCTURAL_RECOVERY_JOB}" \
    --time="${AGRIBRAIN_FINALIZER_WALLTIME:-08:00:00}" \
    --mem="${AGRIBRAIN_FINALIZER_MEMORY:-32G}" \
    --cpus-per-task="${AGRIBRAIN_FINALIZER_CPUS:-4}" \
    --output="${CONTROL_ROOT}/recovery_logs/finalizer_%j.out" \
    --error="${CONTROL_ROOT}/recovery_logs/finalizer_%j.err" \
    --export=ALL,RUN_TAG="finalizer_${AGRIBRAIN_CORE_RUN_TAG}",AGRIBRAIN_SOURCE_SNAPSHOT="$FINALIZER_PUBLICATION_SNAPSHOT",AGRIBRAIN_SOURCE_SNAPSHOT_MODE=detached_readonly_git_worktree_v1,AGRIBRAIN_SOURCE_TREE_SHA256="$PUBLICATION_TREE_SHA256",AGRIBRAIN_GIT_COMMIT="$PUBLICATION_COMMIT",AGRIBRAIN_SIMULATION_COMMIT="$AGRIBRAIN_SIMULATION_COMMIT",AGRIBRAIN_PUBLICATION_CODE_COMMIT="$PUBLICATION_COMMIT",AGRIBRAIN_VENV=".publication_venvs/finalizer_${AGRIBRAIN_CORE_RUN_TAG}",AGRIBRAIN_CORE_RECOVERY_JOB_ID="$CORE_RECOVERY_JOB",AGRIBRAIN_STRUCTURAL_RECOVERY_JOB_ID="$STRUCTURAL_RECOVERY_JOB",AGRIBRAIN_FINALIZER_AUTHORIZATION="$FINALIZER_AUTHORIZATION",AGRIBRAIN_FULL_EVIDENCE_DIR="$FULL_EVIDENCE_DIR",AGRIBRAIN_CORE_ARCHIVE="$CORE_ARCHIVE",AGRIBRAIN_CORE_ARCHIVE_RECEIPT="$CORE_ARCHIVE_RECEIPT",AGRIBRAIN_CORE_ARCHIVE_READY="$CORE_ARCHIVE_READY",AGRIBRAIN_CORE_COMPLETE_ARCHIVE="$CORE_COMPLETE_ARCHIVE",AGRIBRAIN_CORE_COMPLETE_RECEIPT="$CORE_COMPLETE_RECEIPT",AGRIBRAIN_CORE_COMPLETE_READY="$CORE_COMPLETE_READY",AGRIBRAIN_STRUCTURAL_ARCHIVE="$STRUCTURAL_ARCHIVE",AGRIBRAIN_STRUCTURAL_ARCHIVE_RECEIPT="$STRUCTURAL_ARCHIVE_RECEIPT" \
    hpc/hpc_full_submission_recovery.sh)"
FINALIZER_RECOVERY_JOB="${FINALIZER_SUBMISSION%%;*}"
if [[ ! "$FINALIZER_RECOVERY_JOB" =~ ^[1-9][0-9]*$ ]] \
    || [ "$FINALIZER_RECOVERY_JOB" = "$CORE_RECOVERY_JOB" ] \
    || [ "$FINALIZER_RECOVERY_JOB" = "$STRUCTURAL_RECOVERY_JOB" ]; then
    echo "BLOCK: sbatch returned an invalid or repeated finalizer job id."
    exit 1
fi
require_held_finalizer_dependency
PYTHONPATH="$FINALIZER_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$FINALIZER_PUBLICATION_SNAPSHOT/hpc/finalizer_submission_authorization.py" create \
    --receipt "$FINALIZER_AUTHORIZATION" \
    --finalizer-job-id "$FINALIZER_RECOVERY_JOB" \
    --core-publisher-job-id "$CORE_RECOVERY_JOB" \
    --structural-publisher-job-id "$STRUCTURAL_RECOVERY_JOB"
PYTHONPATH="$FINALIZER_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$FINALIZER_PUBLICATION_SNAPSHOT/hpc/finalizer_submission_authorization.py" validate \
    --receipt "$FINALIZER_AUTHORIZATION" \
    --finalizer-job-id "$FINALIZER_RECOVERY_JOB" \
    --core-publisher-job-id "$CORE_RECOVERY_JOB" \
    --structural-publisher-job-id "$STRUCTURAL_RECOVERY_JOB" \
    --require-live-held

PYTHONPATH="$CORE_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$CORE_PUBLICATION_SNAPSHOT/hpc/publication_recovery_receipt.py" create \
    --output "$CORE_RECOVERY_RECEIPT" --repo-root "$CORE_PUBLICATION_SNAPSHOT" \
    --kind core --run-tag "$AGRIBRAIN_CORE_RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$PUBLICATION_COMMIT" \
    --original-submission-receipt "$CORE_ORIGINAL_RECEIPT" \
    --failed-accounting-record "$CORE_FAILED_ACCOUNTING" \
    --failed-stdout "$AGRIBRAIN_CORE_FAILED_STDOUT" \
    --failed-stderr "$AGRIBRAIN_CORE_FAILED_STDERR" \
    --raw-output-manifest "$CORE_RAW_MANIFEST" \
    --held-recovery-publisher-job-id "$CORE_RECOVERY_JOB" \
    --reason-code terminal_failed_publisher_publication_only_recovery
PYTHONPATH="$STRUCTURAL_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$STRUCTURAL_PUBLICATION_SNAPSHOT/hpc/publication_recovery_receipt.py" create \
    --output "$STRUCTURAL_RECOVERY_RECEIPT" \
    --repo-root "$STRUCTURAL_PUBLICATION_SNAPSHOT" \
    --kind structural --run-tag "$AGRIBRAIN_STRUCTURAL_RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$PUBLICATION_COMMIT" \
    --original-submission-receipt "$STRUCTURAL_ORIGINAL_RECEIPT" \
    --failed-accounting-record "$STRUCTURAL_FAILED_ACCOUNTING" \
    --failed-stdout "$AGRIBRAIN_STRUCTURAL_FAILED_STDOUT" \
    --failed-stderr "$AGRIBRAIN_STRUCTURAL_FAILED_STDERR" \
    --raw-output-manifest "$STRUCTURAL_RAW_MANIFEST" \
    --held-recovery-publisher-job-id "$STRUCTURAL_RECOVERY_JOB" \
    --reason-code terminal_failed_publisher_publication_only_recovery

PYTHONPATH="$CORE_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$CORE_PUBLICATION_SNAPSHOT/hpc/publication_recovery_receipt.py" validate \
    --receipt "$CORE_RECOVERY_RECEIPT" \
    --original-submission-receipt "$CORE_ORIGINAL_RECEIPT" --kind core \
    --run-tag "$AGRIBRAIN_CORE_RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$PUBLICATION_COMMIT" \
    --recovery-publisher-slurm-job-id "$CORE_RECOVERY_JOB"
PYTHONPATH="$STRUCTURAL_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$STRUCTURAL_PUBLICATION_SNAPSHOT/hpc/publication_recovery_receipt.py" validate \
    --receipt "$STRUCTURAL_RECOVERY_RECEIPT" \
    --original-submission-receipt "$STRUCTURAL_ORIGINAL_RECEIPT" --kind structural \
    --run-tag "$AGRIBRAIN_STRUCTURAL_RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$PUBLICATION_COMMIT" \
    --recovery-publisher-slurm-job-id "$STRUCTURAL_RECOVERY_JOB"

# Both jobs and both control-root receipts are now valid while the jobs remain
# held. Only now add the two canonical recovery files to the structural run.
require_user_held_job "$CORE_RECOVERY_JOB"
require_user_held_job "$STRUCTURAL_RECOVERY_JOB"
require_held_finalizer_dependency
PYTHONPATH="$FINALIZER_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$FINALIZER_PUBLICATION_SNAPSHOT/hpc/finalizer_submission_authorization.py" validate \
    --receipt "$FINALIZER_AUTHORIZATION" \
    --finalizer-job-id "$FINALIZER_RECOVERY_JOB" \
    --core-publisher-job-id "$CORE_RECOVERY_JOB" \
    --structural-publisher-job-id "$STRUCTURAL_RECOVERY_JOB" \
    --require-live-held
copy_canonical_evidence() {
    local source="$1"
    local target="$2"
    "$PUBLICATION_PYTHON" - "$source" "$target" <<'PY'
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
if source.is_symlink() or target.is_symlink() or target.exists():
    raise SystemExit(
        "BLOCK: canonical structural recovery evidence is unsafe or already exists"
    )
for component in target.parents:
    if component.is_symlink():
        raise SystemExit(
            f"BLOCK: canonical structural recovery parent is symlinked: {component}"
        )
    if component.exists() and not component.is_dir():
        raise SystemExit(
            f"BLOCK: canonical structural recovery parent is not a directory: {component}"
        )
target.parent.mkdir(parents=True, exist_ok=True)
if target.parent.is_symlink() or not target.parent.is_dir():
    raise SystemExit("BLOCK: canonical structural recovery parent became unsafe")
payload = source.read_bytes()
with target.open("xb") as handle:
    handle.write(payload)
if target.read_bytes() != payload:
    raise SystemExit("BLOCK: canonical structural recovery-evidence copy mismatch")
PY
}
copy_canonical_evidence "$STRUCTURAL_RAW_MANIFEST" "$STRUCTURAL_CANONICAL_RAW"
STRUCTURAL_CANONICAL_RAW_CREATED=true
copy_canonical_evidence "$STRUCTURAL_RECOVERY_RECEIPT" \
    "$STRUCTURAL_CANONICAL_RECEIPT"
STRUCTURAL_CANONICAL_RECEIPT_CREATED=true
PYTHONPATH="$STRUCTURAL_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$STRUCTURAL_PUBLICATION_SNAPSHOT/hpc/publication_recovery_receipt.py" validate \
    --receipt "$STRUCTURAL_CANONICAL_RECEIPT" \
    --original-submission-receipt "$STRUCTURAL_ORIGINAL_RECEIPT" --kind structural \
    --run-tag "$AGRIBRAIN_STRUCTURAL_RUN_TAG" \
    --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
    --publication-commit "$PUBLICATION_COMMIT" \
    --recovery-publisher-slurm-job-id "$STRUCTURAL_RECOVERY_JOB"

# No job is released until both independent authorizations have validated.
require_user_held_job "$CORE_RECOVERY_JOB"
require_user_held_job "$STRUCTURAL_RECOVERY_JOB"
require_held_finalizer_dependency
PYTHONPATH="$FINALIZER_PUBLICATION_SNAPSHOT" "$PUBLICATION_PYTHON" \
    "$FINALIZER_PUBLICATION_SNAPSHOT/hpc/finalizer_submission_authorization.py" validate \
    --receipt "$FINALIZER_AUTHORIZATION" \
    --finalizer-job-id "$FINALIZER_RECOVERY_JOB" \
    --core-publisher-job-id "$CORE_RECOVERY_JOB" \
    --structural-publisher-job-id "$STRUCTURAL_RECOVERY_JOB" \
    --require-live-held
RELEASE_ATTEMPTED=true
scontrol release "${CORE_RECOVERY_JOB},${STRUCTURAL_RECOVERY_JOB},${FINALIZER_RECOVERY_JOB}"
require_not_user_held_job "$CORE_RECOVERY_JOB"
require_not_user_held_job "$STRUCTURAL_RECOVERY_JOB"
require_not_user_held_job "$FINALIZER_RECOVERY_JOB"
trap - EXIT
echo "Authorized and released core recovery publisher ${CORE_RECOVERY_JOB}."
echo "Authorized and released structural recovery publisher ${STRUCTURAL_RECOVERY_JOB}."
echo "Released combined-evidence finalizer ${FINALIZER_RECOVERY_JOB} (afterok both publishers)."
echo "Combined receipt/READY destination: ${FULL_EVIDENCE_DIR}"
echo "No simulation arrays were submitted or rerun."
