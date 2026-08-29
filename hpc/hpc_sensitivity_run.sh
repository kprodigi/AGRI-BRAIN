#!/bin/bash
# Submit the isolated 100-point structural-sensitivity treatment.
#
# Required:
#   AGRIBRAIN_PARTITION=<slurm-partition>
#   AGRIBRAIN_SENSITIVITY_ROOT=/absolute/shared/scratch/path
#
# The external root is mandatory.  This workflow never writes structural
# endpoints into mvp/simulation/results, so they cannot be mistaken for the
# core 20-seed publication panel.
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
    echo "BLOCK: git is required for structural source verification."
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "BLOCK: sbatch is unavailable; run this on a Slurm login node."
    exit 1
fi
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "BLOCK: cannot resolve the repository root."
    exit 1
}
if [ "$(pwd -P)" != "$(cd "$REPO_ROOT" && pwd -P)" ]; then
    echo "BLOCK: run hpc/hpc_sensitivity_run.sh from the repository root."
    exit 1
fi

PARTITION="${AGRIBRAIN_PARTITION:-${SBATCH_PARTITION:-}}"
if [ -z "$PARTITION" ]; then
    echo "BLOCK: set AGRIBRAIN_PARTITION (or SBATCH_PARTITION)."
    exit 1
fi
if [ -z "${AGRIBRAIN_SENSITIVITY_ROOT:-}" ]; then
    echo "BLOCK: set AGRIBRAIN_SENSITIVITY_ROOT to an absolute shared-scratch path."
    exit 1
fi
if [ "${DETERMINISTIC_MODE:-false}" = "true" ]; then
    echo "BLOCK: structural publication sensitivity requires stochastic execution."
    exit 1
fi
case "$AGRIBRAIN_SENSITIVITY_ROOT" in
    /*) ;;
    *)
        echo "BLOCK: AGRIBRAIN_SENSITIVITY_ROOT must be absolute."
        exit 1
        ;;
esac

# Static evidence-wiring gate runs before creating a worktree, run directory,
# or virtual environment and therefore consumes no cluster compute on drift.
PUBLICATION_PYTHON_BIN="${AGRIBRAIN_PYTHON_BIN:-python3.11}"
if ! command -v "$PUBLICATION_PYTHON_BIN" >/dev/null 2>&1; then
    echo "BLOCK: Python 3.11 is required for structural publication evidence."
    echo "       Set AGRIBRAIN_PYTHON_BIN to the cluster's Python 3.11 executable."
    exit 1
fi
source hpc/publication_env.sh
"$PUBLICATION_PYTHON_BIN" hpc/validate_launch_preflight.py --workflow structural
"$PUBLICATION_PYTHON_BIN" hpc/validate_pinn_artifacts.py
"$PUBLICATION_PYTHON_BIN" hpc/validate_publication_env.py

GIT_COMMIT="$(git rev-parse HEAD)"
export AGRIBRAIN_GIT_COMMIT="$GIT_COMMIT"
export AGRIBRAIN_SENSITIVITY_SOURCE_COMMIT="$GIT_COMMIT"
"$PUBLICATION_PYTHON_BIN" hpc/validate_source_checkout.py
export RUN_TAG="sensitivity_${GIT_COMMIT:0:7}_$(date +%Y%m%d_%H%M%S)"
SOURCE_SNAPSHOT_PARENT="${REPO_ROOT}/.publication_sources"
SOURCE_SNAPSHOT_PATH="${SOURCE_SNAPSHOT_PARENT}/${RUN_TAG}"
mkdir -p "$SOURCE_SNAPSHOT_PARENT"
if [ -e "$SOURCE_SNAPSHOT_PATH" ]; then
    echo "BLOCK: source snapshot path already exists: ${SOURCE_SNAPSHOT_PATH}"
    exit 1
fi
if ! git worktree add --detach "$SOURCE_SNAPSHOT_PATH" "$GIT_COMMIT"; then
    echo "BLOCK: could not create the detached structural source snapshot."
    exit 1
fi
export AGRIBRAIN_SOURCE_SNAPSHOT="$(cd "$SOURCE_SNAPSHOT_PATH" && pwd -P)"
export AGRIBRAIN_SOURCE_SNAPSHOT_MODE="detached_readonly_git_worktree_v1"
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
export AGRIBRAIN_VENV=".publication_venvs/${RUN_TAG}"
export SENSITIVITY_RUN_DIR="${AGRIBRAIN_SENSITIVITY_ROOT%/}/${RUN_TAG}"
export SENSITIVITY_RUN_PLAN="${SENSITIVITY_RUN_DIR}/run_plan.json"

# Resolve the proposed location before creating anything.  Symlink resolution
# and Path.relative_to close string-prefix tricks such as repo-other/.
"$PUBLICATION_PYTHON_BIN" - "$REPO_ROOT" "$AGRIBRAIN_SENSITIVITY_ROOT" "$SENSITIVITY_RUN_DIR" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
if not root.is_absolute() or not run_dir.is_absolute():
    raise SystemExit("BLOCK: structural output paths must be absolute")
resolved_root = root.resolve()
resolved_run = run_dir.resolve()
try:
    resolved_root.relative_to(repo)
except ValueError:
    pass
else:
    raise SystemExit("BLOCK: structural root must be outside the repository")
try:
    resolved_run.relative_to(repo)
except ValueError:
    pass
else:
    raise SystemExit("BLOCK: structural run directory must be outside the repository")
PY

mkdir -p "$AGRIBRAIN_SENSITIVITY_ROOT"
if ! mkdir "$SENSITIVITY_RUN_DIR"; then
    echo "BLOCK: run directory already exists: ${SENSITIVITY_RUN_DIR}"
    exit 1
fi
mkdir -p "${SENSITIVITY_RUN_DIR}/logs" .publication_venvs
if ! mkdir "$AGRIBRAIN_VENV"; then
    echo "BLOCK: run-scoped venv already exists: ${AGRIBRAIN_VENV}"
    exit 1
fi

source hpc/publication_env.sh
"$PUBLICATION_PYTHON_BIN" hpc/validate_publication_env.py
"$PUBLICATION_PYTHON_BIN" hpc/validate_source_checkout.py
"$PUBLICATION_PYTHON_BIN" hpc/validate_structural_sensitivity_hpc.py --allow-missing-plan

"$PUBLICATION_PYTHON_BIN" -m venv "$AGRIBRAIN_VENV"
source "$AGRIBRAIN_VENV/bin/activate"
python -m pip install -r agribrain/backend/requirements-lock.txt --quiet
BACKEND_BUILD_SRC="${AGRIBRAIN_VENV}/backend-build-source"
mkdir "$BACKEND_BUILD_SRC"
cp -a agribrain/backend/. "$BACKEND_BUILD_SRC/"
python -m pip install "$BACKEND_BUILD_SRC" --no-deps --quiet
python -m pip check
python hpc/validate_source_checkout.py
python hpc/capture_publication_environment.py --validate-only
python hpc/validate_pinn_artifacts.py

echo "=== Generate and audit the immutable structural run plan ==="
python -m mvp.simulation.sensitivity.run_structural_sensitivity generate \
    --output-dir "$SENSITIVITY_RUN_DIR" \
    --run-tag "$RUN_TAG"
python hpc/validate_structural_sensitivity_hpc.py

# Freeze and hash every tracked source byte outside the core results tree.
# Structural workers use only this detached snapshot and verify it both before
# and after each manifest task.
while IFS= read -r -d '' tracked_path; do
    case "$tracked_path" in
        mvp/simulation/results/*) ;;
        *) chmod a-w -- "$tracked_path" ;;
    esac
done < <(git ls-files -z)
export AGRIBRAIN_SOURCE_TREE_SHA256="$(
    python hpc/validate_source_snapshot.py --print-digest
)"
if [ -z "$AGRIBRAIN_SOURCE_TREE_SHA256" ]; then
    echo "BLOCK: structural source snapshot digest was not produced."
    exit 1
fi
python hpc/validate_source_snapshot.py

TASK_COUNT="$(python - "$SENSITIVITY_RUN_PLAN" <<'PY'
import json
import sys
from pathlib import Path
plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
manifest = json.loads(
    (plan_path.parent / plan["artifacts"]["task_manifest"]).read_text(encoding="utf-8")
)
print(manifest["n_tasks"])
PY
)"
if [ "$TASK_COUNT" != "3000" ]; then
    echo "BLOCK: immutable plan contains ${TASK_COUNT} tasks; expected 3000."
    exit 1
fi

# Chunks default to 1,000 indices so the workflow also fits clusters whose
# MaxArraySize is 1,001.  Chunks are chained with afterok: no later chunk or
# publisher can run after an earlier failure.
CHUNK_SIZE="${AGRIBRAIN_SENSITIVITY_ARRAY_CHUNK_SIZE:-1000}"
MAX_CONCURRENT="${AGRIBRAIN_SENSITIVITY_MAX_CONCURRENT:-50}"
case "$CHUNK_SIZE" in *[!0-9]*|'') echo "BLOCK: chunk size must be an integer."; exit 1;; esac
case "$MAX_CONCURRENT" in *[!0-9]*|'') echo "BLOCK: concurrency cap must be an integer."; exit 1;; esac
if [ "$CHUNK_SIZE" -lt 1 ] || [ "$CHUNK_SIZE" -gt 1000 ]; then
    echo "BLOCK: AGRIBRAIN_SENSITIVITY_ARRAY_CHUNK_SIZE must be in 1..1000."
    exit 1
fi
if [ "$MAX_CONCURRENT" -lt 1 ] || [ "$MAX_CONCURRENT" -gt 1000 ]; then
    echo "BLOCK: AGRIBRAIN_SENSITIVITY_MAX_CONCURRENT must be in 1..1000."
    exit 1
fi

TASK_JOB_IDS=()
TASK_OFFSETS=()
TASK_COUNTS=()
TASK_DEPENDENCIES=()
OFFSET=0
PREVIOUS_JOB=""
while [ "$OFFSET" -lt "$TASK_COUNT" ]; do
    REMAINING=$((TASK_COUNT - OFFSET))
    COUNT="$CHUNK_SIZE"
    if [ "$REMAINING" -lt "$COUNT" ]; then COUNT="$REMAINING"; fi
    LAST_LOCAL=$((COUNT - 1))
    SBATCH_ARGS=(
        --parsable
        --partition="$PARTITION"
        --chdir="$AGRIBRAIN_SOURCE_SNAPSHOT"
        --array="0-${LAST_LOCAL}%${MAX_CONCURRENT}"
        --export="ALL,SENSITIVITY_TASK_OFFSET=${OFFSET}"
        --output="${SENSITIVITY_RUN_DIR}/logs/task_%A_%a.out"
        --error="${SENSITIVITY_RUN_DIR}/logs/task_%A_%a.err"
    )
    DEPENDENCY="none"
    if [ -n "$PREVIOUS_JOB" ]; then
        SBATCH_ARGS+=(--dependency="afterok:${PREVIOUS_JOB}")
        DEPENDENCY="$PREVIOUS_JOB"
    fi
    RAW_JOB="$(sbatch "${SBATCH_ARGS[@]}" hpc/hpc_sensitivity_task.sh)"
    JOB_ID="${RAW_JOB%%;*}"
    case "$JOB_ID" in *[!0-9]*|'') echo "BLOCK: sbatch returned invalid job id: ${RAW_JOB}"; exit 1;; esac
    TASK_JOB_IDS+=("$JOB_ID")
    TASK_OFFSETS+=("$OFFSET")
    TASK_COUNTS+=("$COUNT")
    TASK_DEPENDENCIES+=("$DEPENDENCY")
    PREVIOUS_JOB="$JOB_ID"
    echo "Submitted structural tasks ${OFFSET}..$((OFFSET + COUNT - 1)) as ${JOB_ID}."
    OFFSET=$((OFFSET + COUNT))
done

RAW_PUBLISH_JOB="$(sbatch --parsable \
    --partition="$PARTITION" \
    --chdir="$AGRIBRAIN_SOURCE_SNAPSHOT" \
    --dependency="afterok:${PREVIOUS_JOB}" \
    --export=ALL \
    --output="${SENSITIVITY_RUN_DIR}/logs/publish_%j.out" \
    --error="${SENSITIVITY_RUN_DIR}/logs/publish_%j.err" \
    hpc/hpc_sensitivity_publish.sh)"
PUBLISH_JOB="${RAW_PUBLISH_JOB%%;*}"
case "$PUBLISH_JOB" in *[!0-9]*|'') echo "BLOCK: sbatch returned invalid publisher id: ${RAW_PUBLISH_JOB}"; exit 1;; esac

SUBMISSION_RECEIPT="${SENSITIVITY_RUN_DIR}/slurm_submission.json"
python - "$SUBMISSION_RECEIPT" "$RUN_TAG" "$GIT_COMMIT" "$TASK_COUNT" \
    "$CHUNK_SIZE" "$MAX_CONCURRENT" "$PUBLISH_JOB" "$PREVIOUS_JOB" \
    "${TASK_JOB_IDS[*]}" "${TASK_OFFSETS[*]}" "${TASK_COUNTS[*]}" \
    "${TASK_DEPENDENCIES[*]}" "$AGRIBRAIN_SOURCE_SNAPSHOT_MODE" \
    "$AGRIBRAIN_SOURCE_TREE_SHA256" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
job_ids = sys.argv[9].split()
offsets = [int(value) for value in sys.argv[10].split()]
counts = [int(value) for value in sys.argv[11].split()]
dependencies = sys.argv[12].split()
if not (len(job_ids) == len(offsets) == len(counts) == len(dependencies)):
    raise SystemExit("BLOCK: inconsistent Slurm submission receipt arrays")
if sys.argv[13] != "detached_readonly_git_worktree_v1":
    raise SystemExit("BLOCK: invalid structural source snapshot mode")
if re.fullmatch(r"[0-9a-f]{64}", sys.argv[14]) is None:
    raise SystemExit("BLOCK: invalid structural source-tree SHA-256")
payload = {
    "schema_version": 2,
    "analysis_label": "structural sensitivity",
    "receipt_scope": "submission_only_not_scheduler_completion",
    "scheduler_completion_attested": False,
    "run_tag": sys.argv[2],
    "source_commit": sys.argv[3],
    "source_snapshot_mode": sys.argv[13],
    "source_tree_sha256": sys.argv[14],
    "task_count": int(sys.argv[4]),
    "array_chunk_size_limit": int(sys.argv[5]),
    "max_concurrent_per_array": int(sys.argv[6]),
    "task_arrays": [
        {
            "job_id": job_id,
            "offset": offset,
            "count": count,
            "afterok_job_id": None if dependency == "none" else dependency,
        }
        for job_id, offset, count, dependency in zip(
            job_ids, offsets, counts, dependencies, strict=True
        )
    ],
    "publisher": {
        "job_id": sys.argv[7],
        "afterok_job_id": sys.argv[8],
    },
}
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
).hexdigest()
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, allow_nan=False)
    handle.write("\n")
PY

echo "Structural RUN_TAG: ${RUN_TAG}"
echo "External run dir:  ${SENSITIVITY_RUN_DIR}"
echo "Task jobs:        ${TASK_JOB_IDS[*]}"
echo "Publisher job:    ${PUBLISH_JOB} (afterok ${PREVIOUS_JOB})"
echo "No structural result is written to mvp/simulation/results."
