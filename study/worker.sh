#!/bin/bash
#SBATCH --job-name=agri-study
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
: "${AGRI_STUDY_REPO:?Missing repository path}"
: "${AGRI_STUDY_OUTPUT:?Missing output path}"
: "${AGRI_STUDY_VENV:?Missing environment}"
cd "$AGRI_STUDY_REPO"
source hpc/ensure_git_available.sh
source "$AGRI_STUDY_VENV/bin/activate"
source hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
KIND="${1:?Expected pilot, repeat, check, study or controls}"
PASSED="$AGRI_STUDY_OUTPUT/PILOT_PASSED.json"
run() {  # one more attempt after a transient failure; a repeatable failure still fails the job
    "$@" && return 0
    echo 'First attempt failed; trying once more.' >&2
    sleep 60
    "$@"
}
released() {  # only a passed pilot check of this plan releases what lies outside the pilot
    python -c 'import hashlib, json, sys
plan = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
passed = json.load(open(sys.argv[2], encoding="utf-8"))
sys.exit(0 if passed.get("plan_sha256") == plan else 1)' \
        "$AGRI_STUDY_OUTPUT/plan.json" "$PASSED" 2> /dev/null || {
        echo 'The pilot check has not passed for this plan: nothing outside the pilot runs.' >&2
        exit 3
    }
}
if [ "$KIND" = check ]; then
    # An earlier pass does not stand in for this check.
    rm -f "$PASSED"
    python study/study.py pilot-check --output "$AGRI_STUDY_OUTPUT"
    exit
fi
: "${SLURM_ARRAY_TASK_ID:?Must run through a Slurm array}"
listed() {  # the array task's entry of a one-index-per-line file
    local index
    index=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
    [[ "$index" =~ ^[0-9]+$ ]] || { echo "No entry $SLURM_ARRAY_TASK_ID in $1" >&2; exit 2; }
    echo "$index"
}
case "$KIND" in
    pilot)
        run python study/study.py task --output "$AGRI_STUDY_OUTPUT" \
            --index "$(listed "$AGRI_STUDY_OUTPUT/pilot_indices.txt")" ;;
    repeat)
        run python study/study.py task --output "$AGRI_STUDY_OUTPUT/reproducibility_repeat" \
            --index "$(listed "$AGRI_STUDY_OUTPUT/repeat_indices.txt")" ;;
    study)
        released
        run python study/study.py task --output "$AGRI_STUDY_OUTPUT" \
            --index $((SLURM_ARRAY_TASK_ID + ${2:?Missing offset})) ;;
    controls)
        released
        # The primary settings come first in the design: their tasks are 0, 1, ...
        run python study/controls.py run --output "$AGRI_STUDY_OUTPUT" --index "$SLURM_ARRAY_TASK_ID" ;;
    *)
        echo 'Unknown worker type' >&2
        exit 2 ;;
esac
