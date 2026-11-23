#!/bin/bash
# Prepare the study and submit it as one dependency-linked chain:
# pilot cells and their independent repeat, the pilot check, the study in
# blocks of 500 tasks, the frozen-policy controls, and the collection.
# Pass only account/partition/resource options; arrays and dependencies are
# managed here. Submit once: a second submission starts a second study.
# Usage: bash study/submit.sh [sbatch options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export AGRI_STUDY_REPO="$(pwd -P)"
source study/setup.env
source hpc/ensure_git_available.sh
source "$AGRI_STUDY_VENV/bin/activate"
source hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
export AGRI_STUDY_CONCURRENCY="${AGRI_STUDY_CONCURRENCY:-20}"
[[ "$AGRI_STUDY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency' >&2; exit 2; }
export AGRI_STUDY_OUTPUT="${AGRI_STUDY_OUTPUT:-$(dirname "$AGRI_STUDY_REPO")/study_$(date -u +%Y%m%d_%H%M%S)_results}"
python study/study.py prepare --output "$AGRI_STUDY_OUTPUT"
python study/study.py prepare --output "$AGRI_STUDY_OUTPUT/reproducibility_repeat"
mkdir -p "$AGRI_STUDY_OUTPUT/logs"
git bundle create "$AGRI_STUDY_OUTPUT/source.bundle" HEAD
for var in AGRI_STUDY_REPO AGRI_STUDY_VENV AGRI_STUDY_OUTPUT AGRI_STUDY_CONCURRENCY; do
    printf 'export %s=%q\n' "$var" "${!var}"
done > "$AGRI_STUDY_OUTPUT/run.env"
LOGS="$AGRI_STUDY_OUTPUT/logs"
IDS="$AGRI_STUDY_OUTPUT/job_ids.txt"
TASKS=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["counts"]["tasks"])' "$AGRI_STUDY_OUTPUT/plan.json")
PILOTS=$(wc -l < "$AGRI_STUDY_OUTPUT/pilot_indices.txt")
REPEATS=$(wc -l < "$AGRI_STUDY_OUTPUT/repeat_indices.txt")
CONTROLS=$(python study/controls.py missing --output "$AGRI_STUDY_OUTPUT" | tr ',' '\n' | grep -c .)

submit() {  # name, then sbatch arguments; prints the job id
    local name=$1 job
    shift
    job=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes "$@") || job=''
    job=${job%%;*}
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        echo "Submission of $name failed. Cancel the jobs listed in $IDS before submitting again." >&2
        return 1
    fi
    printf '%s=%s\n' "$name" "$job" >> "$IDS"
    echo "$job"
}

PILOT=$(submit PILOT "$@" --array="0-$((PILOTS - 1))" \
    --output="$LOGS/pilot_%A_%a.out" --error="$LOGS/pilot_%A_%a.err" study/worker.sh pilot)
REPEAT=$(submit REPEAT "$@" --array="0-$((REPEATS - 1))" \
    --output="$LOGS/repeat_%A_%a.out" --error="$LOGS/repeat_%A_%a.err" study/worker.sh repeat)
PREVIOUS=$(submit CHECK "$@" --dependency="afterok:$PILOT:$REPEAT" \
    --output="$LOGS/check_%j.out" --error="$LOGS/check_%j.err" study/worker.sh check)
# Chained blocks bound the total concurrency and stay below common array limits.
for ((OFFSET = 0; OFFSET < TASKS; OFFSET += 500)); do
    LAST=$((TASKS - OFFSET - 1))
    if ((LAST > 499)); then LAST=499; fi
    PREVIOUS=$(submit "BLOCK_$OFFSET" "$@" --dependency="afterok:$PREVIOUS" \
        --array="0-$LAST%$AGRI_STUDY_CONCURRENCY" \
        --output="$LOGS/study_${OFFSET}_%A_%a.out" --error="$LOGS/study_${OFFSET}_%A_%a.err" \
        study/worker.sh study "$OFFSET")
done
PREVIOUS=$(submit CONTROLS "$@" --dependency="afterok:$PREVIOUS" \
    --array="0-$((CONTROLS - 1))%$AGRI_STUDY_CONCURRENCY" \
    --output="$LOGS/controls_%A_%a.out" --error="$LOGS/controls_%A_%a.err" study/worker.sh controls)
submit COLLECT "$@" --dependency="afterok:$PREVIOUS" \
    --output="$LOGS/collect_%j.out" --error="$LOGS/collect_%j.err" study/collect.sh > /dev/null
cat "$IDS"
echo "RESULTS=$AGRI_STUDY_OUTPUT"
echo "$TASKS tasks and $CONTROLS control cells. Everything waits for the checked pilot."
echo 'The analysis is complete when COMPLETE.json exists in the results folder.'
echo "The collection is complete when $(basename "$AGRI_STUDY_OUTPUT")_COLLECTED.sha256 exists beside it, not when the jobs end."
