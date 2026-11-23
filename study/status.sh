#!/bin/bash
# Report how far a study has come, from its results folder alone, and name the
# first stage that is not complete with the logs that belong to it. It only
# reads: nothing is submitted, verified or changed. A result counted here has
# been accepted by its task; study/retry.sh verifies every one again.
# Usage: bash study/status.sh <results folder>
set -euo pipefail
OUT=$(cd "${1:?Usage: bash study/status.sh <results folder>}" && pwd -P)
NAME=$(basename "$OUT")
RECORD="$(dirname "$OUT")/${NAME}_COLLECTED.sha256"

listed() {  # how many tasks of the list $1 have a result in the folder $2
    local index name count=0
    while read -r index; do
        printf -v name '%04d' "$index"
        if [ -e "$2/tasks/$name/result.json" ]; then count=$((count + 1)); fi
    done < "$1"
    echo "$count"
}
results() {  # how many results the folder $1 holds
    if [ -d "$1" ]; then find "$1" -mindepth 2 -maxdepth 2 -name result.json | wc -l; else echo 0; fi
}
present() { if [ -e "$1" ]; then echo yes; else echo no; fi; }

PLANNED=$(grep -o '"tasks": *[0-9]*' "$OUT/plan.json" | grep -o '[0-9]*$')
PILOTS=$(wc -l < "$OUT/pilot_indices.txt")
REPEATS=$(wc -l < "$OUT/repeat_indices.txt")
PILOT_DONE=$(listed "$OUT/pilot_indices.txt" "$OUT")
REPEAT_DONE=$(listed "$OUT/repeat_indices.txt" "$OUT/reproducibility_repeat")
TASKS_DONE=$(results "$OUT/tasks")
PASSED=$(present "$OUT/PILOT_PASSED.json")
ANALYSED=$(present "$OUT/COMPLETE.json")
COLLECTED=$(present "$RECORD")

echo "Results folder: $OUT"
printf '%-30s %s of %s\n' 'pilot cells with a result' "$PILOT_DONE" "$PILOTS"
printf '%-30s %s of %s\n' 'repeats with a result' "$REPEAT_DONE" "$REPEATS"
printf '%-30s %s\n' 'pilot check passed' "$PASSED"
printf '%-30s %s of %s\n' 'tasks with a result' "$TASKS_DONE" "$PLANNED"
printf '%-30s %s\n' 'control cells with a result' "$(results "$OUT/controls")"
printf '%-30s %s\n' 'analysis complete' "$ANALYSED"
printf '%-30s %s\n' 'collection complete' "$COLLECTED"
if QUEUED=$(squeue --noheader --user="$(id -un)" --format='%F' 2> /dev/null); then
    ACTIVE=$(sort -u <<< "$QUEUED" | grep -Fx -f <(cut -d= -f2 "$OUT/job_ids.txt" 2> /dev/null) \
        | paste -sd' ' -) || true
    printf '%-30s %s\n' 'jobs still in the queue' "${ACTIVE:-none}"
fi

# Logs of a resumption carry the prefix retry_; the collection of one is recollect_.
if [ "$COLLECTED" = yes ]; then
    echo "Complete. Download ${NAME}_analysis.tar.gz and its .sha256 from $(dirname "$OUT")."
elif [ "$PILOT_DONE" -lt "$PILOTS" ] || [ "$REPEAT_DONE" -lt "$REPEATS" ]; then
    echo 'Not complete: the pilot cells or their repeat. Logs: logs/*pilot_*.err, logs/*repeat_*.err'
elif [ "$PASSED" = no ]; then
    echo 'Not complete: the pilot check. Logs: logs/*check_*.err'
elif [ "$TASKS_DONE" -lt "$PLANNED" ]; then
    echo 'Not complete: the tasks. Logs: logs/study_*.err, logs/retry_[0-9]*.err'
elif [ "$ANALYSED" = no ]; then
    echo 'Not complete: the controls or the analysis. Logs: logs/*controls_*.err, logs/*collect_*.err'
else
    echo 'Not complete: the archives. Logs: logs/*collect_*.err'
fi
