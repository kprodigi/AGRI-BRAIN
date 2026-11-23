#!/bin/bash
# Resume a study whose jobs have ended before the collection finished.
# The stages are recovered in their order: the missing pilot cells and their
# missing repeats, the pilot check, the other missing tasks in blocks of 500,
# the missing controls, the collection. The pilot check runs again in every
# resumption, and nothing outside the pilot is released unless it passes.
# Usage: source <results>/run.env; bash "$AGRI_STUDY_REPO/study/retry.sh" [sbatch options]
# Verification reads every task: run it from a compute allocation.
set -euo pipefail
: "${AGRI_STUDY_OUTPUT:?Source the run.env of the intended results folder}"
cd "$AGRI_STUDY_REPO"
source hpc/ensure_git_available.sh
source "$AGRI_STUDY_VENV/bin/activate"
source hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
OUT="$AGRI_STUDY_OUTPUT"
LOGS="$OUT/logs"
IDS="$OUT/job_ids.txt"
CONCURRENCY="${AGRI_STUDY_CONCURRENCY:-20}"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency' >&2; exit 2; }
touch "$IDS"

# Two jobs must not execute one task: the earlier jobs of this study must have ended.
# A queue that cannot be read proves nothing, so nothing is resumed without it.
if ! QUEUED=$(squeue --noheader --user="$(id -un)" --format='%F'); then
    echo 'The queue could not be read, so it cannot be shown that no job of this study is' >&2
    echo 'still running. Nothing was resumed; run this again where squeue works.' >&2
    exit 1
fi
ACTIVE=$(sort -u <<< "$QUEUED" | grep -Fx -f <(cut -d= -f2 "$IDS") | paste -sd' ' -) || true
if [ -n "$ACTIVE" ]; then
    echo "Jobs of this study are still queued or running: $ACTIVE" >&2
    echo 'Resume after they have ended or have been cancelled.' >&2
    exit 1
fi

if [ -e "$(dirname "$OUT")/$(basename "$OUT")_COLLECTED.sha256" ]; then
    echo 'The collection of this study is complete: there is nothing to resume.'
    exit 0
fi

# Incomplete attempts are kept; a task whose completed result is corrupt is listed for inspection.
python study/study.py missing --output "$OUT" > "$OUT/missing.txt"
python study/study.py missing --output "$OUT/reproducibility_repeat" > "$OUT/missing_repeat.txt"
# What runs again, by stage: the missing pilot cells and repeats as positions in their lists,
# and the other missing tasks as positions within blocks of 500.
python - "$OUT" > "$OUT/resumption.txt" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
sys.stdout.reconfigure(newline='\n')


def numbers(name):
    return [int(x) for x in (out / name).read_text(encoding='utf-8').replace(',', ' ').split()]


def ranges(values):
    """Increasing integers as an array specification: 1,3-5,9."""
    parts = []
    for value in values:
        if parts and value == parts[-1][1] + 1:
            parts[-1][1] = value
        else:
            parts.append([value, value])
    return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in parts)


pilot = numbers('pilot_indices.txt')
missing = set(numbers('missing.txt'))
for stage, cells, absent in [('PILOT', pilot, missing),
                             ('REPEAT', numbers('repeat_indices.txt'),
                              set(numbers('missing_repeat.txt')))]:
    positions = [position for position, index in enumerate(cells) if index in absent]
    if positions:
        print(stage, 0, ranges(positions))
rest = sorted(missing - set(pilot))
for offset in sorted({index // 500 * 500 for index in rest}):
    print('BLOCK', offset, ranges([i - offset for i in rest if offset <= i < offset + 500]))
PY

BEFORE=$(wc -l < "$IDS")
submit() {  # name, then sbatch arguments; prints the job id
    local name=$1 job
    shift
    job=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes "$@") || job=''
    job=${job%%;*}
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        echo "Submission of $name failed. Cancel the jobs listed in $IDS before resuming again." >&2
        return 1
    fi
    printf 'RETRY_%s=%s\n' "$name" "$job" >> "$IDS"
    echo "$job"
}

# 1. The pilot: its missing cells and repeats, then the check, whatever is missing.
GATE=''
for STAGE in PILOT REPEAT; do
    ARRAY=$(awk -v stage="$STAGE" '$1 == stage { print $3 }' "$OUT/resumption.txt")
    if [ -z "$ARRAY" ]; then continue; fi
    KIND=${STAGE,,}
    JOB=$(submit "$STAGE" "$@" --array="$ARRAY" \
        --output="$LOGS/retry_${KIND}_%A_%a.out" --error="$LOGS/retry_${KIND}_%A_%a.err" \
        study/worker.sh "$KIND")
    GATE="$GATE:$JOB"
done
DEPS=()
if [ -n "$GATE" ]; then DEPS=(--dependency="afterok$GATE"); fi
CHECK=$(submit CHECK "$@" ${DEPS[@]+"${DEPS[@]}"} \
    --output="$LOGS/retry_check_%j.out" --error="$LOGS/retry_check_%j.err" study/worker.sh check)

# 2. The other missing tasks. Every block waits for the passed check; a failed task does
#    not hold back the blocks after it.
PREVIOUS=$CHECK
while read -r -u 3 _ OFFSET ARRAY; do
    AFTER="afterok:$CHECK"
    if [ "$PREVIOUS" != "$CHECK" ]; then AFTER="$AFTER,afterany:$PREVIOUS"; fi
    PREVIOUS=$(submit "BLOCK_$OFFSET" "$@" --dependency="$AFTER" --array="$ARRAY%$CONCURRENCY" \
        --output="$LOGS/retry_${OFFSET}_%A_%a.out" --error="$LOGS/retry_${OFFSET}_%A_%a.err" \
        study/worker.sh study "$OFFSET")
done 3< <(awk '$1 == "BLOCK"' "$OUT/resumption.txt")

# 3. The missing controls, then the collection.
CONTROLS=$(python study/controls.py missing --output "$OUT")
if [ -n "$CONTROLS" ]; then
    AFTER="afterok:$CHECK"
    if [ "$PREVIOUS" != "$CHECK" ]; then AFTER="$AFTER:$PREVIOUS"; fi
    PREVIOUS=$(submit CONTROLS "$@" --dependency="$AFTER" --array="$CONTROLS%$CONCURRENCY" \
        --output="$LOGS/retry_controls_%A_%a.out" --error="$LOGS/retry_controls_%A_%a.err" \
        study/worker.sh controls)
fi
submit COLLECT "$@" --dependency="afterok:$PREVIOUS" \
    --output="$LOGS/recollect_%j.out" --error="$LOGS/recollect_%j.err" study/collect.sh > /dev/null
tail -n +"$((BEFORE + 1))" "$IDS"
echo 'Nothing outside the pilot runs unless the pilot check passes again.'
echo "The collection is complete when $(basename "$OUT")_COLLECTED.sha256 exists beside the results folder."
