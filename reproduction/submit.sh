#!/bin/bash
# Pass only real account/partition/resource options; job arrays/dependencies are managed here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export AGRI_WEIGHT_PACKAGE="$(pwd -P)"
source setup.env
source "$AGRI_WEIGHT_VENV/bin/activate"
source source/hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
export AGRI_WEIGHT_CONCURRENCY="${AGRI_WEIGHT_CONCURRENCY:-10}"
[[ "$AGRI_WEIGHT_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency' >&2; exit 2; }
export AGRI_WEIGHT_OUTPUT="${AGRI_WEIGHT_OUTPUT:-$(dirname "$AGRI_WEIGHT_PACKAGE")/weights_$(date -u +%Y%m%d_%H%M%S)_results}"
python study.py prepare --output "$AGRI_WEIGHT_OUTPUT"
mkdir -p "$AGRI_WEIGHT_OUTPUT/logs"
for var in AGRI_WEIGHT_PACKAGE AGRI_WEIGHT_VENV AGRI_WEIGHT_OUTPUT AGRI_WEIGHT_CONCURRENCY; do
    printf 'export %s=%q\n' "$var" "${!var}"
done > "$AGRI_WEIGHT_OUTPUT/run.env"
PILOT=$(sbatch --parsable "$@" --export=ALL --array=0-8%3 --output="$AGRI_WEIGHT_OUTPUT/logs/pilot_%A_%a.out" --error="$AGRI_WEIGHT_OUTPUT/logs/pilot_%A_%a.err" worker.sh pilot)
PILOT=${PILOT%%;*}; printf 'PILOT=%s\n' "$PILOT" >> "$AGRI_WEIGHT_OUTPUT/job_ids.txt"
CHECK=$(sbatch --parsable "$@" --dependency="afterok:$PILOT" --kill-on-invalid-dep=yes --export=ALL --output="$AGRI_WEIGHT_OUTPUT/logs/check_%j.out" --error="$AGRI_WEIGHT_OUTPUT/logs/check_%j.err" worker.sh check)
CHECK=${CHECK%%;*}; printf 'CHECK=%s\n' "$CHECK" >> "$AGRI_WEIGHT_OUTPUT/job_ids.txt"
PREVIOUS=$CHECK
# Chained blocks keep total concurrency bounded and avoid typical 1001-task array limits.
for OFFSET in 0 500 1000 1500 2000; do
    LAST=499; if [ "$OFFSET" = 2000 ]; then LAST=299; fi
    JOB=$(sbatch --parsable "$@" --dependency="afterok:$PREVIOUS" --kill-on-invalid-dep=yes --export=ALL --array="0-$LAST%$AGRI_WEIGHT_CONCURRENCY" --output="$AGRI_WEIGHT_OUTPUT/logs/study_${OFFSET}_%A_%a.out" --error="$AGRI_WEIGHT_OUTPUT/logs/study_${OFFSET}_%A_%a.err" worker.sh study "$OFFSET")
    JOB=${JOB%%;*}; printf 'BLOCK_%s=%s\n' "$OFFSET" "$JOB" >> "$AGRI_WEIGHT_OUTPUT/job_ids.txt"
    PREVIOUS=$JOB
done
COLLECT=$(sbatch --parsable "$@" --dependency="afterok:$PREVIOUS" --kill-on-invalid-dep=yes --export=ALL --output="$AGRI_WEIGHT_OUTPUT/logs/collect_%j.out" --error="$AGRI_WEIGHT_OUTPUT/logs/collect_%j.err" collect.sh)
printf 'COLLECT=%s\n' "${COLLECT%%;*}" >> "$AGRI_WEIGHT_OUTPUT/job_ids.txt"
cat "$AGRI_WEIGHT_OUTPUT/job_ids.txt"
echo "RESULTS=$AGRI_WEIGHT_OUTPUT"
echo '2300 tasks; 20700 study episodes. All bulk jobs wait for the checked pilot.'
