#!/bin/bash
# Run after earlier study jobs are terminal. Verification may take minutes: use a compute allocation.
set -euo pipefail
: "${AGRI_WEIGHT_OUTPUT:?Source the run.env from the intended results folder}"
cd "$AGRI_WEIGHT_PACKAGE"
source "$AGRI_WEIGHT_VENV/bin/activate"
source source/hpc/publication_env.sh
export PYTHONDONTWRITEBYTECODE=1
python study.py missing --output "$AGRI_WEIGHT_OUTPUT" > "$AGRI_WEIGHT_OUTPUT/missing.txt"
if [ ! -s "$AGRI_WEIGHT_OUTPUT/missing.txt" ] || [ -z "$(tr -d '\n\r ' < "$AGRI_WEIGHT_OUTPUT/missing.txt")" ]; then
    echo 'No missing or invalid tasks. Submit collect.sh with --export=ALL.'; exit
fi
echo 'Missing/invalid task IDs are in missing.txt.'
echo 'Incomplete attempts are retained; a task with a corrupt completed result fails for inspection.'
python - "$AGRI_WEIGHT_OUTPUT/missing.txt" <<'PY' > "$AGRI_WEIGHT_OUTPUT/retry_arrays.txt"
import sys
ids=[int(x) for x in open(sys.argv[1]).read().strip().split(',') if x]
for offset in range(0,2300,500):
    local=[str(i-offset) for i in ids if offset<=i<offset+500]
    if local: print(offset,' '.join([','.join(local)]))
PY
PREVIOUS=''
while read -r OFFSET ARRAY; do
    DEPS=(); if [ -n "$PREVIOUS" ]; then DEPS=(--dependency="afterok:$PREVIOUS" --kill-on-invalid-dep=yes); fi
    JOB=$(sbatch --parsable "$@" "${DEPS[@]}" --export=ALL --array="$ARRAY%${AGRI_WEIGHT_CONCURRENCY:-10}" --output="$AGRI_WEIGHT_OUTPUT/logs/retry_%A_%a.out" --error="$AGRI_WEIGHT_OUTPUT/logs/retry_%A_%a.err" worker.sh study "$OFFSET")
    PREVIOUS=${JOB%%;*}; echo "RETRY=$PREVIOUS"
done < "$AGRI_WEIGHT_OUTPUT/retry_arrays.txt"
sbatch "$@" --dependency="afterok:$PREVIOUS" --kill-on-invalid-dep=yes --export=ALL --output="$AGRI_WEIGHT_OUTPUT/logs/recollect_%j.out" --error="$AGRI_WEIGHT_OUTPUT/logs/recollect_%j.err" collect.sh
