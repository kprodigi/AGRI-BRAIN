#!/bin/bash
# Submit from the clean checkout as described in docs/NO_CONTEXT_RERUN.md.
#SBATCH --job-name=agribrain-noctx
#SBATCH --array=0-19
#SBATCH --time=18:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail
: "${AGRIBRAIN_SOURCE_SNAPSHOT:?Set the absolute clean checkout path}"
: "${AGRIBRAIN_VENV:?Set the run-scoped Python 3.11 venv}"
: "${AGRIBRAIN_GIT_COMMIT:?Set the full committed source SHA}"
: "${RUN_TAG:?Set a new run tag}"
: "${NO_CONTEXT_OUTPUT_ROOT:?Set an output directory outside the checkout}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}"
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
source hpc/ensure_git_available.sh
source "$AGRIBRAIN_VENV/bin/activate"
source hpc/publication_env.sh

SEEDS=(42 1337 2024 7 99 101 202 303 404 505 606 707 808 909 1010 1111 1212 1313 1414 1515)
if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || (( SLURM_ARRAY_TASK_ID > 19 )); then
    echo "Invalid seed-array index" >&2
    exit 2
fi
ARGS=()
case "${NO_CONTEXT_INCLUDE_COMPARATORS:-0}" in
    0) ;;
    1) ARGS+=(--include-comparators) ;;
    learned_pair) ARGS+=(--include-agribrain) ;;
    *) echo "NO_CONTEXT_INCLUDE_COMPARATORS must be 0, 1, or learned_pair" >&2; exit 2 ;;
esac
python -m mvp.simulation.benchmarks.rerun_no_context \
    "${SEEDS[$SLURM_ARRAY_TASK_ID]}" \
    --output-root "$NO_CONTEXT_OUTPUT_ROOT" "${ARGS[@]}"
