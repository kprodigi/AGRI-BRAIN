#!/bin/bash
# Prepare the canonical runtime and submit exactly the two learned modes.
# Usage: bash hpc/submit_context_pair.sh [sbatch account/partition options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export AGRIBRAIN_SOURCE_SNAPSHOT="$(pwd -P)"
source hpc/ensure_git_available.sh
export AGRIBRAIN_GIT_COMMIT="$(git rev-parse HEAD)"
export RUN_TAG="context_pair_$(date -u +%Y%m%d_%H%M%S)_${AGRIBRAIN_GIT_COMMIT:0:7}"
export AGRIBRAIN_VENV=".publication_venvs/${RUN_TAG}"
export NO_CONTEXT_OUTPUT_ROOT="$(dirname "$AGRIBRAIN_SOURCE_SNAPSHOT")/${RUN_TAG}_results"
export NO_CONTEXT_INCLUDE_COMPARATORS=learned_pair
source hpc/publication_env.sh

PAIR_PYTHON="${AGRIBRAIN_PYTHON_BIN:-python3.11}"
if ! command -v "$PAIR_PYTHON" >/dev/null 2>&1; then
    echo "Load your cluster's Python 3.11 module, or set AGRIBRAIN_PYTHON_BIN to its executable." >&2
    exit 2
fi
"$PAIR_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 is required"'
"$PAIR_PYTHON" hpc/validate_source_checkout.py
if [ -e "$NO_CONTEXT_OUTPUT_ROOT" ] || [ -e "$AGRIBRAIN_VENV" ]; then
    echo "Run location already exists; rerun this command with a fresh timestamp." >&2
    exit 2
fi
"$PAIR_PYTHON" -m venv "$AGRIBRAIN_VENV"
source "$AGRIBRAIN_VENV/bin/activate"
python -m pip install -r agribrain/backend/requirements-lock.txt
mkdir "$AGRIBRAIN_VENV/backend-build-source"
cp -a agribrain/backend/. "$AGRIBRAIN_VENV/backend-build-source/"
python -m pip install "$AGRIBRAIN_VENV/backend-build-source" --no-deps
python -m pip check
python -m mvp.simulation.benchmarks.rerun_no_context 42 \
    --output-root "$NO_CONTEXT_OUTPUT_ROOT" --include-agribrain --check-only

mkdir -p "$NO_CONTEXT_OUTPUT_ROOT/logs"
git bundle create "$NO_CONTEXT_OUTPUT_ROOT/source.bundle" HEAD
# Only non-secret run coordinates are written; this file can be sourced later
# to retry selected array indices using exactly the original environment.
for variable in AGRIBRAIN_SOURCE_SNAPSHOT AGRIBRAIN_GIT_COMMIT RUN_TAG AGRIBRAIN_VENV \
    NO_CONTEXT_OUTPUT_ROOT NO_CONTEXT_INCLUDE_COMPARATORS; do
    printf 'export %s=%q\n' "$variable" "${!variable}"
done > "$NO_CONTEXT_OUTPUT_ROOT/run.env"
JOB_ID=$(sbatch --parsable --export=ALL \
    --output="$NO_CONTEXT_OUTPUT_ROOT/logs/seed_%A_%a.out" \
    --error="$NO_CONTEXT_OUTPUT_ROOT/logs/seed_%A_%a.err" \
    "$@" hpc/no_context_rerun.sh)
printf '%s\n' "$JOB_ID" > "$NO_CONTEXT_OUTPUT_ROOT/slurm_job_id.txt"
echo "Submitted: $JOB_ID"
echo "Modes: revised No-context and standard-RAG AGRI-BRAIN (800 episodes)"
echo "Results: $NO_CONTEXT_OUTPUT_ROOT"
printf 'After completion, run: bash hpc/collect_context_pair.sh %q\n' "$NO_CONTEXT_OUTPUT_ROOT"
