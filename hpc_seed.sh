#!/bin/bash
# SLURM job-array task: run one seed of the AgriBrain benchmark.
# Submitted by hpc_run.sh; expects RUN_TAG to be exported by sbatch.
#
# Output: mvp/simulation/results/benchmark_seeds/${RUN_TAG}/seed_${SEED}.json
# One file per array task, isolated per run by the hash-tagged subdirectory.
#SBATCH --job-name=agribrain-seed
#SBATCH --array=0-19
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/seed_%A_%a.out
#SBATCH --error=logs/seed_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Activate the venv prepared by hpc_run.sh on the login node.
if [ ! -f .venv/bin/activate ]; then
    echo "BLOCK: .venv not found. Run hpc_run.sh (or set up .venv manually) first."
    exit 1
fi
source .venv/bin/activate

if [ -z "${RUN_TAG:-}" ]; then
    echo "BLOCK: RUN_TAG not exported. Submit via hpc_run.sh."
    exit 1
fi

# Map array index to the canonical 20-seed list.
SEEDS=(42 1337 2024 7 99 101 202 303 404 505 \
       606 707 808 909 1010 1111 1212 1313 1414 1515)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

OUT_DIR="mvp/simulation/results/benchmark_seeds/${RUN_TAG}"
mkdir -p "$OUT_DIR"

echo "[seed=${SEED} tag=${RUN_TAG}] starting at $(date), output -> ${OUT_DIR}/seed_${SEED}.json"

# Fail fast if Path B is not loaded in this environment. Costs <1 s and
# avoids 2-6 h of wasted compute producing numbers from an old code path.
python -c "
import sys
sys.path.insert(0, 'agri-brain-mvp-1.0.0/backend')
from pirag.mcp.registry import get_default_registry
from pirag.context_to_logits import THETA_CONTEXT
names = {t['name'] for t in get_default_registry().list_tools()}
assert 'yield_query' in names, 'BLOCK: yield_query missing'
assert THETA_CONTEXT.shape == (3, 6), 'BLOCK: THETA_CONTEXT shape not (3,6)'
print('Path B loaded')
"

python mvp/simulation/benchmarks/run_single_seed.py "$SEED" --output-dir "$OUT_DIR"

echo "[seed=${SEED}] complete at $(date)"
