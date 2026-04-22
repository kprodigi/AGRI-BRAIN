#!/bin/bash
# AgriBrain HPC orchestrator. Runs on the login node, prepares the venv,
# then submits the seed array and the dependent aggregation job.
#
# The benchmark runs as:
#   hpc_seed.sh (20-task array, one seed per task, parallel)
#     -> hpc_aggregate.sh (single task, runs after all array tasks succeed)
#
# Usage:
#   bash hpc_run.sh
#
# Outputs land under mvp/simulation/results/ at the default locations and
# are also archived to hpc_results_<RUN_TAG>.tar.gz. The per-seed JSONs
# are written to mvp/simulation/results/benchmark_seeds/<RUN_TAG>/.
set -euo pipefail

echo "=== AgriBrain HPC orchestrator ==="
echo "Started: $(date)"
echo "Host: $(hostname)"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "BLOCK: sbatch not available. This script expects a SLURM login node."
    exit 1
fi

# One-time venv setup on the login node. The array and aggregate tasks
# source the same venv, so they see the same package versions.
if [ ! -d .venv ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -e agri-brain-mvp-1.0.0/backend --quiet
pip install pytest --quiet

# Fail fast if Path B is missing, before any SLURM time is consumed.
echo ""
echo "=== Path B load check ==="
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

# Compute a run tag that uniquely identifies this submission.
RUN_TAG="$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M)"
echo ""
echo "RUN_TAG=${RUN_TAG}"

mkdir -p logs

# Submit the 20-task seed array.
SEED_JOB=$(sbatch --parsable --export=ALL,RUN_TAG="$RUN_TAG" hpc_seed.sh)
echo "Submitted seed array as job ${SEED_JOB}"

# Submit the aggregation job with a dependency on the array completing OK.
AGG_JOB=$(sbatch --parsable --export=ALL,RUN_TAG="$RUN_TAG" \
    --dependency=afterok:${SEED_JOB} hpc_aggregate.sh)
echo "Submitted aggregation as job ${AGG_JOB} (depends on ${SEED_JOB})"

echo ""
echo "Queue:"
squeue -u "$USER"

echo ""
echo "Transfer archive after completion:"
echo "  scp <hpc-host>:\$PWD/hpc_results_${RUN_TAG}.tar.gz ."
