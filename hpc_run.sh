#!/bin/bash
# AgriBrain HPC orchestrator. Runs on the login node, prepares the venv,
# then submits the seed array and the dependent aggregation job.
#
# The benchmark runs as:
#   hpc_seed.sh (20-task array, one seed per task, parallel)
#     -> hpc_aggregate.sh (single task, runs after all array tasks succeed)
#
# Usage:
#   AGRIBRAIN_PARTITION=compute bash hpc_run.sh
#
# The partition name is required because SLURM installs without a system
# default partition (e.g. SDSMT) would otherwise fail the sbatch submit
# with "No partition specified or system default partition". Check the
# cluster's available partitions with ``sinfo -s``.
#
# Outputs land under mvp/simulation/results/ at the default locations and
# are also archived to hpc_results_<RUN_TAG>.tar.gz. The per-seed JSONs
# are written to mvp/simulation/results/benchmark_seeds/<RUN_TAG>/.
set -euo pipefail

echo "=== AgriBrain HPC orchestrator ==="
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Commit: $(git rev-parse HEAD)"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "BLOCK: sbatch not available. This script expects a SLURM login node."
    exit 1
fi

# Partition selection. AGRIBRAIN_PARTITION wins, then the stock SLURM env
# variable SBATCH_PARTITION, then abort loudly. Never silently pick a
# default; some clusters do not have one and silent failure costs queue
# time.
PARTITION="${AGRIBRAIN_PARTITION:-${SBATCH_PARTITION:-}}"
if [ -z "$PARTITION" ]; then
    echo "BLOCK: no SLURM partition selected."
    echo "       Set AGRIBRAIN_PARTITION (or SBATCH_PARTITION) before re-running, e.g.:"
    echo "           AGRIBRAIN_PARTITION=compute bash hpc_run.sh"
    echo "       Inspect the cluster's partitions with: sinfo -s"
    exit 1
fi
echo "Partition: ${PARTITION}"

# This orchestrator only ships stochastic benchmark numbers. Refuse to launch
# if the env requests deterministic mode so the seed array cannot quietly
# produce identical-per-seed results.
if [ "${DETERMINISTIC_MODE:-false}" = "true" ]; then
    echo "BLOCK: DETERMINISTIC_MODE=true is set in the environment."
    echo "       This script is for stochastic seed runs. Unset it (or run a"
    echo "       deterministic driver instead) and re-submit."
    exit 1
fi
# Make the choice explicit and inheritable by every sbatch task.
export DETERMINISTIC_MODE=false

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
assert 'demand_query' in names, 'BLOCK: demand_query missing'
from src.models.action_selection import THETA
assert THETA_CONTEXT.shape == (3, 5), 'BLOCK: THETA_CONTEXT shape not (3,5)'
assert THETA.shape == (3, 10), 'BLOCK: THETA shape not (3,10)'
print('Path B loaded')
"

# Compute a run tag that uniquely identifies this submission.
RUN_TAG="$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M)"
echo ""
echo "RUN_TAG=${RUN_TAG}"

mkdir -p logs

# Submit the 20-task seed array. DETERMINISTIC_MODE is pinned so a stale
# value in the cluster env cannot turn the array into identical-per-seed runs.
# --partition is passed explicitly so clusters without a system default
# (e.g. SDSMT) do not reject the submit.
SEED_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --export=ALL,RUN_TAG="$RUN_TAG",DETERMINISTIC_MODE=false hpc_seed.sh)
echo "Submitted seed array as job ${SEED_JOB}"

# Submit the aggregation job with a dependency on the array completing OK.
AGG_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --export=ALL,RUN_TAG="$RUN_TAG",DETERMINISTIC_MODE=false \
    --dependency=afterok:${SEED_JOB} hpc_aggregate.sh)
echo "Submitted aggregation as job ${AGG_JOB} (depends on ${SEED_JOB})"

echo ""
echo "Queue:"
squeue -u "$USER"

echo ""
echo "Transfer archive after completion:"
echo "  scp <hpc-host>:\$PWD/hpc_results_${RUN_TAG}.tar.gz ."
