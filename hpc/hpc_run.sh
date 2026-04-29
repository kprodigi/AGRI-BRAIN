#!/bin/bash
# AgriBrain HPC orchestrator. Runs on the login node, prepares the venv,
# then submits the seed array and the dependent aggregation job.
#
# The benchmark runs as:
#   hpc/hpc_seed.sh (20-task array, one seed per task, parallel)
#     -> hpc/hpc_aggregate.sh (single task, runs after all array tasks succeed)
#
# Usage (run from repo root):
#   AGRIBRAIN_PARTITION=compute bash hpc/hpc_run.sh
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
echo "Mode:   STOCHASTIC, 20-seed array (canonical published-results posture)"
echo "Note:   the deterministic regression-guard snapshot is regenerated"
echo "        out-of-band via:"
echo "          DETERMINISTIC_MODE=true REGRESSION_GUARD_INIT=true \\"
echo "              python -m mvp.simulation.validation.run_regression_guard"
echo "        It is intentionally NOT part of this orchestrator because the"
echo "        published numbers are stochastic; the snapshot exists only to"
echo "        catch later code drift on a deterministic re-run."

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
    echo "           AGRIBRAIN_PARTITION=compute bash hpc/hpc_run.sh"
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
pip install -e agribrain/backend --quiet
pip install pytest --quiet

# Pre-flight invariants check, before any SLURM time is consumed.
# Verifies the MCP tool registry contains the canonical forecast tools
# and that the policy / context matrices have the documented shapes.
# Fails fast on any mismatch so 6+ hours of seed-array compute are
# never spent on a broken codebase or a stale venv.
echo ""
echo "=== Pre-flight invariants check ==="
python -c "
import sys
sys.path.insert(0, 'agribrain/backend')
from pirag.mcp.registry import get_default_registry
from pirag.context_to_logits import THETA_CONTEXT
names = {t['name'] for t in get_default_registry().list_tools()}
assert 'yield_query' in names, 'BLOCK: yield_query missing from MCP registry'
assert 'demand_query' in names, 'BLOCK: demand_query missing from MCP registry'
from src.models.action_selection import THETA
assert THETA_CONTEXT.shape == (3, 5), 'BLOCK: THETA_CONTEXT shape not (3,5)'
assert THETA.shape == (3, 10), 'BLOCK: THETA shape not (3,10)'
print('Pre-flight invariants OK')
"

# Compute a run tag that uniquely identifies this submission.
RUN_TAG="$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M)"
# Capture the full source-code SHA so build_artifact_manifest.py can
# stamp it into artifact_manifest.json.git_commit. The aggregator job
# may run on a slurm worker where 'git rev-parse' fails (PATH issue,
# repo not visible from the worker, etc.); exporting AGRIBRAIN_GIT_COMMIT
# from the login node where git definitely works avoids the silent
# "unknown" fallback that broke the CI artifact-validation gate on the
# previous HPC run.
GIT_COMMIT="$(git rev-parse HEAD)"
echo ""
echo "RUN_TAG=${RUN_TAG}"
echo "GIT_COMMIT=${GIT_COMMIT}"

mkdir -p logs

# Submit the 20-task seed array. DETERMINISTIC_MODE is pinned so a stale
# value in the cluster env cannot turn the array into identical-per-seed runs.
# --partition is passed explicitly so clusters without a system default
# (e.g. SDSMT) do not reject the submit.
SEED_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --export=ALL,RUN_TAG="$RUN_TAG",DETERMINISTIC_MODE=false,AGRIBRAIN_GIT_COMMIT="$GIT_COMMIT" hpc/hpc_seed.sh)
echo "Submitted seed array as job ${SEED_JOB}"

# Submit the aggregation job with a dependency on the array completing OK.
AGG_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --export=ALL,RUN_TAG="$RUN_TAG",DETERMINISTIC_MODE=false,AGRIBRAIN_GIT_COMMIT="$GIT_COMMIT" \
    --dependency=afterok:${SEED_JOB} hpc/hpc_aggregate.sh)
echo "Submitted aggregation as job ${AGG_JOB} (depends on ${SEED_JOB})"

echo ""
echo "Queue:"
squeue -u "$USER"

echo ""
echo "Transfer archive after completion:"
echo "  scp <hpc-host>:\$PWD/hpc_results_${RUN_TAG}.tar.gz ."
