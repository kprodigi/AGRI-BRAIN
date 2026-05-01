#!/bin/bash
# SLURM job-array task: run one seed of the AgriBrain benchmark.
# Submitted by hpc/hpc_run.sh; expects RUN_TAG to be exported by sbatch.
#
# Output: mvp/simulation/results/benchmark_seeds/${RUN_TAG}/seed_${SEED}.json
# One file per array task, isolated per run by the hash-tagged subdirectory.
#SBATCH --job-name=agribrain-seed
#SBATCH --array=0-19
# 18h wall-time per seed task. Realistic runtime is 2-4h on a compute node,
# so 18h is 4-9x headroom. SDSMT's compute partition allows up to 14 days,
# so this is still well under the cluster cap. Higher limits bias the
# scheduler toward longer queue waits on some sites; on SDSMT compute the
# node count (41) makes that impact negligible.
#SBATCH --time=18:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/seed_%A_%a.out
#SBATCH --error=logs/seed_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Activate the venv prepared by hpc/hpc_run.sh on the login node.
if [ ! -f .venv/bin/activate ]; then
    echo "BLOCK: .venv not found. Run hpc/hpc_run.sh (or set up .venv manually) first."
    exit 1
fi
source .venv/bin/activate

if [ -z "${RUN_TAG:-}" ]; then
    echo "BLOCK: RUN_TAG not exported. Submit via hpc/hpc_run.sh."
    exit 1
fi

# Belt-and-suspenders: even if --export skipped DETERMINISTIC_MODE, force
# stochastic. Aborts visibly if something upstream tried to set it true.
if [ "${DETERMINISTIC_MODE:-false}" = "true" ]; then
    echo "BLOCK: DETERMINISTIC_MODE=true reached the seed task. Stochastic seeds expected."
    exit 1
fi
export DETERMINISTIC_MODE=false

# Enable the three anomaly-defense flags so fig 4 panel C
# (Cumulative Anomaly Defenses Triggered) actually carries non-zero
# defense events for AgriBrain. The coordinator gates each defense
# on a separate policy_flags entry:
#   FAILURE_INJECTION       -> coordinator injects MCP-tool faults
#                              every 11h and the fault_recovery
#                              trace records each one. without this,
#                              fault_recovery_trace is all zeros for
#                              every mode and panel C only shows
#                              cooperative_veto events (firing
#                              narrowly inside the 12-30h window).
#   PHYSICS_CONSISTENCY_GATE -> compute_context_modifier zeros the
#                              modifier when the retrieved-context
#                              physics_consistency_score < 0.03 and
#                              the physics_gate trace records each
#                              firing. without this, the gate is
#                              silent and physics_gate_trace stays
#                              at zero.
# These two are propagated into Policy.policy_flags by
# generate_results.run_episode (see env-var read at the top of the
# function); the coordinator reads them via obs.raw["policy_flags"]
# during _compute_step_context.
export FAILURE_INJECTION=true
export PHYSICS_CONSISTENCY_GATE=true

# Map array index to the canonical 20-seed list.
SEEDS=(42 1337 2024 7 99 101 202 303 404 505 \
       606 707 808 909 1010 1111 1212 1313 1414 1515)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

OUT_DIR="mvp/simulation/results/benchmark_seeds/${RUN_TAG}"
mkdir -p "$OUT_DIR"

echo "[seed=${SEED} tag=${RUN_TAG}] starting at $(date), output -> ${OUT_DIR}/seed_${SEED}.json"

# Pre-flight invariants check. Costs <1 s and avoids 2-6 h of wasted
# compute producing numbers from a stale code path.
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

python mvp/simulation/benchmarks/run_single_seed.py "$SEED" --output-dir "$OUT_DIR"

echo "[seed=${SEED}] complete at $(date)"
