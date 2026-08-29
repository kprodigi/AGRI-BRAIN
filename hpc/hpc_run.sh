#!/bin/bash
# AgriBrain HPC orchestrator. Runs on the login node, prepares the venv,
# then submits the benchmark-seed array, the scenario-parallel stress array,
# and the dependent aggregation job.
#
# The benchmark runs as:
#   hpc/hpc_seed.sh (20-task array, one seed per task, parallel)
#     -> hpc/hpc_stress.sh (5-task array, one scenario per task, parallel;
#        reuses the completed primary nominal cells)
#       -> hpc/hpc_publish.sh (single strict aggregation/validation task)
#
# Usage (run from repo root):
#   AGRIBRAIN_PARTITION=compute bash hpc/hpc_run.sh
#
# The partition name is required because SLURM installs without a system
# default partition (common at many sites) would otherwise fail the sbatch submit
# with "No partition specified or system default partition". Check the
# cluster's available partitions with ``sinfo -s``.
#
# Outputs land under mvp/simulation/results/ at the locked locations.  The
# complete transfer unit is publication_bundle_<RUN_TAG>/, containing
# hpc_results_<RUN_TAG>.tar.gz, its external receipt, and READY.json.  The
# per-seed JSONs are written to
# mvp/simulation/results/benchmark_seeds/<RUN_TAG>/.
set -euo pipefail

# Establish the immutable source identity once, before any environment setup or
# job submission. Every worker executes from a unique detached worktree of this
# exact commit, never from the operator's live checkout.
if ! command -v git >/dev/null 2>&1; then
    echo "BLOCK: git not available. Publication jobs require checkout verification."
    exit 1
fi
if ! GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null)"; then
    echo "BLOCK: cannot resolve the source checkout commit."
    exit 1
fi
export AGRIBRAIN_GIT_COMMIT="$GIT_COMMIT"

echo "=== AgriBrain HPC orchestrator ==="
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Commit: ${GIT_COMMIT}"
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
source hpc/publication_env.sh
PUBLICATION_PYTHON_BIN="${AGRIBRAIN_PYTHON_BIN:-python3.11}"
if ! command -v "$PUBLICATION_PYTHON_BIN" >/dev/null 2>&1; then
    echo "BLOCK: Python 3.11 is required for publication evidence."
    echo "       Set AGRIBRAIN_PYTHON_BIN to the cluster's Python 3.11 executable."
    exit 1
fi
"$PUBLICATION_PYTHON_BIN" hpc/validate_launch_preflight.py --workflow core
"$PUBLICATION_PYTHON_BIN" hpc/validate_pinn_artifacts.py
"$PUBLICATION_PYTHON_BIN" hpc/validate_publication_env.py
"$PUBLICATION_PYTHON_BIN" hpc/validate_source_checkout.py

# Resolve the run identity and create a detached, run-scoped source snapshot
# before environment creation. A queued or long-running worker therefore
# cannot observe later edits in the operator's checkout.
export RUN_TAG="${GIT_COMMIT:0:7}_$(date +%Y%m%d_%H%M%S)"
ORCHESTRATOR_ROOT="$(pwd -P)"
SOURCE_SNAPSHOT_PARENT="${ORCHESTRATOR_ROOT}/.publication_sources"
SOURCE_SNAPSHOT_PATH="${SOURCE_SNAPSHOT_PARENT}/${RUN_TAG}"
mkdir -p "$SOURCE_SNAPSHOT_PARENT"
if [ -e "$SOURCE_SNAPSHOT_PATH" ]; then
    echo "BLOCK: source snapshot path already exists: ${SOURCE_SNAPSHOT_PATH}"
    exit 1
fi
if ! git worktree add --detach "$SOURCE_SNAPSHOT_PATH" "$GIT_COMMIT"; then
    echo "BLOCK: could not create the detached publication source snapshot."
    exit 1
fi
export AGRIBRAIN_SOURCE_SNAPSHOT="$(cd "$SOURCE_SNAPSHOT_PATH" && pwd -P)"
export AGRIBRAIN_SOURCE_SNAPSHOT_MODE="detached_readonly_git_worktree_v1"
cd "$AGRIBRAIN_SOURCE_SNAPSHOT"
"$PUBLICATION_PYTHON_BIN" hpc/validate_source_checkout.py

# Each submission gets a new venv inside its detached snapshot and never
# reuses or deletes an existing environment. A collision is a hard failure.
export AGRIBRAIN_VENV=".publication_venvs/${RUN_TAG}"
export CORE_SUBMISSION_RECEIPT="mvp/simulation/results/core_submission_receipts/${RUN_TAG}.json"
mkdir -p .publication_venvs
# Claim the exact run path atomically. Two submissions launched in the same
# second must not both pass a check-then-create race and write into one venv.
if ! mkdir "$AGRIBRAIN_VENV"; then
    echo "BLOCK: run-scoped venv already exists: ${AGRIBRAIN_VENV}"
    echo "       Refusing to reuse or overwrite it; submit with a new RUN_TAG."
    exit 1
fi
"$PUBLICATION_PYTHON_BIN" -m venv "$AGRIBRAIN_VENV"
source "$AGRIBRAIN_VENV/bin/activate"
python -m pip install -r agribrain/backend/requirements-lock.txt --quiet
# Legacy setuptools builds can leave ``build/`` under the input source tree.
# Stage the backend inside the already ignored, run-scoped venv so package
# construction cannot dirty the immutable publication checkout.
BACKEND_BUILD_SRC="${AGRIBRAIN_VENV}/backend-build-source"
mkdir "$BACKEND_BUILD_SRC"
cp -a agribrain/backend/. "$BACKEND_BUILD_SRC/"
python -m pip install "$BACKEND_BUILD_SRC" --no-deps --quiet
python -m pip check
python hpc/capture_publication_environment.py --validate-only
python hpc/validate_pinn_artifacts.py
# Recheck after both installation steps.  This catches any build tool that
# unexpectedly writes into the checkout before costly arrays are submitted.
python hpc/validate_source_checkout.py

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
expected = {
    'calculator', 'chain_query', 'check_compliance', 'context_features',
    'convert_units', 'demand_query', 'explain', 'footprint_query',
    'pirag_query', 'policy_oracle', 'slca_lookup', 'spoilage_forecast',
    'yield_query',
}
assert names == expected, f'BLOCK: MCP registry drift: {sorted(names ^ expected)}'
from src.models.action_selection import THETA
assert THETA_CONTEXT.shape == (3, 5), 'BLOCK: THETA_CONTEXT shape not (3,5)'
assert THETA.shape == (3, 10), 'BLOCK: THETA shape not (3,10)'
print('Pre-flight invariants OK')
"

# Remove every write bit from tracked source files outside the evidence-output
# tree, then export one literal-byte tree digest. Every worker checks this
# digest before and after its expensive computation. The unique detached
# worktree plus read-only tracked files prevents accidental mid-run code drift.
while IFS= read -r -d '' tracked_path; do
    case "$tracked_path" in
        mvp/simulation/results/*) ;;
        *) chmod a-w -- "$tracked_path" ;;
    esac
done < <(git ls-files -z)
export AGRIBRAIN_SOURCE_TREE_SHA256="$(
    python hpc/validate_source_snapshot.py --print-digest
)"
if [ -z "$AGRIBRAIN_SOURCE_TREE_SHA256" ]; then
    echo "BLOCK: source snapshot digest was not produced."
    exit 1
fi
python hpc/validate_source_snapshot.py

echo ""
echo "RUN_TAG=${RUN_TAG}"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "VENV=${AGRIBRAIN_VENV}"

mkdir -p logs

# Submit the 20-task seed array.  publication_env.sh has already replaced
# every treatment-relevant ambient value with the declared canonical setting.
# --partition is passed explicitly so clusters without a system default
# (common at many sites) do not reject the submit.
SEED_SUBMISSION=$(sbatch --parsable \
    --partition="$PARTITION" \
    --chdir="$AGRIBRAIN_SOURCE_SNAPSHOT" \
    --export=ALL,RUN_TAG="$RUN_TAG",DETERMINISTIC_MODE=false,AGRIBRAIN_GIT_COMMIT="$GIT_COMMIT",AGRIBRAIN_VENV="$AGRIBRAIN_VENV",AGRIBRAIN_SOURCE_SNAPSHOT="$AGRIBRAIN_SOURCE_SNAPSHOT",AGRIBRAIN_SOURCE_SNAPSHOT_MODE="$AGRIBRAIN_SOURCE_SNAPSHOT_MODE",AGRIBRAIN_SOURCE_TREE_SHA256="$AGRIBRAIN_SOURCE_TREE_SHA256" hpc/hpc_seed.sh)
SEED_JOB="${SEED_SUBMISSION%%;*}"
if [[ ! "$SEED_JOB" =~ ^[0-9]+$ ]]; then
    echo "BLOCK: sbatch returned an invalid seed job id: ${SEED_SUBMISSION}"
    exit 1
fi
echo "Submitted seed array as job ${SEED_JOB}"

# Submit the five-scenario H3 stress array only after the primary benchmark
# succeeds. Each task runs the stressed AGRI-BRAIN arms for all 20 seeds in one
# scenario and reuses the matching frozen primary nominal endpoints.
STRESS_SUBMISSION=$(sbatch --parsable \
    --partition="$PARTITION" \
    --chdir="$AGRIBRAIN_SOURCE_SNAPSHOT" \
    --dependency=afterok:${SEED_JOB} \
    --export=ALL,RUN_TAG="$RUN_TAG",DETERMINISTIC_MODE=false,AGRIBRAIN_GIT_COMMIT="$GIT_COMMIT",AGRIBRAIN_VENV="$AGRIBRAIN_VENV",AGRIBRAIN_SOURCE_SNAPSHOT="$AGRIBRAIN_SOURCE_SNAPSHOT",AGRIBRAIN_SOURCE_SNAPSHOT_MODE="$AGRIBRAIN_SOURCE_SNAPSHOT_MODE",AGRIBRAIN_SOURCE_TREE_SHA256="$AGRIBRAIN_SOURCE_TREE_SHA256" hpc/hpc_stress.sh)
STRESS_JOB="${STRESS_SUBMISSION%%;*}"
if [[ ! "$STRESS_JOB" =~ ^[0-9]+$ ]]; then
    echo "BLOCK: sbatch returned an invalid stress job id: ${STRESS_SUBMISSION}"
    exit 1
fi
echo "Submitted stress array as job ${STRESS_JOB} (depends on ${SEED_JOB})"

# Submit the aggregation job with a dependency on the array completing OK.
AGG_SUBMISSION=$(sbatch --parsable \
    --partition="$PARTITION" \
    --chdir="$AGRIBRAIN_SOURCE_SNAPSHOT" \
    --export=ALL,RUN_TAG="$RUN_TAG",DETERMINISTIC_MODE=false,AGRIBRAIN_GIT_COMMIT="$GIT_COMMIT",AGRIBRAIN_VENV="$AGRIBRAIN_VENV",AGRIBRAIN_SOURCE_SNAPSHOT="$AGRIBRAIN_SOURCE_SNAPSHOT",AGRIBRAIN_SOURCE_SNAPSHOT_MODE="$AGRIBRAIN_SOURCE_SNAPSHOT_MODE",AGRIBRAIN_SOURCE_TREE_SHA256="$AGRIBRAIN_SOURCE_TREE_SHA256" \
    --dependency=afterok:${SEED_JOB}:${STRESS_JOB} hpc/hpc_publish.sh)
AGG_JOB="${AGG_SUBMISSION%%;*}"
if [[ ! "$AGG_JOB" =~ ^[0-9]+$ ]]; then
    echo "BLOCK: sbatch returned an invalid publisher job id: ${AGG_SUBMISSION}"
    exit 1
fi
echo "Submitted aggregation as job ${AGG_JOB} (depends on ${SEED_JOB} and ${STRESS_JOB})"

# Record the exact three-stage Slurm DAG after every submission succeeds.
# The receipt is self-hashed, run-scoped, refuses overwrite, and is exported to
# the dependent publisher so final evidence validation can require this exact
# partition/job/dependency identity rather than infer execution from outputs.
python hpc/core_submission_receipt.py create \
    --output "$CORE_SUBMISSION_RECEIPT" \
    --repo-root . \
    --run-tag "$RUN_TAG" \
    --source-commit "$GIT_COMMIT" \
    --partition "$PARTITION" \
    --seed-job-id "$SEED_JOB" \
    --stress-job-id "$STRESS_JOB" \
    --publisher-job-id "$AGG_JOB" \
    --source-snapshot-mode "$AGRIBRAIN_SOURCE_SNAPSHOT_MODE" \
    --source-tree-sha256 "$AGRIBRAIN_SOURCE_TREE_SHA256"
python hpc/core_submission_receipt.py validate \
    --receipt "$CORE_SUBMISSION_RECEIPT" \
    --run-tag "$RUN_TAG" \
    --source-commit "$GIT_COMMIT"

echo ""
echo "Queue:"
squeue -u "$USER"

echo ""
echo "Transfer both verified atomic bundles after completion:"
echo "  scp -r <hpc-host>:${AGRIBRAIN_SOURCE_SNAPSHOT}/publication_bundle_${RUN_TAG}/ ."
echo "  scp -r <hpc-host>:${AGRIBRAIN_SOURCE_SNAPSHOT}/mvp/simulation/results/complete_run_evidence/${RUN_TAG}/ ."
echo "The first bundle contains publication artifacts; the second preserves the"
echo "lossless episode/decision evidence for future reviewer analyses."
