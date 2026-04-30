#!/bin/bash
# SLURM aggregation job: runs after the seed array completes.
# Stage 1 (generate_results), Stage 3 (context benchmark aggregator),
# Stage 5 (canonical aggregate_seeds), Stage 6 (stress suite),
# Stage 7 (figures), and the paper-evidence / manifest / validation stages.
#SBATCH --job-name=agribrain-aggregate
# 24h wall-time. Realistic runtime is 1-2h (stress suite is the longest
# single stage at ~1h). 24h is a full day of headroom in case the stress
# suite balloons or one of the generate_figures stages hangs on a slow
# matplotlib backend. Well under the 14-day compute partition cap.
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ ! -f .venv/bin/activate ]; then
    echo "BLOCK: .venv not found. Run hpc/hpc_run.sh first."
    exit 1
fi
source .venv/bin/activate

if [ -z "${RUN_TAG:-}" ]; then
    echo "BLOCK: RUN_TAG not exported. Submit via hpc/hpc_run.sh."
    exit 1
fi

# Stage 1's generate_results.py also reads DETERMINISTIC_MODE; pin it so the
# aggregator cannot regenerate base tables in deterministic mode while the
# seed array ran stochastic.
if [ "${DETERMINISTIC_MODE:-false}" = "true" ]; then
    echo "BLOCK: DETERMINISTIC_MODE=true reached the aggregator. Stochastic run expected."
    exit 1
fi
export DETERMINISTIC_MODE=false

# Validator policy on HPC: REPORT mode, not strict.
#
# `validate_results.py` defaults to STRICT_VALIDATION=1 (intentional —
# local devs and CI should see a hard gate when range checks fire so a
# clearly broken simulator setup fails fast). On HPC we explicitly opt
# out: a single legitimate range drift (e.g. an HPC seed lands ARI at
# 0.301 vs a 0.30 lower bound under the larger-CV stochastic regime in
# .env.example, or a noisy stress-suite seed pushes equity slightly
# above the calibrated ceiling) would otherwise abort the entire 6–10 h
# pipeline at Stage 2 with `set -euo pipefail`, lose the seed array's
# work, and prevent the tar archive from being produced.
#
# In REPORT mode the validator still runs at Stages 2 and 10, writes
# `validation_report.json` with the full list of flagged ranges, and
# prints them to stdout for human review — but exits 0 so the pipeline
# continues and the manifest + archive are produced. After the run
# lands, audit `validation_report.json` and decide whether the flagged
# items are real range failures or expected drift under the new
# stochastic regime; re-run `STRICT_VALIDATION=1 python
# mvp/simulation/validation/validate_results.py` locally to confirm.
#
# Override on the rare run where you want a strict gate (i.e. you have
# already widened the validator's ranges to match the new CVs and want
# the build to refuse the archive on any deviation):
#     STRICT_VALIDATION=1 sbatch hpc/hpc_aggregate.sh
export STRICT_VALIDATION="${STRICT_VALIDATION:-0}"

SEEDS_DIR="mvp/simulation/results/benchmark_seeds/${RUN_TAG}"
RESULTS_DIR="mvp/simulation/results"

if [ ! -d "$SEEDS_DIR" ]; then
    echo "BLOCK: seeds directory ${SEEDS_DIR} missing. Array job did not produce outputs."
    exit 1
fi

echo "[aggregate tag=${RUN_TAG}] starting at $(date)"
echo "[aggregate] seeds dir: ${SEEDS_DIR}"
echo "[aggregate] results dir: ${RESULTS_DIR}"

# Pre-flight invariants check, same assertion as the seed tasks.
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

# Stage 1: generate base tables at the default results location.
echo ""
echo "=== Stage 1: generate base tables ==="
(cd mvp/simulation && python generate_results.py)

# Stage 2: validation pass on the freshly written tables.
echo ""
echo "=== Stage 2: validate ==="
(cd mvp/simulation && python validation/validate_results.py)

# Stage 3: refactored context-ablation aggregator reads the tagged seeds dir.
echo ""
echo "=== Stage 3: context benchmark aggregator ==="
python mvp/simulation/benchmarks/run_benchmark_suite.py \
    --seeds-dir "$SEEDS_DIR" \
    --output-dir "$RESULTS_DIR"

# Stage 5: canonical aggregator expects flat seed_<n>.json files under
# mvp/simulation/results/benchmark_seeds/. Copy (not symlink) the tagged
# files into the flat location so aggregate_seeds.py can find them
# unchanged. cp works on every HPC filesystem; symlinks can fail on
# some Lustre / NFS tiers. JSONs are <50 KB each, the copy cost is
# negligible vs the benefit of portability.
echo ""
echo "=== Stage 5: canonical aggregate_seeds ==="
mkdir -p "$RESULTS_DIR/benchmark_seeds"
for f in "$SEEDS_DIR"/seed_*.json; do
    [ -e "$f" ] || continue
    cp -f "$f" "$RESULTS_DIR/benchmark_seeds/$(basename "$f")"
done
python mvp/simulation/benchmarks/aggregate_seeds.py

# Stage 6: stress suite (5 scenarios, 4 stressors, ~55 episodes; ~1 h on HPC).
echo ""
echo "=== Stage 6: stress suite ==="
python mvp/simulation/benchmarks/run_stress_suite.py

# Stage 7: figures.
echo ""
echo "=== Stage 7: figures ==="
(cd mvp/simulation && python generate_figures.py)

# Stage 7.5: explainability assessment metrics (§4.10 100/100/100 numbers).
# Walks every per-episode decision_ledger/*.jsonl, recomputes leaf hashes
# and Merkle roots for the provenance integrity check, and recomputes
# the dominant psi feature against THETA_CONTEXT for the sign-consistency
# check. Output goes to results/explainability_metrics.json with a
# one-screen aggregate the paper can cite without re-running anything.
echo ""
echo "=== Stage 7.5: explainability assessment metrics ==="
python -m mvp.simulation.analysis.explainability_metrics \
    --ledger "$RESULTS_DIR/decision_ledger" \
    --output "$RESULTS_DIR/explainability_metrics.json"

# Stage 8: paper evidence export.
echo ""
echo "=== Stage 8: paper evidence ==="
(cd mvp/simulation && python analysis/export_paper_evidence.py)

# Stage 9: artifact manifest.
echo ""
echo "=== Stage 9: artifact manifest ==="
(cd mvp/simulation && python analysis/build_artifact_manifest.py)

# Stage 10: final validation.
echo ""
echo "=== Stage 10: final validation ==="
(cd mvp/simulation && python validation/validate_results.py)

# Archive the results into a tag-specific tarball so re-runs do not clobber.
# Build the file list explicitly so a missing optional output (e.g. one
# figure that did not render) does not fail the whole archive step.
echo ""
echo "=== Packaging ==="
ARCHIVE="hpc_results_${RUN_TAG}.tar.gz"

archive_files=()
for f in \
    mvp/simulation/results/table1_summary.csv \
    mvp/simulation/results/table2_ablation.csv \
    mvp/simulation/results/benchmark_summary.json \
    mvp/simulation/results/benchmark_significance.json \
    mvp/simulation/results/benchmark_context_summary.json \
    mvp/simulation/results/benchmark_context_significance.json \
    mvp/simulation/results/stress_summary.json \
    mvp/simulation/results/stress_degradation.csv \
    mvp/simulation/results/stress_passfail.csv \
    mvp/simulation/results/feature_heatmap_data.json \
    mvp/simulation/results/explainability_metrics.json \
    mvp/simulation/results/artifact_manifest.json
do
    [ -f "$f" ] && archive_files+=("$f")
done

# decision_ledger is a directory of per-episode jsonl files; pull the
# whole tree into the archive so explainability_metrics +
# verify_anchored_root can be re-run from the tarball alone.
[ -d mvp/simulation/results/decision_ledger ] && \
    archive_files+=("mvp/simulation/results/decision_ledger")

shopt -s nullglob
for f in mvp/simulation/results/fig*.png mvp/simulation/results/fig*.pdf; do
    archive_files+=("$f")
done
shopt -u nullglob

[ -d "$SEEDS_DIR" ] && archive_files+=("$SEEDS_DIR")

if [ ${#archive_files[@]} -eq 0 ]; then
    echo "WARNING: no files matched for archive, skipping tar"
else
    tar czf "$ARCHIVE" "${archive_files[@]}"
    echo "[aggregate] archive: $ARCHIVE (${#archive_files[@]} items)"
fi

echo ""
echo "[aggregate] complete at $(date)"
