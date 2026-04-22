#!/bin/bash
# SLURM aggregation job: runs after the seed array completes.
# Stage 1 (generate_results), Stage 3 (context benchmark aggregator),
# Stage 5 (canonical aggregate_seeds), Stage 6 (stress suite),
# Stage 7 (figures), and the paper-evidence / manifest / validation stages.
#SBATCH --job-name=agribrain-aggregate
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ ! -f .venv/bin/activate ]; then
    echo "BLOCK: .venv not found. Run hpc_run.sh first."
    exit 1
fi
source .venv/bin/activate

if [ -z "${RUN_TAG:-}" ]; then
    echo "BLOCK: RUN_TAG not exported. Submit via hpc_run.sh."
    exit 1
fi

SEEDS_DIR="mvp/simulation/results/benchmark_seeds/${RUN_TAG}"
RESULTS_DIR="mvp/simulation/results"

if [ ! -d "$SEEDS_DIR" ]; then
    echo "BLOCK: seeds directory ${SEEDS_DIR} missing. Array job did not produce outputs."
    exit 1
fi

echo "[aggregate tag=${RUN_TAG}] starting at $(date)"
echo "[aggregate] seeds dir: ${SEEDS_DIR}"
echo "[aggregate] results dir: ${RESULTS_DIR}"

# Path B sanity check, same assertion as the seed tasks.
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
# mvp/simulation/results/benchmark_seeds/. Link the tagged files into the
# flat location so aggregate_seeds.py can find them unchanged.
echo ""
echo "=== Stage 5: canonical aggregate_seeds ==="
for f in "$SEEDS_DIR"/seed_*.json; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$RESULTS_DIR/benchmark_seeds/$(basename "$f")"
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
echo ""
echo "=== Packaging ==="
ARCHIVE="hpc_results_${RUN_TAG}.tar.gz"
tar czf "$ARCHIVE" \
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
    mvp/simulation/results/artifact_manifest.json \
    "mvp/simulation/results/fig"*.png \
    "mvp/simulation/results/fig"*.pdf \
    mvp/simulation/baseline_snapshot.json \
    "$SEEDS_DIR"

echo ""
echo "[aggregate] complete at $(date)"
echo "[aggregate] archive: $ARCHIVE"
