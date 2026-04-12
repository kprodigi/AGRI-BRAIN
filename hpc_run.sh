#!/bin/bash
#SBATCH --job-name=agribrain-repro
#SBATCH --output=agribrain-repro-%j.out
#SBATCH --error=agribrain-repro-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# =========================================================================
# AGRI-BRAIN Full Reproducibility Pipeline for HPC
#
# This script clones the repo, installs deps, runs the complete simulation
# pipeline with 20 seeds, and packages all outputs for transfer back.
#
# Usage:
#   Option A (SLURM): sbatch hpc_run.sh
#   Option B (direct): bash hpc_run.sh
#
# After completion, copy hpc_results.tar.gz back to your local machine.
# =========================================================================
set -euo pipefail

echo "=== AGRI-BRAIN HPC Reproducibility Pipeline ==="
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Python: $(python3 --version)"

# --- Setup ---
WORK_DIR="${WORK_DIR:-$(pwd)/agribrain-hpc-run}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone fresh (or pull if already exists)
if [ -d "AGRI-BRAIN" ]; then
    echo "Repo exists, pulling latest..."
    cd AGRI-BRAIN && git pull origin main && cd ..
else
    echo "Cloning repo..."
    git clone https://github.com/kprodigi/AGRI-BRAIN.git
fi
cd AGRI-BRAIN

# Virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -e agri-brain-mvp-1.0.0/backend
pip install pytest

# --- Verify setup ---
echo ""
echo "=== Verification ==="
python -m pytest agri-brain-mvp-1.0.0/backend/pirag/tests agri-brain-mvp-1.0.0/backend/tests -q
echo "Backend tests: PASS"

# --- Run full pipeline ---
echo ""
echo "=== Stage 1: Generate base results (5 scenarios x 8 modes) ==="
cd mvp/simulation
python generate_results.py
echo "Base results: DONE"

echo ""
echo "=== Stage 2: Validate results ==="
python validation/validate_results.py
echo "Validation: PASS"

echo ""
echo "=== Stage 3: Run benchmark suite (20 seeds) ==="
export BENCHMARK_SEEDS="42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515"
python benchmarks/run_benchmark_suite.py
echo "Benchmark suite: DONE"

echo ""
echo "=== Stage 4: Run per-seed episodes ==="
for seed in 42 1337 2024 7 99 101 202 303 404 505 606 707 808 909 1010 1111 1212 1313 1414 1515; do
    echo "  Seed $seed..."
    python benchmarks/run_single_seed.py $seed
done
echo "Per-seed runs: DONE"

echo ""
echo "=== Stage 5: Aggregate seeds (summary + significance) ==="
python benchmarks/aggregate_seeds.py
echo "Aggregation: DONE"

echo ""
echo "=== Stage 6: Run stress suite (full length, all scenarios) ==="
python benchmarks/run_stress_suite.py
echo "Stress suite: DONE"

echo ""
echo "=== Stage 7: Generate figures ==="
python generate_figures.py
echo "Figures: DONE"

echo ""
echo "=== Stage 8: Export paper evidence ==="
python analysis/export_paper_evidence.py
echo "Paper evidence: DONE"

echo ""
echo "=== Stage 9: Build artifact manifest ==="
python analysis/build_artifact_manifest.py
echo "Manifest: DONE"

echo ""
echo "=== Stage 10: Final validation ==="
python validation/validate_results.py
echo "Final validation: PASS"

# --- Package outputs ---
echo ""
echo "=== Packaging results ==="
cd "$WORK_DIR/AGRI-BRAIN"

tar czf "$WORK_DIR/hpc_results.tar.gz" \
    mvp/simulation/results/table1_summary.csv \
    mvp/simulation/results/table2_ablation.csv \
    mvp/simulation/results/benchmark_summary.json \
    mvp/simulation/results/benchmark_significance.json \
    mvp/simulation/results/stress_summary.json \
    mvp/simulation/results/stress_degradation.csv \
    mvp/simulation/results/feature_heatmap_data.json \
    mvp/simulation/results/artifact_manifest.json \
    mvp/simulation/results/fig*.png \
    mvp/simulation/results/fig*.pdf \
    mvp/simulation/baseline_snapshot.json

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo "Output: $WORK_DIR/hpc_results.tar.gz"
echo ""
echo "Transfer to local machine with:"
echo "  scp <hpc-host>:$WORK_DIR/hpc_results.tar.gz ."
echo ""
echo "Then on local machine, run:"
echo "  cd C:/AgriBrain"
echo "  tar xzf hpc_results.tar.gz"
echo "  # Files land in mvp/simulation/results/ — ready to commit"
