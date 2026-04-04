#!/bin/bash
#SBATCH --job-name=agribrain-gen
#SBATCH --output=agribrain-generate-%j.out
#SBATCH --error=agribrain-generate-%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=compute

# ============================================================
# Stage 1: Generate base results (single seed, stochastic mode)
# SDSU Innovator: 48 cores/node, 256 GB RAM
# Expected runtime: ~1-2 hours
# ============================================================

echo "=== AgriBrain Generate Results ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

cd $HOME/AgriBrain

# Load Python and activate venv
module load python 2>/dev/null || true
source .venv/bin/activate

# Stochastic mode with LSTM demand forecasting
export DETERMINISTIC_MODE=false
export FORECAST_METHOD=lstm

cd mvp/simulation
echo "Running generate_results.py..."
python generate_results.py

echo ""
echo "Running post-generation stages..."
python validate_results.py
python generate_figures.py
python export_paper_evidence.py
python build_artifact_manifest.py

echo ""
echo "=== Stage 1 complete: $(date) ==="
