#!/bin/bash
#SBATCH --job-name=agribrain-generate
#SBATCH --output=agribrain-generate-%j.out
#SBATCH --error=agribrain-generate-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute

# ============================================================
# Stage 1: Generate base results (single seed, stochastic mode)
# Expected runtime: ~2-4 hours on HPC
# ============================================================

echo "Job started: $(date)"
echo "Node: $(hostname)"

cd $HOME/AgriBrain

# Activate environment
source .venv/bin/activate 2>/dev/null || module load python/3.11

# Stochastic mode with LSTM demand forecasting
export DETERMINISTIC_MODE=false
export FORECAST_METHOD=lstm

cd mvp/simulation
python generate_results.py

echo ""
echo "Generate results done: $(date)"
echo "Running validation..."

python validate_results.py
python generate_figures.py
python export_paper_evidence.py
python build_artifact_manifest.py

echo ""
echo "Full pipeline complete: $(date)"
