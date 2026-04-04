#!/bin/bash
# ============================================================
# AgriBrain Full Pipeline for SDSU Innovator HPC
# Submits 3 chained SLURM jobs:
#   1. Generate base results (1 seed)
#   2. Parallel benchmark (5 seeds simultaneously)
#   3. Aggregate + regenerate figures
#
# Usage:
#   cd ~/AgriBrain/hpc
#   bash submit_all.sh
#
# Total wall time: ~2-4 hours
# ============================================================

set -e
echo "============================================"
echo "AgriBrain HPC Pipeline — SDSU Innovator"
echo "============================================"
echo ""

# Step 1: Base results (generates tables, traces, protocol logs)
GEN_JOB=$(sbatch --parsable run_generate.slurm)
echo "[1/3] Submitted generate_results:     Job $GEN_JOB"

# Step 2: Parallel benchmark — 5 seeds run simultaneously
# Depends on Step 1 completing successfully
BENCH_JOB=$(sbatch --parsable --dependency=afterok:$GEN_JOB run_benchmark_parallel.slurm)
echo "[2/3] Submitted benchmark (5 seeds):  Job $BENCH_JOB (array, depends on $GEN_JOB)"

# Step 3: Aggregate all seed results + regenerate figures with error bars
# Depends on ALL 5 benchmark seeds completing
AGG_JOB=$(sbatch --parsable --dependency=afterok:$BENCH_JOB aggregate_benchmark.slurm)
echo "[3/3] Submitted aggregation:          Job $AGG_JOB (depends on $BENCH_JOB)"

echo ""
echo "Pipeline submitted successfully."
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f agribrain-generate-*.out"
echo ""
echo "When all jobs complete, results are in:"
echo "  ~/AgriBrain/mvp/simulation/results/"
echo ""
echo "Transfer to your laptop with:"
echo "  scp -r $(whoami)@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/ C:\\AgriBrain\\mvp\\simulation\\results\\"
