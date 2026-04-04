#!/bin/bash
# Submit the full AgriBrain HPC pipeline:
#   1. Generate base results (single seed)
#   2. Parallel benchmark (5 seeds simultaneously)
#   3. Aggregate results + regenerate figures
#
# Usage:
#   cd ~/AgriBrain/hpc
#   bash submit_all.sh

set -e

echo "=== AgriBrain HPC Pipeline ==="

# Step 1: Base results
GEN_JOB=$(sbatch --parsable run_generate.slurm)
echo "Submitted generate_results: Job $GEN_JOB"

# Step 2: Parallel benchmark (5 seeds, runs after generate completes)
BENCH_JOB=$(sbatch --parsable --dependency=afterok:$GEN_JOB run_benchmark_parallel.slurm)
echo "Submitted benchmark (5 parallel seeds): Job $BENCH_JOB"

# Step 3: Aggregate (runs after all benchmark seeds complete)
sed -i "s/REPLACE_WITH_ARRAY_JOB_ID/$BENCH_JOB/" aggregate_benchmark.slurm
AGG_JOB=$(sbatch --parsable --dependency=afterok:$BENCH_JOB aggregate_benchmark.slurm)
echo "Submitted aggregation: Job $AGG_JOB"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Expected total wall time: ~4-6 hours (stages run in sequence, seeds in parallel)"
