#!/bin/bash
#SBATCH --job-name=agribrain-bench
#SBATCH --output=agribrain-bench-seed%a-%j.out
#SBATCH --error=agribrain-bench-seed%a-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute
#SBATCH --array=0-4

# ============================================================
# Parallel benchmark: runs 5 seeds as a SLURM array job
# Each array task runs one seed independently (~2-4 hrs each)
# All 5 run in parallel = same wall time as 1
# ============================================================

SEEDS=(42 1337 2024 7 99)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID, Seed: $SEED"

cd $HOME/AgriBrain
source .venv/bin/activate 2>/dev/null || module load python/3.11

export DETERMINISTIC_MODE=false
export FORECAST_METHOD=lstm

cd mvp/simulation

# Run simulation for this seed, save results to seed-specific directory
python -c "
import json, os, sys
from pathlib import Path
from generate_results import run_all, SCENARIOS

seed = $SEED
print(f'Running seed={seed}')
data = run_all(seed=seed)

# Save per-seed metrics for later aggregation
out_dir = Path('results/benchmark_seeds')
out_dir.mkdir(parents=True, exist_ok=True)

metrics = {}
for sc in SCENARIOS:
    metrics[sc] = {}
    for mode in ('agribrain', 'mcp_only', 'pirag_only', 'no_context'):
        ep = data['results'][sc][mode]
        metrics[sc][mode] = {
            'ari': float(ep['ari']),
            'waste': float(ep['waste']),
            'rle': float(ep['rle']),
            'slca': float(ep['slca']),
        }

out_file = out_dir / f'seed_{seed}.json'
out_file.write_text(json.dumps(metrics, indent=2))
print(f'Saved: {out_file}')
"

echo "Seed $SEED complete: $(date)"
