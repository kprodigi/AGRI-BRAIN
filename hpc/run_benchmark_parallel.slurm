#!/bin/bash
#SBATCH --job-name=agribrain-bench
#SBATCH --output=agribrain-bench-seed%a-%j.out
#SBATCH --error=agribrain-bench-seed%a-%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --array=0-4

# ============================================================
# Stage 2: Parallel benchmark — 5 seeds as SLURM array job
# Each array task runs one seed independently (~1-2 hrs each)
# All 5 run simultaneously = same wall time as 1
# SDSU Innovator: 48 cores/node, 256 GB RAM
# ============================================================

SEEDS=(42 1337 2024 7 99)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== AgriBrain Benchmark Seed $SEED ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

cd $HOME/AgriBrain

# Load Python and activate venv
module load python 2>/dev/null || true
source .venv/bin/activate

export DETERMINISTIC_MODE=false
export FORECAST_METHOD=lstm

cd mvp/simulation

python -c "
import json, sys
from pathlib import Path
from generate_results import run_all, SCENARIOS

seed = $SEED
print(f'Running full simulation with seed={seed}...')
data = run_all(seed=seed)

# Save per-seed metrics for later aggregation
out_dir = Path('results/benchmark_seeds')
out_dir.mkdir(parents=True, exist_ok=True)

metrics = {}
for sc in SCENARIOS:
    metrics[sc] = {}
    for mode in ('agribrain', 'mcp_only', 'pirag_only', 'no_context',
                 'static', 'hybrid_rl', 'no_pinn', 'no_slca'):
        ep = data['results'][sc][mode]
        metrics[sc][mode] = {
            'ari': float(ep['ari']),
            'waste': float(ep['waste']),
            'rle': float(ep['rle']),
            'slca': float(ep['slca']),
            'carbon': float(ep['carbon']),
            'equity': float(ep['equity']),
        }

out_file = out_dir / f'seed_{seed}.json'
out_file.write_text(json.dumps(metrics, indent=2))
print(f'Saved: {out_file}')
print(f'Done: seed {seed}')
"

echo ""
echo "=== Seed $SEED complete: $(date) ==="
