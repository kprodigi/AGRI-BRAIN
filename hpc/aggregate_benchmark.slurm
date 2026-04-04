#!/bin/bash
#SBATCH --job-name=agribrain-aggregate
#SBATCH --output=agribrain-aggregate-%j.out
#SBATCH --error=agribrain-aggregate-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=compute
#SBATCH --dependency=afterok:REPLACE_WITH_ARRAY_JOB_ID

# ============================================================
# Aggregate parallel benchmark results into summary files
# Submit AFTER the array job completes:
#   ARRAY_JOB_ID=$(sbatch --parsable run_benchmark_parallel.slurm)
#   sbatch --dependency=afterok:$ARRAY_JOB_ID aggregate_benchmark.slurm
# ============================================================

echo "Aggregation started: $(date)"

cd $HOME/AgriBrain
source .venv/bin/activate 2>/dev/null || module load python/3.11

cd mvp/simulation

python -c "
import json, sys
import numpy as np
from pathlib import Path

SEEDS = [42, 1337, 2024, 7, 99]
SCENARIOS = ['heatwave', 'overproduction', 'cyber_outage', 'adaptive_pricing', 'baseline']
seed_dir = Path('results/benchmark_seeds')

# Load all seed results
all_data = {}
for seed in SEEDS:
    f = seed_dir / f'seed_{seed}.json'
    if not f.exists():
        print(f'WARNING: {f} not found, skipping')
        continue
    all_data[seed] = json.loads(f.read_text())
    print(f'Loaded seed {seed}')

if len(all_data) < 2:
    print('ERROR: Need at least 2 seeds for statistics')
    sys.exit(1)

print(f'Aggregating {len(all_data)} seeds...')

def bootstrap_ci(vals, n_boot=1000, alpha=0.05):
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(42)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1-alpha/2))

def permutation_pvalue(a, b, n_perm=4000):
    x, y = np.array(a), np.array(b)
    observed = abs(float(np.mean(x) - np.mean(y)))
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(123)
    ge = sum(1 for _ in range(n_perm) if abs(float(np.mean(p := rng.permutation(pooled)[:len(x)]) - np.mean(pooled[len(x):]))) >= observed)
    return float((ge + 1) / (n_perm + 1))

def cohens_d(a, b):
    x, y = np.array(a), np.array(b)
    if len(x) < 2 or len(y) < 2: return 0.0
    pooled = ((len(x)-1)*np.var(x,ddof=1) + (len(y)-1)*np.var(y,ddof=1)) / max(len(x)+len(y)-2, 1)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled)) if pooled > 0 else 0.0

# Build summary
summary = {}
for sc in SCENARIOS:
    summary[sc] = {}
    for mode in ('agribrain', 'mcp_only', 'pirag_only', 'no_context'):
        summary[sc][mode] = {}
        for metric in ('ari', 'waste', 'rle', 'slca'):
            vals = [all_data[s][sc][mode][metric] for s in all_data]
            lo, hi = bootstrap_ci(vals)
            summary[sc][mode][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'ci_low': lo,
                'ci_high': hi,
            }

# Build significance
significance = {}
for sc in SCENARIOS:
    significance[sc] = {}
    agri_data = {m: [all_data[s][sc]['agribrain'][m] for s in all_data] for m in ('ari','waste','rle','slca')}
    for baseline in ('mcp_only', 'pirag_only', 'no_context'):
        base_data = {m: [all_data[s][sc][baseline][m] for s in all_data] for m in ('ari','waste','rle','slca')}
        comp = {}
        for metric in ('ari', 'waste', 'rle', 'slca'):
            comp[metric] = {
                'p_value': permutation_pvalue(agri_data[metric], base_data[metric]),
                'cohens_d': cohens_d(agri_data[metric], base_data[metric]),
                'mean_diff': float(np.mean(agri_data[metric]) - np.mean(base_data[metric])),
            }
        significance[sc][f'agribrain_vs_{baseline}'] = comp

# Save
Path('results').mkdir(exist_ok=True)
Path('results/benchmark_summary.json').write_text(json.dumps(summary, indent=2))
Path('results/benchmark_significance.json').write_text(json.dumps(significance, indent=2))
print('Saved benchmark_summary.json')
print('Saved benchmark_significance.json')

# Print summary
for sc in SCENARIOS:
    ari = summary[sc]['agribrain']['ari']
    print(f\"  {sc}: ARI={ari['mean']:.4f} CI=[{ari['ci_low']:.4f}, {ari['ci_high']:.4f}] std={ari['std']:.6f}\")
"

# Regenerate figures with new benchmark data (for error bars)
python generate_figures.py
python export_paper_evidence.py
python build_artifact_manifest.py

echo "Aggregation complete: $(date)"
