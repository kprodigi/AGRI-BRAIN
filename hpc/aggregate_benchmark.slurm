#!/bin/bash
#SBATCH --job-name=agribrain-agg
#SBATCH --output=agribrain-aggregate-%j.out
#SBATCH --error=agribrain-aggregate-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute

# ============================================================
# Stage 3: Aggregate 5 seed results into benchmark summary
# Then regenerate figures (with error bars) and evidence
# Expected runtime: ~5 minutes
# ============================================================

echo "=== AgriBrain Aggregate Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
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

n_seeds = len(all_data)
if n_seeds < 2:
    print(f'ERROR: Only {n_seeds} seed(s) found, need at least 2')
    sys.exit(1)

print(f'Aggregating {n_seeds} seeds...')

def bootstrap_ci(vals, n_boot=1000, alpha=0.05):
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(42)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))

def permutation_pvalue(a, b, n_perm=4000):
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    observed = abs(float(np.mean(x) - np.mean(y)))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    rng = np.random.default_rng(123)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(float(np.mean(perm[:n_x]) - np.mean(perm[n_x:])))
        if diff >= observed:
            ge += 1
    return float((ge + 1) / (n_perm + 1))

def cohens_d(a, b):
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((len(x) - 1) * sx2 + (len(y) - 1) * sy2) / max(len(x) + len(y) - 2, 1)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled)) if pooled > 0 else 0.0

# Build summary (all 8 modes)
all_modes = ['agribrain', 'mcp_only', 'pirag_only', 'no_context',
             'static', 'hybrid_rl', 'no_pinn', 'no_slca']
summary = {}
for sc in SCENARIOS:
    summary[sc] = {}
    for mode in all_modes:
        summary[sc][mode] = {}
        for metric in ('ari', 'waste', 'rle', 'slca'):
            vals = [all_data[s][sc][mode][metric] for s in all_data if mode in all_data[s][sc]]
            if not vals:
                continue
            lo, hi = bootstrap_ci(vals)
            summary[sc][mode][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'ci_low': lo,
                'ci_high': hi,
                'n_seeds': len(vals),
            }

# Build significance (agribrain vs ablations)
significance = {}
for sc in SCENARIOS:
    significance[sc] = {}
    agri_data = {}
    for m in ('ari', 'waste', 'rle', 'slca'):
        agri_data[m] = [all_data[s][sc]['agribrain'][m] for s in all_data]

    for baseline in ('mcp_only', 'pirag_only', 'no_context', 'hybrid_rl', 'static'):
        base_data = {}
        for m in ('ari', 'waste', 'rle', 'slca'):
            base_data[m] = [all_data[s][sc][baseline][m] for s in all_data]

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

# Print key results
print()
print('=== Stochastic Benchmark Results ===')
for sc in SCENARIOS:
    ari = summary[sc]['agribrain']['ari']
    print(f\"  {sc}: ARI mean={ari['mean']:.4f} CI=[{ari['ci_low']:.4f}, {ari['ci_high']:.4f}] std={ari['std']:.6f}\")
print()
for sc in SCENARIOS:
    for comp in ('agribrain_vs_no_context', 'agribrain_vs_hybrid_rl'):
        rec = significance[sc][comp]['ari']
        print(f\"  {sc} {comp}: p={rec['p_value']:.4f} d={rec['cohens_d']:+.3f} diff={rec['mean_diff']:+.4f}\")
"

echo ""
echo "Regenerating figures with benchmark error bars..."
python generate_figures.py

echo "Exporting paper evidence..."
python export_paper_evidence.py

echo "Building artifact manifest..."
python build_artifact_manifest.py

echo ""
echo "=== Aggregation complete: $(date) ==="
echo "Results in: $(pwd)/results/"
