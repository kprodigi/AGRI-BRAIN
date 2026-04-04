# Running AgriBrain on SDSU HPC (SLURM)

## 1. Transfer code to HPC

```bash
# From your local machine
scp -r C:\AgriBrain username@hpc.sdsu.edu:~/AgriBrain
```

Or use `git clone` on the HPC:
```bash
ssh username@hpc.sdsu.edu
git clone https://github.com/kprodigi/AGRI-BRAIN.git AgriBrain
```

## 2. Set up Python environment on HPC

```bash
module load python/3.11    # or whatever version is available
# module load anaconda       # alternative if anaconda is available

cd ~/AgriBrain
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib reportlab orjson pyyaml
```

## 3. Run the full pipeline

### Option A: Single job (sequential, ~10 hours)
```bash
cd ~/AgriBrain/hpc
sbatch run_generate.slurm
```

### Option B: Parallel benchmark (5 seeds simultaneously, ~10 hours total)
```bash
cd ~/AgriBrain/hpc
sbatch run_benchmark_parallel.slurm
```
This submits a SLURM array job — one task per seed, all running in parallel.

### Option C: Full pipeline (generate + benchmark + figures)
```bash
cd ~/AgriBrain/hpc
sbatch run_full_pipeline.slurm
```

## 4. Check job status
```bash
squeue -u $USER
```

## 5. Retrieve results

After jobs complete, results are in `mvp/simulation/results/`. Transfer back:
```bash
# From your local machine
scp -r username@hpc.sdsu.edu:~/AgriBrain/mvp/simulation/results/ C:\AgriBrain\mvp\simulation\results\
```

## Files produced
- `table1_summary.csv` — main results table
- `table2_ablation.csv` — ablation study
- `benchmark_summary.json` — multi-seed CIs
- `benchmark_significance.json` — p-values, Cohen's d
- `results/fig*.png` and `fig*.pdf` — publication figures
- `artifact_manifest.json` — SHA-256 hashes
