# Running AgriBrain on SDSU Innovator HPC

**Cluster:** Innovator (Rocky 9 Linux, 46 compute nodes, 48 cores each, 256 GB RAM)
**Scheduler:** SLURM
**Login:** `innovator.sdstate.edu`

---

## Step 1: SSH into the cluster

```bash
ssh firstname.lastname@jacks.local@innovator.sdstate.edu
```

Enter your SDSU password when prompted.

---

## Step 2: Clone the repo

```bash
cd ~
git clone https://github.com/kprodigi/AGRI-BRAIN.git AgriBrain
cd AgriBrain
```

---

## Step 3: Set up Python environment

```bash
# Check available Python modules
module avail python

# Load Python (use whatever version is available)
module load python

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy pandas matplotlib reportlab orjson pyyaml requests
```

---

## Step 4: Run the full pipeline (one command)

```bash
cd ~/AgriBrain/hpc
bash submit_all.sh
```

This submits 3 chained SLURM jobs:
1. **generate_results** — base simulation (1 seed, ~1-2 hrs on 48-core node)
2. **benchmark** — 5 seeds in parallel (SLURM array, ~1-2 hrs same wall time)
3. **aggregate** — combine results + generate figures (~5 min)

**Total wall time: ~2-4 hours**

---

## Step 5: Monitor progress

```bash
# Check all your jobs
squeue -u $USER

# Watch job output in real time
tail -f agribrain-generate-*.out

# Check if results are done
ls ~/AgriBrain/mvp/simulation/results/
```

---

## Step 6: Transfer results back to your laptop

From **PowerShell on your Windows machine**:

```powershell
scp -r firstname.lastname@jacks.local@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/ C:\AgriBrain\mvp\simulation\results\
```

Or transfer specific files:
```powershell
scp firstname.lastname@jacks.local@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/table1_summary.csv C:\AgriBrain\mvp\simulation\results\
scp firstname.lastname@jacks.local@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/table2_ablation.csv C:\AgriBrain\mvp\simulation\results\
scp firstname.lastname@jacks.local@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/benchmark_summary.json C:\AgriBrain\mvp\simulation\results\
scp firstname.lastname@jacks.local@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/benchmark_significance.json C:\AgriBrain\mvp\simulation\results\
scp firstname.lastname@jacks.local@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/fig*.png C:\AgriBrain\mvp\simulation\results\
scp firstname.lastname@jacks.local@innovator.sdstate.edu:~/AgriBrain/mvp/simulation/results/fig*.pdf C:\AgriBrain\mvp\simulation\results\
```

---

## Files produced

| File | Description |
|------|-------------|
| `table1_summary.csv` | Main results (5 scenarios x 3 methods) |
| `table2_ablation.csv` | Ablation study (5 scenarios x 8 variants) |
| `benchmark_summary.json` | Multi-seed means, stds, 95% CIs |
| `benchmark_significance.json` | Permutation p-values, Cohen's d |
| `paper_benchmark_table.json` | Combined export for LaTeX tables |
| `fig2_heatwave.png/pdf` through `fig10_*.png/pdf` | Publication figures |
| `artifact_manifest.json` | SHA-256 hashes for reproducibility |

---

## Troubleshooting

**Module not found:** Run `module avail` to see what's installed. If Python isn't available, email `SDSU.HPC@sdstate.edu`.

**Job stuck in queue:** The `compute` partition has a 14-day limit but may be busy. Try `quickq` (12-hour limit, usually faster to schedule):
```bash
# Edit the .slurm files and change:
#SBATCH --partition=compute
# to:
#SBATCH --partition=quickq
```

**Permission denied on submit_all.sh:**
```bash
chmod +x submit_all.sh
bash submit_all.sh
```
