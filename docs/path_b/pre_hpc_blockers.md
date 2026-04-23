# Pre-HPC Blockers

> **HISTORICAL SNAPSHOT.** This document describes blockers as of the
> listed repo HEAD. It is kept for traceability only. Later work in
> `hpc_run.sh`, `hpc_seed.sh`, and `hpc_aggregate.sh` introduced a
> SLURM array + dependent aggregate stage, which resolved the
> parallelism concerns recorded here. For the current pipeline see the
> root ``HOW_TO_RUN.md`` Section 9 (HPC).

Recorded: 2026-04-22
Repo HEAD: `30e3298`

## BLOCK-1, Section 2.6, SLURM wall-time projection exceeds budget

**Where**: `hpc_run.sh` pipeline, Stages 1 + 3 + 4.

**Measured**: single-cell wall time on local machine (Windows/Python 3.13, seed=0, scenario=baseline, mode=agribrain): **795.89 s** (~13.3 min).

**Projected serial cost** (not counting Stage 6 stress suite):

| Stage | Script | Cells | Serial time |
|---|---|---|---|
| 1 | `generate_results.py` | 5 × 9 = 45 | 35,815 s (~10 h) |
| 3 | `run_benchmark_suite.py` (20 seeds × `run_all()`) | 20 × 45 = 900 | 716,301 s (~199 h) |
| 4 | `run_single_seed.py` × 20 seeds | 20 × 45 = 900 | 716,301 s (~199 h) |
| **Total** | | **1,845** | **~408 h** |

**SLURM budget**: `--time=24:00:00` = 86,400 s. Gap: ~17× over budget.

**Parallelism**: none. `hpc_run.sh` has no job array, no `xargs -P`, no GNU parallel, and the Python scripts use no `multiprocessing` / `concurrent.futures`. `--cpus-per-task=8` only helps via numpy/BLAS thread pools on linalg kernels, which do not dominate this workload.

**Why this is a BLOCK**: even granting a 5× HPC-vs-laptop speedup (generous for single-threaded Python), ~82 h serial still overshoots 24 h. Submitting `sbatch hpc_run.sh` would result in the job being killed at the 24 h wall-time limit with Stage 4 incomplete.

**Architectural note**: Stage 3 and Stage 4 both call `run_all(seed)` per seed, so they duplicate each other fully. One of the two stages can be removed without changing coverage.

**Action requested from user**: this is a pipeline-design decision, not a clerical fix. The three options that unblock:

1. Deduplicate Stage 3 and Stage 4. If `run_benchmark_suite.py` can ingest per-seed JSONs from `benchmark_seeds/` instead of re-running `run_all()`, Stage 3 becomes an aggregator and the serial cost roughly halves.
2. Add parallelism, either SLURM `--array=0-19` (one seed per task, same budget, runs in parallel), or `xargs -P 8` inside the Stage 4 loop.
3. Increase `--time` on the SLURM header to match the projected wall time (72 h or more).

Do not proceed to `sbatch` until one of the above is applied.
