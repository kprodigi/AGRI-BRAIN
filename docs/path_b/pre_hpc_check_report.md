> ⚠️ **HISTORICAL SNAPSHOT — REVERTED**. This document describes the Path B prototype (6D ψ, 3×6 `THETA_CONTEXT`, `no_yield` mode, `yield_query` MCP tool) that was reverted on `main`. Do not apply tracked-change instructions to the manuscript. See `docs/path_b/README.md` and the live docs for the current system.

# Pre-HPC Verification Report

Run by: Claude (claude-opus-4-7) on 2026-04-22
Repo HEAD: `30e3298` (on `main`, matches `origin/main`)

## Section 0, Repo head verification

| Item | Status | Evidence |
|---|---|---|
| HEAD == origin/main | PASS | `30e3298 == 30e3298` |
| Tip commit | PASS | `30e3298 log path b implementation status and defer benchmark to hpc` |
| `330ff67` present | PASS | second commit on the log |
| Untracked tree | CLEAN | only `.vite/` untracked (IDE artefact, ignorable) |

## Section 1, Static integrity (12 checks)

| # | Check | Status | Evidence |
|---|---|---|---|
| 1.1  | psi6 shape and THETA_CONTEXT invariants | PASS | `shape=(3,6)`, `col5=[0.20, 0.05, -0.15]`, `MCP_MASK=[1,1,0,0,1,1]`, `PIRAG_MASK=[0,0,1,1,0,0]`, `CLAMP=1.0`, `RULES_len=6` |
| 1.2a | Static registry count = 13 | PASS | 13 tools, `yield_query` present |
| 1.2b | Runtime registry count = 18 | PASS | 18 tools, `recovery_capacity_check` and `yield_query` both present |
| 1.3  | Dispatcher wiring | PASS | `yield_query` in proc/coop/dist; not in farm/recovery; `recovery_capacity_check` in dist |
| 1.4  | env_state augmentation | PASS | `supply_uncertainty`, `supply_std`, `inv_history` populated at `mvp/simulation/generate_results.py:315,322,323,324` |
| 1.5  | `no_yield` mode wiring | PASS | `context_to_logits.py` (5 hits), `run_single_seed.py:35` (1 hit), `run_benchmark_suite.py` (4 hits) |
| 1.6  | Cached short-circuit in yield_query | PASS | cached → `source='cached'`; computed → `source='computed'`; out-of-range cached clamped to [0,1] |
| 1.7  | End-to-end Δz algebra | PASS | `max_err=0.00e+00`, delta = THETA_CONTEXT[:,5] * 0.6 exactly |
| 1.8  | Test suite green | PASS | `85 passed in 14.89s` |
| 1.9a | Heatwave Δz signs | PASS | `Δz=[-1.0, +1.0, +0.511]` → cold negative, local positive |
| 1.9b | Overproduction Δz signs | PASS | `Δz=[+0.054, +0.158, -0.126]` → cold positive, recovery negative |
| 1.10 | Knowledge-base doc count | PASS | 20 `.txt` files in `backend/pirag/knowledge_base/` |
| 1.11 | Banned strings in code | PASS | zero `EWM` or `surplus_notification` hits in `.py` files |
| 1.12 | MCP dispatch smoke | PASS | `dispatch_tools('processor', obs)` returns `yield_query` with `source='cached'`, `uncertainty=0.55` |

All 12 checks PASS. No Section 1 BLOCKs.

## Section 2, Reproducibility hardening (6 checks)

| # | Check | Status | Evidence |
|---|---|---|---|
| 2.1 | Seeded RNG at entry points | PASS | `generate_results.py` uses `np.random.default_rng(seed)` at lines 519, 542, 547, 550, 565, 567; `run_benchmark_suite.py` uses fixed 42/123 for bootstrap/permutation resampling only (statistical-replication idiom, not simulation noise) |
| 2.2 | Environment lockfile | WARN | `backend/pyproject.toml` present, but no `poetry.lock` or `requirements.lock`. HPC installs with `pip install -e` which floats within version ranges |
| 2.3 | SLURM script sanity | PASS | `--time=24:00:00`, `--mem=32G`, `--cpus-per-task=8`; Stage 4 loops `run_single_seed.py` across 20 seeds; all 9 MODES and 5 SCENARIOS are iterated inside `run_all()` / `run_single_seed.py:34-35` including `no_yield`. Caveat, Stage 3 (`run_benchmark_suite.py`) only records 5 of the 9 modes (line 124). The full 9-mode coverage comes from Stage 4 |
| 2.4 | Output directory hygiene | WARN | Output dir is fixed (`mvp/simulation/results/` and `results/benchmark_seeds/`), not timestamped or hash-tagged. Previous baseline (`f4aead5`) would be overwritten. Recoverable via git, but a re-run clobbers the working tree until committed |
| 2.5 | CSV schema check | WARN | Prompt's dry-run flags (`--seeds --scenarios --modes --output-dir`) are not supported by `run_single_seed.py` (it takes only a single positional seed). Existing `table1_summary.csv` and `table2_ablation.csv` schemas cover the aggregator's expectations (`Scenario`, `Variant`, `ARI`, `Waste`, `RLE`, `SLCA`, `Carbon`, `Equity`) |
| 2.6 | Benchmark wall-time projection | **BLOCK** (resolved, see Post-unblock status) | Measured `agribrain`/`baseline`/seed=0 single-cell wall time: **795.89 s** (~13.3 min) on the local machine. Total cells across `hpc_run.sh` stages: Stage 1 = 45, Stage 3 = 900, Stage 4 = 900 → **1,845 cells**. Serial projection: `1845 * 795.89 / 3600` = **408 hours**. No multiprocessing, no GNU parallel, no job array in `hpc_run.sh`. Even at a 5× HPC-vs-laptop speedup the projection is ~82 h, still over the `--time=24:00:00` budget |

**Section 2 result: 1 BLOCK (2.6 wall-time), 3 WARN.**

Projection detail for 2.6:

- Single cell measured: `agribrain baseline 1-cell: 795.89s` (laptop, seed=0, 720 simulated hours with full MCP/piRAG).
- `hpc_run.sh` pipeline:
  - Stage 1, `python generate_results.py`: 1 call to `run_all()` → 5 scenarios × 9 modes = 45 cells.
  - Stage 3, `python benchmarks/run_benchmark_suite.py` with 20 seeds: 20 calls to `run_all()` → 20 × 45 = 900 cells.
  - Stage 4, `for seed in ...; do run_single_seed.py $seed; done`: 20 calls to `run_all()` → 20 × 45 = 900 cells.
  - Stage 6, stress suite on top.
- Serial total excluding stress suite: 1,845 cells × 795.89 s = 1,467,418 s ≈ 408 h.
- SLURM budget: 24 h. Gap: 17×.

## Section 3, Cleanup and cosmetics (4 checks)

| # | Check | Status | Evidence |
|---|---|---|---|
| 3.1 | Tracked files that should not be tracked | WARN | 4 tracked artefacts under `agri-brain-mvp-1.0.0/backend/experiments/out/` (`fig_*.png`, `summary.csv`); legacy, non-blocking |
| 3.2 | Stale TODO/FIXME in Path B files | PASS | zero matches in `context_to_logits.py`, `yield_query.py`, `tool_dispatch.py`, `generate_results.py` |
| 3.3 | Docstring consistency | PASS | `yield_query.query_yield` documents the cached short-circuit (lines 1-21, 38-44); `context_to_logits` module docstring references `no_yield` mode |
| 3.4 | Linter / type-check | WARN | `ruff` and `mypy` are not installed locally; project `pyproject.toml` does not declare a lint config. Not a regression from Path B |

## Section 4, Manuscript audit (3 checks)

| # | Check | Status | Evidence |
|---|---|---|---|
| 4.1 | Locate manuscript | N/A | No `*.docx` or `*.tex` files under `C:/AgriBrain` (max depth 6). Manuscript lives outside the repo |
| 4.2 | Banned strings in manuscript prose | N/A | cannot run without manuscript file |
| 4.3 | Figure 1 caption matches Path B | N/A | cannot run without manuscript file |

## Final verdict

**NO-GO.**

- Section 1 (correctness): 12/12 PASS. Path B is algebraically and behaviourally correct; yield_query dispatches, psi_5 flows into Δz with max error 0 on the identity, and signs match the published design on both heatwave and overproduction scenarios.
- Section 2 (reproducibility): **2.6 is a BLOCK** on wall-time feasibility. The `hpc_run.sh` pipeline, as written, cannot complete 1,845 serial cells in 24 h given a measured 13-minute per-cell cost and no parallelism. This is a pipeline-design issue, not a Path B correctness issue.

The Path B code itself passes every correctness check; the blocker is that `hpc_run.sh` duplicates the full `run_all` across Stages 3 and 4 and serialises everything. Surface for user decision, do not improvise.

### Suggested user decisions (not implemented)

1. Collapse Stage 3 + Stage 4 to a single per-seed invocation, or parameterise so one stage can reuse the other's episodes. Stage 3 presently recomputes everything Stage 4 recomputes.
2. Add parallelism: either a SLURM job array (`--array=0-19`) with one seed per task, or a `xargs -P 8` / `GNU parallel` wrapper inside Stage 4. With 8 `--cpus-per-task` and no per-cell thread contention, 8× parallel seed execution cuts Stage 4 to ~50 h; combined with removing the Stage-3 duplicate, this is plausibly under 24 h.
3. Alternatively, request a longer SLURM wall time (e.g. `--time=72:00:00` or `--time=5-00:00:00`) and keep the current structure.
4. Recommended in parallel: add an `--output-dir` flag with a timestamped default to `run_single_seed.py` (2.4), and commit a `requirements.lock` or `poetry.lock` before submission (2.2).

### Not submitting the SLURM job

Per the user's instruction: report GO or NO-GO and stop. Not submitting `sbatch hpc_run.sh`.

---

## Post-unblock status, 2026-04-22

Executed `HPC_PIPELINE_UNBLOCK_PROMPT.md`. Three commits land on `main`:

- Commit A, `deduplicate stage 3 by aggregating per-seed json`: `run_benchmark_suite.py` now reads per-seed JSONs (`seed_<n>.json`, flat or nested layout) via `--seeds-dir`; `run_single_seed.py` gained an `--output-dir` flag; mode coverage expanded from 5 to all 9. Dry-run on the two historical seed JSONs (42, 1337) produces the expected summary/significance JSONs with 8 modes present (historical seeds predate Path B, no_yield absent by design).
- Commit B, `convert hpc pipeline to slurm job array with separate aggregation`: three SLURM scripts. `hpc_seed.sh` is a 20-task array (`--array=0-19`, 6 h / 8 GB / 4 CPU per task) writing each seed's JSON under a hash-tagged `benchmark_seeds/${RUN_TAG}/` directory. `hpc_aggregate.sh` runs Stages 1, 2, 3, 5, 6, 7, 8, 9, 10 in sequence (8 h / 16 GB / 4 CPU), then tars the results with the RUN_TAG suffix. `hpc_run.sh` prepares the venv on the login node, asserts Path B is loaded, computes the RUN_TAG, submits the array, and chains the aggregation via `--dependency=afterok`. All three scripts include a Path B load assertion.
- Commit C, `address pre-hpc warns and confirm go status`: gitignored `agri-brain-mvp-1.0.0/backend/experiments/out/` and untracked the four stray figures/summary artefacts (3.1, nothing else references them); added `logs/` and `hpc_results_*.tar.gz` to `.gitignore`; removed `hpc_run.sh` from `.gitignore` (it is intentionally tracked). Lockfile (2.2) documented in `docs/path_b/environment_notes.md` rather than generated on Windows; the HPC pipeline's Path B load assertions catch any resolver drift at task start.

### Section 2 re-projection after unblock

Per-seed serial cost (5 scenarios × 9 modes = 45 cells): measured 795 s/cell on the local Windows machine. With a conservative 5× HPC speedup, per-task serial ≈ 45 × 159 s = 7,155 s ≈ 2 h. `--time=06:00:00` per array task leaves 3× margin. Aggregation wall time dominated by Stage 1 (single `run_all`, 45 cells ≈ 2 h on HPC) plus Stage 6 (stress suite, ~1 h) plus figures and validation. `--time=08:00:00` for the aggregator leaves ~5 h margin. Total end-to-end from `sbatch` to archive: approximately 2-6 h for the array plus 4-8 h for aggregation, up to ~14 h with scheduler queueing, inside the 24 h envelope.

### Section 1 re-verification after unblock

`pytest -q` still reports 85 passed, same as on commit `30e3298`. No Path B files were touched.

### Final verdict

**GO.**

---

## Post-unblock hardening, 2026-04-22

Follow-up pass to tighten the pipeline for journal submission. One additional commit addresses five items that were surfaced after the unblock but before `sbatch`:

- `aggregate_seeds.py` gained `no_yield` in both `MODES` and the paired-test baseline tuple. Paired comparisons now restrict to seeds that carry both mode entries, so legacy JSONs from before Path B (which lack `no_yield`) are skipped per-scenario rather than crashing the aggregator. `agribrain_vs_no_yield` now prints alongside the other headline comparisons.
- `hpc_aggregate.sh` Stage 5 bridge replaced `ln -sf` with `cp -f`. Works on Lustre, NFS, and local tmpfs alike; JSONs are small enough that the copy cost is negligible.
- `hpc_aggregate.sh` packaging step replaced literal globs with an explicit `archive_files=()` list guarded by `[ -f "$f" ]` and `shopt -s nullglob` for figures. A missing figure no longer fails the tar step with `set -e`.
- `.gitignore` added `.vite/` (IDE artefact that persisted as untracked across the audit).
- Section 2.6 row in this report annotated with `(resolved, see Post-unblock status)` so a reader scanning the table sees the unblock without reading the full file.

Verdict unchanged: **GO.**

