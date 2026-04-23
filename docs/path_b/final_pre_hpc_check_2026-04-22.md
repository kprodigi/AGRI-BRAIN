# Final Pre-HPC Check

Run by: Claude, 2026-04-22
Repo HEAD: `b30f924` (on `main`, matches `origin/main`)

No code execution of the simulator itself; verification by static reads,
targeted Python imports, bash -n syntax checks, and a dry-run of the
aggregator on the 5 legacy seed JSONs.

## Final verdict: **GO**

No Category A items remain; no new issues found. Safe to submit
`bash hpc_run.sh` from the HPC login node.

## Verification table

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Git tree clean | PASS | `git status` clean, no untracked files |
| 2 | `HEAD == origin/main` | PASS | both at `b30f924` |
| 3 | Commits pushed | PASS | 9 commits on top of the baseline, all pushed |
| 4 | Default pytest | PASS | 89 passed, 16 deselected (slow), 14.66 s |
| 5 | Fast consistency tests | PASS | 4/4 in 0.94 s |
| 6 | Slow smoke test sanity | PASS | `test_no_yield_under_cyber_outage_runs` → PASS in 17.95 s |
| 7 | `MODES` count | PASS | 9 modes (canonical + no_yield) |
| 8 | `VALID_MODES ⊇ MODES` | PASS | empty missing set |
| 9 | `CYBER_REROUTE_PROB ⊇ MODES \ {static}` | PASS | empty missing set |
| 10 | `_CONTEXT_ENABLED_MODES` | PASS | `{agribrain, mcp_only, pirag_only, no_yield}` |
| 11 | `_AGRIBRAIN_LOGIT_MODES` | PASS | 5 modes |
| 12 | `_PINN_MODES` includes no_yield | PASS | yes |
| 13 | `_MCP_WASTE_MODES` | PASS | `{agribrain, mcp_only, no_yield}` |
| 14 | `THETA.shape` | PASS | `(3, 6)` |
| 15 | `THETA_CONTEXT.shape` | PASS | `(3, 6)` |
| 16 | `MODIFIER_RULES` length | PASS | 6 |
| 17 | ψ_5 sign convention | PASS | `THETA_CONTEXT[:, 5] = [+0.20, +0.05, -0.15]` |
| 18 | MCP/piRAG masks length 6 | PASS | both 6-element |
| 19 | `_MCP_FEATURE_MASK[5] == 1.0` | PASS | yes |
| 20 | `_PIRAG_FEATURE_MASK[5] == 0.0` | PASS | yes |
| 21 | Static registry count | PASS | 13 tools |
| 22 | Runtime registry count | PASS | 18 tools |
| 23 | `yield_query` in processor/coop/dist workflows | PASS | all three |
| 24 | `yield_query` NOT in farm/recovery workflows | PASS | correctly excluded |
| 25 | `recovery_capacity_check` in distributor | PASS | still wired |
| 26 | Path B identity `full − no_yield == Θ[:,5] · u` | PASS | `max_err = 0.0` |
| 27 | Cyber outage Bernoulli probs | PASS | `[0.18, 0.82, 0.0]` for agribrain |
| 28 | `hpc_run.sh` syntax | PASS | bash -n clean |
| 29 | `hpc_seed.sh` syntax | PASS | bash -n clean |
| 30 | `hpc_aggregate.sh` syntax | PASS | bash -n clean |
| 31 | RUN_TAG threaded through all 3 scripts | PASS | 7 + 6 + 5 references |
| 32 | hpc_seed.sh array size `--array=0-19` | PASS | 20 seeds |
| 33 | Path B load assertion from hpc_seed.sh | PASS | `yield_query` present, Θ shape `(3,6)` |
| 34 | Path B load assertion from hpc_aggregate.sh | PASS | identical and executes |
| 35 | `run_single_seed.py --help` | PASS | `--output-dir` flag present |
| 36 | `run_benchmark_suite.py --help` | PASS | `--seeds-dir` and `--output-dir` flags present |
| 37 | `aggregate_seeds.py` imports | PASS | `holm_bonferroni`, `benjamini_hochberg` both exported |
| 38 | All aggregator stages present | PASS | generate_results, validate, stress, figures, paper evidence, manifest |
| 39 | Aggregator dry-run (5 seeds) | PASS | `benchmark_summary.json` has `n_boot=10000`, `n_perm=10000` |
| 40 | Primary H1 Holm-Bonferroni | PASS | `primary_h1_holm_adjusted` present; `correction_method = holm_bonferroni_across_scenarios` |
| 41 | Secondary BH-FDR per scenario | PASS | `correction_method = bh_fdr_within_scenario`; distinct `p_value_adj_bh` field populated |
| 42 | Primary endpoint uses Holm | PASS | `agribrain_vs_no_context` ARI records have `p_value_adj = p_value_adj_holm` |

42 checks, all PASS.

## Wall-time projection (unchanged from prior check)

- Per-seed serial on HPC (5× laptop speedup assumed): ~2 h / seed.
- SLURM array `--time=06:00:00` per task → 3× margin per seed.
- 20 tasks in parallel → total array wall time ≈ 2-4 h (queue-dependent).
- Aggregator SLURM `--time=08:00:00`:
  - Stage 1 `generate_results.py` ~2 h (single run, 45 cells).
  - Stage 6 stress suite ~30 min.
  - `aggregate_seeds.py` with 10,000 resamples × 10,000 permutations ≈ 5 min.
  - `run_benchmark_suite.py` aggregator ≈ 1 min.
  - Figures + paper evidence + manifest + final validation ≈ 30 min total.
  - Aggregator total ≈ 3-4 h; ~4 h margin.
- End-to-end wall time from `bash hpc_run.sh` to archive tarball:
  approximately 6-10 h with queue waits, inside both SLURM budgets.

## Commit chain since the original pre-HPC audit

```
b30f924 record deep review report and paper snapshot at afccd7f
03bd4bd refresh post-path-b docstrings, extract retrieval guard, add mode-coverage tests
3eb6aa2 align multiplicity correction and resample counts with paper section 3.13
afccd7f wire no_yield through pinn spoilage, mcp waste penalty, and api types
59dbc1c report policy bernoulli under cyber outage and register no_yield mode
ca8f588 include no_yield in canonical aggregator and harden hpc packaging
14798d0 address pre-hpc warns and confirm go status
99a3979 convert hpc pipeline to slurm job array with separate aggregation
ac162e7 deduplicate stage 3 by aggregating per-seed json
30e3298 log path b implementation status and defer benchmark to hpc (baseline)
```

Nine commits on top of the Path B baseline. Every correctness concern
raised in the deep review is addressed; every remaining item in the deep
review is a manuscript-side edit (Category B) that depends on the HPC
numbers.

## Submission command

On the HPC login node, from the repo root:

```bash
bash hpc_run.sh
```

This will:

1. Create `.venv` if absent; `pip install -e agri-brain-mvp-1.0.0/backend`.
2. Run the login-node Path B load assertion.
3. Compute `RUN_TAG=$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M)`.
4. Submit `hpc_seed.sh` as a 20-task array.
5. Submit `hpc_aggregate.sh` with `--dependency=afterok:<seed_job>`.

Expected archive: `hpc_results_<RUN_TAG>.tar.gz` in the repo root after
both jobs complete. Transfer locally with
`scp <hpc-host>:$PWD/hpc_results_<RUN_TAG>.tar.gz .`, then
`tar xzf` into the results tree.

## After the HPC run

Three manuscript-side tasks remain (all Category B from the deep review):

1. Update paper to reflect ψ ∈ ℝ⁶ and Θ_context ∈ ℝ^(3×6) (equations,
   Table 3, Figure 3, Algorithm 1).
2. Update paper to reflect 9 modes × 5 scenarios × 20 seeds = 900 episodes
   and add the no_yield row to Table 4.
3. Update paper's cyber-outage narrative in Section 4.4 and Figure 6c to
   reflect the honest Bernoulli policy distribution (no more "entropy
   collapses to 0" rationalisation).

These are word processing tasks, not code changes, and can be done with
the HPC-generated table numbers in hand.
