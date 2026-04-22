# Path B — Implementation Log (2026-04-22)

Successor to `sanity_report.md`. Records what landed, what was deferred, and why the benchmark remains open.

---

## What landed (commit `330ff67`)

### Code (committed on `main`)

| File | Change |
|---|---|
| `agri-brain-mvp-1.0.0/backend/pirag/mcp/tools/yield_query.py` | New. MCP tool wrapping the existing `yield_supply_forecast`. Short-circuits when pre-computed uncertainty is in `obs.raw`, else computes CV from `inventory_history`. |
| `agri-brain-mvp-1.0.0/backend/pirag/context_to_logits.py` | ψ extended 5 → 6. `THETA_CONTEXT` grown to 3×6 with ψ₅ column [+0.20, +0.05, −0.15]. Added `context_mode="no_yield"` ablation. Masks extended. |
| `agri-brain-mvp-1.0.0/backend/pirag/mcp/registry.py` | `yield_query` registered in `get_default_registry()`. |
| `agri-brain-mvp-1.0.0/backend/pirag/mcp/tool_dispatch.py` | `_yield_query_args` added. `yield_query` appended to PROCESSOR, COOPERATIVE, DISTRIBUTOR workflows. `recovery_capacity_check` **kept** in DISTRIBUTOR (it is late-bound, not a ghost). |
| `agri-brain-mvp-1.0.0/backend/pirag/trace_exporter.py` | Two `np.zeros(5)` accumulators bumped to `np.zeros(6)` so traces with ψ⁶ context_features don't broadcast-fail. |
| `agri-brain-mvp-1.0.0/backend/pirag/tests/test_mcp_pirag_integration.py` | Existing 5-vector `context_features` literals and `psi.shape == (5,)` / `len == 5` assertions updated to 6. |
| `agri-brain-mvp-1.0.0/backend/src/agents/coordinator.py` | `_CONTEXT_MODES` and `_CONTEXT_MODE_MAP` gain `no_yield → "no_yield"`. |
| `agri-brain-mvp-1.0.0/backend/tests/test_path_b_integration.py` | New. 20 tests across yield_query, ψ⁶ extraction, 3×6 REINFORCE learner. |
| `mvp/simulation/generate_results.py` | `env_state` now carries `supply_uncertainty`, `supply_std`, `inv_history`. `MODES` gains `"no_yield"`. |
| `mvp/simulation/benchmarks/run_single_seed.py` | 8-mode tuple extended to 9 (adds `"no_yield"`). |
| `mvp/simulation/benchmarks/run_benchmark_suite.py` | Three 4-mode references extended to 5 (adds `"no_yield"`). |

### Tests

- `pytest -q`: **85 passed, 0 failed** (was 65 pre-Path-B; 20 new tests; 0 regressions after the ψ⁶-shape updates documented above).
- Registry: static count 12 → 13; runtime count 17 → 18.
- End-to-end smoke verified: `dispatch_tools("distributor", obs, registry)` invokes `yield_query` (short-circuit path), which populates ψ₅ = 0.6, which multiplies through to `Δz(full) − Δz(no_yield) = [+0.12, +0.03, −0.09] = Θ[:, 5] × 0.6` exactly.

### Self-defence checklist (from Section I)

- [x] `static_count == 13`, `runtime_count == 18`.
- [x] `extract_context_features(...).shape == (6,)` (test passes).
- [x] `THETA_CONTEXT.shape == (3, 6)` (test passes).
- [x] `THETA_CONTEXT[0,5] > 0`, `[1,5] > 0`, `[2,5] < 0` (test passes).
- [x] `_MCP_FEATURE_MASK[5] == 1.0`, `_PIRAG_FEATURE_MASK[5] == 0.0` (tests pass).
- [x] `no_yield` mode produces a Δz different from `full` when ψ₅ ≠ 0 (test passes; numerical verification above).
- [x] `ContextMatrixLearner` accepts 3×6 init and preserves signs under adversarial rewards (test passes).
- [x] `recovery_capacity_check` still in `DISTRIBUTOR_WORKFLOW`.
- [x] `mvp/simulation/generate_results.py` populates `env_state["supply_uncertainty"]` and `env_state["inv_history"]`.

---

## What did not land

### Section D (Phase 1 smoke benchmark) — BLOCKED

A single `run_episode` for one scenario × one mode (288 timesteps on `data_spinach.csv`) did not complete in 90 seconds on this laptop. Extrapolating:

- 1 seed = 9 modes × 5 scenarios ≈ **45 episodes** per seed.
- At ≥ 90 s/episode, 1 seed ≈ **≥ 68 minutes**.
- Phase 2 (20 seeds) ≈ **≥ 22 hours** serially.

This matches the intent of `hpc_run.sh` in the repo root, which allocates 24 h of SLURM wall time and 32 GB memory. The benchmark is not a laptop workload; it is an HPC job.

**Recommended next step for the user:**
```bash
# On HPC
sbatch hpc_run.sh
# Transfer results back
scp <hpc-host>:~/agribrain-hpc-run/hpc_results.tar.gz .
tar xzf hpc_results.tar.gz
# Then compare against the Apr 5 baseline in mvp/simulation/results/benchmark_seeds/
```

Baseline seeds `7, 42, 99, 1337, 2024` already sit on disk (dated 2026-04-05) for pre/post Path B comparison.

### Section E (Phase 2/3 full benchmark) — BLOCKED

Depends on D.

### Section F (manuscript updates) — DRAFT READY, NOT APPLIED

Tracked-changes copy ready at `docs/path_b/manuscript_updates.md`. Apply to the latest manuscript docx (likely `C:\Users\Nahid\Downloads\AGRIBRAIN_edited_2.docx`) after Tables 5/6/7/8 numbers are available from Phase 2/3. Figure replacement uses `C:\Users\Nahid\Downloads\Figure1_AGRI_BRAIN.pptx` (re-insert, do not edit in Word).

The F.4.a honesty fix for the routing-path guard claim is independent of benchmark numbers and can be applied now.

### Section G.1 (three-guard unification) — DEFERRED

Reasons:
1. The prompt marks G.1 as optional ("apply unless Phase 4 sanity is already failing"). Phase 4 has not run, so there is no sanity signal to gate on.
2. The existing `context_builder.py:296-302` comment documents a specific reason for bypassing the unit guard: the `units_consistent` primitive false-positives on the synthesised-answer template preamble ("Based on N relevant sources…"). A naive G.1 implementation would reintroduce the false positive.
3. Test impact is non-trivial and there is no benchmark to verify against this session.

**Recommended safer version** for a follow-up session: apply the three guards to citation *passages* (raw KB text) rather than the synthesised answer, so the template preamble is not in the guarded text. Keep the current retrieval-quality flag as `retrieval_ok` and AND it with `units_ok`; keep `verify_with_sim` optional (can return `None` in offline mode).

### Section H (final commits/PR) — PARTIAL

`330ff67` is the core commit. Remaining commits (benchmark CSVs, manuscript docx, paper figure replacement) happen after Sections D–F complete.

---

## Memory updates the user should authorise

- Test baseline: was 59, now **85** passing.
- Static MCP tool count: **13** (was 12).
- Runtime MCP tool count: **18** (was 17).

---

## Open questions for the user

1. Run the 20-seed benchmark on HPC, or accept Path B ship without a fresh benchmark and paper Tables 5/6/7 staying at the pre-Path-B values? (Strong recommendation: run on HPC. The paper claims reproducibility and Path B changes the routing policy nontrivially.)
2. Apply F.4.a (routing-path guard honesty fix) now in the manuscript, even before F.6 numbers land? (Recommend: yes. It fixes an inaccurate claim and is independent of the benchmark.)
3. G.1 deferral OK? (Recommend: defer. The existing comment in `context_builder.py` documents a real hazard that the naive G.1 would reintroduce.)
