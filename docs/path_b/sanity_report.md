> ⚠️ **HISTORICAL SNAPSHOT — REVERTED**. This document describes the Path B prototype (6D ψ, 3×6 `THETA_CONTEXT`, `no_yield` mode, `yield_query` MCP tool) that was reverted on `main`. Do not apply tracked-change instructions to the manuscript. See `docs/path_b/README.md` and the live docs for the current system.

# Path B — Section 2 Sanity Report

Executed by Claude Code on 2026-04-22, before applying any Path B patch.

**Verdict: STOP — five blocking divergences. Do not proceed to Section 3 without user sign-off.**

---

## Input availability (Section 1 of master prompt)

| Expected input | Actual | Status |
|---|---|---|
| `path_b_patches/yield_query.py` | Not present anywhere on disk | **MISSING** |
| `path_b_patches/context_to_logits.py` | Not present | **MISSING** |
| `path_b_patches/registry_and_dispatch_patches.md` | Not present | **MISSING** |
| `path_b_patches/test_path_b_integration.py` | Not present | **MISSING** |
| `path_b_implementation_guide.md` | `C:\Users\Nahid\Downloads\path_b_implementation_guide.md` | OK (reference only) |
| `Figure1_AGRI_BRAIN.pptx` | `C:\Users\Nahid\Downloads\Figure1_AGRI_BRAIN.pptx` | OK |

Without the four missing files we cannot execute Section 3 (drop-in / replacement / patch / tests). The implementation guide describes them but does not contain them. **User action required: locate and provide the four files, or ask Claude to synthesise them from the guide — the latter is higher-risk and warrants review.**

---

## Section 2 results

| Check | Expected | Actual | Status |
|---|---|---|---|
| 2.1 Default registry count | 12 | 12 (all expected names) | OK |
| 2.2 Tool signatures match dispatcher | All 7 dispatched tools line up | All match; `simulate` returns None on degraded mode (not dispatched) | OK |
| 2.3 KB doc count | 20 | 20 `.txt`, 0 `.md` | OK |
| 2.4 piRAG guards | 3 guards feeding `guards_passed` | **3 guards in `PiRAGPipeline.ask` only; routing path uses a separate retrieval-quality check** | **DIVERGENCE** |
| 2.5 `env_state` construction site | `run_experiments.py`, `app.py`, and simulator | **Only `mvp/simulation/generate_results.py:311`** | **DIVERGENCE** |
| 2.6 Ghost/orphan audit | `recovery_capacity_check` = ghost, remove it | **Not a ghost — registered at runtime by `register_all_agent_capabilities`** | **DIVERGENCE (patch would break code)** |
| 2.7 Dynamic KB cycle | 6 h / 24 steps | 24 steps / 6 h at 15-min resolution, event-driven | OK |
| 2.8 Solidity contracts | 6 by name, each matching implied role | All 6 match | OK |
| 2.9 Test baseline | 59 passing | **65 passing, 0 failing, 14.74 s** | OK (memory note stale) |
| 2.10 End-to-end trace | All 6 arrows wired | All 6 arrows wired in `coordinator.step` → `_compute_step_context` | OK |

---

## Blocking divergences (detail)

### D1. Path B patch files are missing (Section 1)

- **What the master prompt assumed:** patch package at `/mnt/user-data/outputs/path_b_patches/` with four files.
- **What is actually on disk:** only the implementation guide and the PPTX. No `yield_query.py`, no replacement `context_to_logits.py`, no `registry_and_dispatch_patches.md`, no `test_path_b_integration.py`. `/mnt/user-data/outputs/` contains unrelated SINDy figures from another project.
- **Minimal revised proposal:** ask the user to drop the four files into `C:\Users\Nahid\Downloads\path_b_patches\`, or authorise Claude to synthesise them from the guide (not recommended without review — the guide omits implementation details the test file would pin down).

### D2. `recovery_capacity_check` is not a ghost (2.6)

- **What the master prompt assumed (§3.3 / §9):** `recovery_capacity_check` is referenced by `DISTRIBUTOR_WORKFLOW` but never registered, so the step is silently skipped. Path B instructs to remove the entry.
- **What the code actually shows:**
  - Registered at runtime by [`agent_capabilities.py:70`](agri-brain-mvp-1.0.0/backend/pirag/mcp/agent_capabilities.py:70) via `register_recovery_capabilities`.
  - The coordinator wires this in on init: [`coordinator.py:137`](agri-brain-mvp-1.0.0/backend/src/agents/coordinator.py:137) — `register_all_agent_capabilities(self._mcp_server, self.agents)`.
  - With the coordinator fully initialised the registry has **17 tools, 0 ghosts, 9 orphans** (orphans are protocol-level / introspection tools, not ghosts).
  - Exercised by the test `test_advanced_features.py:62`.
- **Why the master prompt got it wrong:** the 2.1 smoke test calls `get_default_registry()` directly, which only returns the 12 statically-registered tools and misses the 5 agent-capability tools added on coordinator init.
- **Minimal revised proposal:** **do not remove** the `recovery_capacity_check` entry from `DISTRIBUTOR_WORKFLOW`. Path B can still add `yield_query` alongside it. Update §3.3 / §9 of the implementation guide, and update manuscript §3.6 to describe all 17 registered tools (or the 13 protocol-level ones plus 5 agent-capability tools as a separate group), not "13".

### D3. `env_state` construction lives in `mvp/simulation/`, not `backend/experiments/` (2.5)

- **What the master prompt assumed (§2.5):** `env_state` is built in `backend/experiments/run_experiments.py` and `backend/src/app.py`; add a `_inv_history_buffer` with a per-episode reset.
- **What the code actually shows:**
  - `backend/experiments/run_experiments.py` is a pandas-based perturbation simulator. It never constructs `env_state`, never calls `coordinator.step`, and never invokes the MCP dispatcher. It cannot exercise Path B's ψ₅ path.
  - `backend/src/app.py::/decide` builds its observation from a DataFrame row directly and calls `select_action` without ever touching `env_state` or the coordinator.
  - The **actual** 20-seed benchmark entrypoint is [`mvp/simulation/generate_results.py::run_all`](mvp/simulation/generate_results.py), which constructs `env_state` at line 311 and drives `coordinator.step` at line 331. This is invoked by `mvp/simulation/benchmarks/run_single_seed.py` and `run_benchmark_suite.py`.
  - The benchmark already computes `yield_supply_forecast(hist_slice, horizon=1, ...)` at line 295 and stores `supply_hat` in `env_state["supply_hat"]` at line 314 — so Holt-Winters is already wired and producing a forecast per step; it's just not surfaced as an MCP tool.
  - A `hist_slice = df.iloc[max(0, idx + 1 - lookback):idx + 1]` exists at line 288, giving a 48-step inventory history window for free.
- **Minimal revised proposal:** one-line change in `generate_results.py` right before `coordinator.step`:
  ```python
  env_state["inv_history"] = hist_slice["inventory_units"].astype(float).tolist()
  ```
  No `_inv_history_buffer` global, no per-episode reset (automatic: `run_all` reloads `df` per seed/scenario). Drop the buffer-maintenance text from §3.4 / §2.5 of the master prompt.

### D4. Holt-Winters already exists as `yield_supply_forecast` (bonus finding)

- Not called out in the master prompt, but material: [`src/models/yield_forecast.py:25`](agri-brain-mvp-1.0.0/backend/src/models/yield_forecast.py:25) already implements Holt's double exponential smoothing with trend + 95% CI. Path B's `yield_query` should wrap this function rather than re-implement it, or risk two parallel Holt-Winters code paths that can drift.
- **Minimal revised proposal:** `backend/pirag/mcp/tools/yield_query.py` should import `yield_supply_forecast` and compute `uncertainty = std / max(|forecast[0]|, 1)` from its return dict. No new smoothing logic.

### D5. Routing-path `guards_passed` ≠ three-guard aggregate (2.4)

- **What the master prompt (and manuscript §3) assume:** three guards (dimensional analysis, feasibility, sim-verify) aggregate to a single `guards_passed` flag that gates Δz.
- **What the code actually shows:**
  - `PiRAGPipeline.ask` (line 89 of `agent_pipeline.py`) does compute `guards_ok = all([u_ok, f_ok, s_ok])` from the three guards. This feeds the `/ask` API.
  - **Routing context** comes from `retrieve_role_context`, which overrides `guards_passed` at [`context_builder.py:299`](agri-brain-mvp-1.0.0/backend/pirag/context_builder.py:299) with `len(citations) > 0 AND top_citation_score > 0.15`. That is a retrieval-quality check, not the three-guard aggregate.
  - `compute_context_modifier` gates Δz on this retrieval-quality `guards_passed`, not on the three-guard aggregate.
- **Impact on Path B:** none directly (ψ₅ is gated by the same flag). But the manuscript should not claim "three guards gate the routing modifier" — it is really a retrieval-quality gate in the routing path and a three-guard aggregate in the /ask path.
- **Minimal revised proposal:** either (a) have `retrieve_role_context` call `units_consistent`/`within_ranges`/`verify_with_sim` on the synthesised answer before overwriting `guards_passed`, OR (b) update manuscript §3 and §4 to describe the routing gate honestly as a retrieval-quality gate, with the three-guard aggregate documented separately as the /ask-path gate. (a) is the more honest fix for the paper claim; (b) is cheaper.

---

## Non-blocking notes

- **Test count drift:** memory says 59, actual is 65. Update memory. Post-Path-B target is 79 (65 + 14), not 73.
- **Tool count drift:** the static default registry has 12 tools; the **runtime registry** (after coordinator init with `context_enabled=True`) has 17 tools. Path B's "13" is therefore a subset claim, not the live count. Reconcile in the manuscript.
- **Δz clamp is already ±1.0:** `_MODIFIER_CLAMP = 1.0` in `context_to_logits.py:29`. The master prompt §5.4 instruction to grep for `±0.30` is moot; no legacy value remains in the code.
- **EWM / surplus_notification / Physics-informed reranker:** `grep` found zero hits in the repo. No stale prose to excise in the code — manuscript-only checks still apply.
- **The 4-mode aggregator vs 8-mode single-seed runner:** `mvp/simulation/benchmarks/run_benchmark_suite.py:104` only aggregates 4 modes (agribrain, mcp_only, pirag_only, no_context). `run_single_seed.py` collects all 8. If Path B needs a `no_yield` 9th mode, both files need updates, not just one.

---

## What Claude recommends

1. **User decision first** — confirm the four Path B patch files exist somewhere not yet searched, or agree on an approach (synthesise from guide vs. regenerate externally).
2. **Update the master prompt / implementation guide** to reflect D2, D3, D5 before any code is written. The current guide's §3.3 (ghost removal) would *break* functionality; §3.4 (buffer patch) would be applied to a file that does not drive the benchmark.
3. **Do not run the 20-seed benchmark** (Section 4) until the patches are in and the tests pass. The benchmark is long and a misconfigured ψ₅ wiring would waste hours of compute.
4. **Manuscript honesty:** count tools as 17, describe the routing gate as retrieval-quality not three-guard, and align the yield-forecaster narrative with the fact that Holt-Winters is already in `src/models/yield_forecast.py` and already feeds `env_state["supply_hat"]`.

---

## Evidence pointers

- [`agri-brain-mvp-1.0.0/backend/pirag/mcp/registry.py`](agri-brain-mvp-1.0.0/backend/pirag/mcp/registry.py) — `get_default_registry` (12 tools)
- [`agri-brain-mvp-1.0.0/backend/pirag/mcp/agent_capabilities.py`](agri-brain-mvp-1.0.0/backend/pirag/mcp/agent_capabilities.py) — 5 late-registered tools including `recovery_capacity_check`
- [`agri-brain-mvp-1.0.0/backend/pirag/mcp/tool_dispatch.py:159`](agri-brain-mvp-1.0.0/backend/pirag/mcp/tool_dispatch.py:159) — distributor workflow step for `recovery_capacity_check`
- [`agri-brain-mvp-1.0.0/backend/src/agents/coordinator.py:137`](agri-brain-mvp-1.0.0/backend/src/agents/coordinator.py:137) — `register_all_agent_capabilities` call
- [`agri-brain-mvp-1.0.0/backend/src/models/yield_forecast.py`](agri-brain-mvp-1.0.0/backend/src/models/yield_forecast.py) — existing Holt-Winters forecaster
- [`agri-brain-mvp-1.0.0/backend/pirag/context_to_logits.py:40`](agri-brain-mvp-1.0.0/backend/pirag/context_to_logits.py:40) — `THETA_CONTEXT ∈ ℝ³ˣ⁵`
- [`agri-brain-mvp-1.0.0/backend/pirag/context_builder.py:299`](agri-brain-mvp-1.0.0/backend/pirag/context_builder.py:299) — retrieval-quality `guards_passed`
- [`agri-brain-mvp-1.0.0/backend/pirag/agent_pipeline.py:89`](agri-brain-mvp-1.0.0/backend/pirag/agent_pipeline.py:89) — three-guard aggregate
- [`mvp/simulation/generate_results.py:311`](mvp/simulation/generate_results.py:311) — `env_state` construction in the benchmark
- [`mvp/simulation/generate_results.py:295`](mvp/simulation/generate_results.py:295) — existing Holt-Winters call in the benchmark
- Knowledge base: 20 `.txt` docs under `agri-brain-mvp-1.0.0/backend/pirag/knowledge_base/`
- Contracts: 6 files under `agri-brain-mvp-1.0.0/contracts/hardhat/contracts/`
- Test baseline: `pytest -q` → 65 passed, 0 failed, 14.74 s
