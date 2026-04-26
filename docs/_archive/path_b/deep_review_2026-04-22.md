> ⚠️ **HISTORICAL SNAPSHOT — REVERTED**. This document describes the Path B prototype (6D ψ, 3×6 `THETA_CONTEXT`, `no_yield` mode, `yield_query` MCP tool) that was reverted on `main`. Do not apply tracked-change instructions to the manuscript. See `docs/path_b/README.md` and the live docs for the current system.

# AgriBrain Deep Review — Code and Paper Reconciliation

Run by: Claude, 2026-04-22
Repo HEAD: `afccd7f` (`origin/main`)
Paper: `AB_R4.docx` (converted snapshot at `docs/path_b/_paper_snapshot.md`)

Scope: exhaustive deep read of the repository (no code execution) cross-checked
against every quantitative and structural claim in the paper. Findings are
grouped by severity and include a recommended fix path for each.

---

## Category A, code bugs and inconsistencies (repo-only, no paper implication)

### A1, BH-FDR correction applied per baseline, not per scenario family

**Where**: `mvp/simulation/benchmarks/aggregate_seeds.py:147-188`.
**Issue**: The Benjamini-Hochberg correction loop resets `pvals = {}` at line 152,
inside the baseline loop, so each `(scenario, baseline)` pair forms a 6-metric
family of its own. Paper Section 3.13 declares the multiplicity-control family
as "across the five scenario comparisons" (single family across scenarios per
primary endpoint). Even on the paper's own description, the per-baseline scope
over-conservatively inflates each p-value within the family while
under-controlling across the real family (all scenario-level H1 tests).
**Severity**: CRITICAL for the statistical narrative.
**Fix**: Decide the intended family first. If it is the 5 scenario-level H1
tests (agribrain vs no_context on ARI per scenario), collect those 5 p-values
centrally, correct them together, and report that single adjustment. If the
intended family is larger (e.g. 5 scenarios × 6 baselines × 6 metrics = 180),
state so and correct jointly. Either way, restructure the loop so `pvals`
accumulates across the appropriate axis before `benjamini_hochberg()` is
called.

### A2, Context learner magnitude cap allows 2x initial, paper says 1x

**Where**: `backend/pirag/context_learner.py:131`, `max_mag = np.maximum(np.abs(self.initial_theta) * 2.0, 0.10)`.
**Paper claim**: Section 3.9, "A magnitude bound caps each entry at its initial
absolute value".
**Code behaviour**: cap is 2 × |initial| with a floor of 0.10, which is a
four-fold gap on any near-zero entry and a 2x cap elsewhere.
**Severity**: MEDIUM, affects reported weight-change-norm in `learner_summary`
and the "prevents runaway updates" claim.
**Fix**: either change the multiplier to 1.0 (paper-faithful; tighter cap) or
update the paper text to state the 2x bound. The 2x bound is more permissive
and was presumably chosen to let learned weights actually move; paper-text fix
is the lower-risk option.

### A3, Permutation / bootstrap sample counts smaller than paper

**Where**: `aggregate_seeds.py:29,36,51`; `run_benchmark_suite.py:47,61`.
**Paper claim**: Section 3.13, "paired permutation tests with 10,000
permutations" and "bias-corrected accelerated bootstrap 95% confidence
intervals over 10,000 resamples".
**Code**: `n_boot=1000`, `n_perm=4000` in all benchmark scripts.
**Severity**: MEDIUM for a journal that reviewers will scrutinise
statistically.
**Fix**: raise `n_boot` to 10_000 and `n_perm` to 10_000 in both files, then
rerun the HPC benchmark. Cost: ~2-3x longer aggregator step on HPC; trivial
compared to the seed runs themselves.

### A4, Paper claims Holm-Bonferroni, code uses Benjamini-Hochberg

**Where**: `aggregate_seeds.py:76` (function `benjamini_hochberg`) is the one
used at line 170.
**Paper claim**: Section 3.13, "Multiplicity is controlled via Holm-Bonferroni
correction".
**Code**: BH-FDR (step-up), not Holm-Bonferroni (step-down, stricter family-
wise error control).
**Severity**: MEDIUM for reviewer scrutiny.
**Fix**: either add a `holm_bonferroni(pvals)` helper and switch to it (most
faithful to paper text; stricter), or update the paper to say "Benjamini-
Hochberg (FDR) correction" which is more common in process systems work and
well-aligned with the existing code. Recommended: paper edit, since BH-FDR is
the defensible modern default for multiple-metric reporting.

### A5, Stale docstrings (post Path B)

**Where**:
- `mvp/simulation/generate_results.py:5,11-13` — module docstring still says
  "5 scenarios × 8 modes" and lists only agribrain/mcp_only/pirag_only as
  context-enabled. Code runs 9 modes, and `no_yield` is context-enabled.
- `backend/pirag/context_learner.py:100` — `update()` docstring says
  `psi : (5,) context feature vector` but psi is 6-dim after Path B.
**Severity**: MINOR but will confuse reviewers reading the source.
**Fix**: three line edits, no functional change.

### A6, Silent default in supply-uncertainty extraction

**Where**: `generate_results.py:315-324`, `_std` retrieved with `.get("std", 0.0)`
while `_point` comes from `fc["forecast"][0] if fc["forecast"] else 1.0`.
If a future refactor lets `forecast` be present without `std`, the ratio will
be computed against a synthetic point of 1.0 and the supply-uncertainty signal
will be silently wrong. The Holt-Winters forecaster today always returns both,
so no current bug, but the defensive pattern is fragile.
**Severity**: MINOR.
**Fix**: either tie both fields to the same `if` gate, or assert presence.

### A7, Third piRAG guard is inline, not a named module

**Where**: paper Section 3.7 names three guards (dimensional analysis,
feasibility, retrieval quality); code has `backend/pirag/guards/unit_guard.py`
and `feasibility_guard.py` but the retrieval-quality check is inlined in
`context_to_logits.py:172` as `rag_context.get("guards_passed", ...)`.
**Severity**: MINOR. Behaviour is correct; surface-area audit will find only
two files.
**Fix**: extract the retrieval-quality check into
`backend/pirag/guards/retrieval_guard.py` for symmetry with the paper, or add
an aliasing comment in the existing guards directory.

### A8, No tests cover cyber_outage or no_yield end-to-end

**Where**: `agri-brain-mvp-1.0.0/backend/tests/` and `backend/pirag/tests/`.
None of the 85 tests exercise `select_action(..., scenario='cyber_outage', hour >= 24)`
or a full `run_episode` with `mode='no_yield'`. The VALID_MODES/_PINN_MODES/
_MCP_WASTE_MODES gaps that were found in the post-Path-B sweep all would have
been caught by a single integration test covering scenario x mode coverage.
**Severity**: MEDIUM for journal reproducibility.
**Fix**: add `tests/test_run_episode_coverage.py` with a ~20-step cyber_outage
+ no_yield smoke test (truncate data_spinach to accelerate). One test per
new mode-scenario edge case. Keep under 10 s total so CI stays cheap.

---

## Category B, paper claims that do not match the repository

### B1, psi dimensionality, 5 vs 6

**Paper**: Section 3.8 Eq 10, Figure 3 caption, Table 3, Algorithm 1 step 5c,
Section 4.9 "ψ = [ψ₀, ..., ψ₄]". Paper treats psi as 5-dimensional throughout.
**Code**: `THETA_CONTEXT.shape == (3, 6)`, `MODIFIER_RULES` length 6, ablation
masks are 6-vectors. Path B (commit 330ff67) added ψ_5 = supply uncertainty
from `yield_query`.
**Severity**: CRITICAL. Every ψ-related figure caption, equation, and table
in the paper is out of date.
**Fix plan**: Path B is the intended forward state. Update the paper:
- Eq 10: `ψ = [ψ₀, ψ₁, ψ₂, ψ₃, ψ₄, ψ₅]ᵀ`
- Table 3: add row for ψ₅ "Supply uncertainty, MCP, [0,1], Normalised coefficient
  of variation of Holt-Winters supply forecast".
- Figure 3 caption: "six-dimensional context vector" and add ψ₅ to the MCP-
  side channel.
- Algorithm 1 step 5c: update the list to include ψ₅.
- Algorithm 1 step 5d ablation masks: `mcp_only=[1,1,0,0,1,1]`,
  `pirag_only=[0,0,1,1,0,0]`, add `no_yield=[1,1,1,1,1,0]` mask line.
- Θ_context matrix: add the +0.20, +0.05, -0.15 column for ψ₅.
- Add a paragraph in Section 3.8 explaining the ψ₅ addition: why supply
  uncertainty, sign justification, and why it enters only the MCP channel.

### B2, Number of experimental modes and episodes

**Paper**: Section 3.13, "eight modes ... 5 × 8 × 20 = 800 episodes". Table 4
lists 8 modes. Table 9 is an 8-mode ablation.
**Code**: `MODES` list in `generate_results.py:114` has 9 modes (adds
`no_yield`). HPC benchmark now produces 5 × 9 × 20 = 900 episodes.
**Severity**: CRITICAL for paper's quantitative statements.
**Fix plan**: update paper:
- Section 3.13 count: "nine modes ... 900 episodes" (or 1000 if the stress
  suite is counted separately).
- Table 4 row for no_yield: "Context = Full minus ψ₅, PINN = Yes, SLCA = Yes,
  Tests = H3 (Path B ablation, isolates supply-uncertainty contribution)".
- Section 4.8 add a subsection on the no_yield ablation (agribrain vs no_yield
  delta isolates ψ₅'s contribution). Numbers come from the HPC run.

### B3, Number of MCP tools, 7 vs 13 static / 18 runtime

**Paper**: Section 3.6, "Seven tools are exposed: compliance,
spoilage_forecast, slca_lookup, chain_query, recovery_capacity, footprint,
calculator".
**Code**: 13 statically registered tools (`get_default_registry` returns
13) + 5 runtime-registered role capability tools = 18 at runtime. Full list
(sorted): calculator, chain_query, check_compliance, context_features,
convert_units, cooperative_coordination_status, distributor_route_feasibility,
explain, farm_freshness_assessment, footprint_query, pirag_query,
policy_oracle, processor_throughput_status, recovery_capacity_check, simulate,
slca_lookup, spoilage_forecast, yield_query.
**Severity**: CRITICAL for Figure 1 caption, Section 3.6, Table 2.
**Fix plan**: update paper:
- Abstract / Fig 1 caption: "thirteen statically registered MCP tools with
  five additional runtime role-capability tools (eighteen total at simulation
  time)".
- Section 3.6 intro: list the thirteen static tools, explain the extra five
  runtime role-capability tools, keep the Table 2 role-specific dispatch but
  update with new tool names (`check_compliance`, not `compliance`; add
  `yield_query` to processor, cooperative, distributor; use
  `recovery_capacity_check` consistently).

### B4, Table 2 role-tool mapping is stale

**Paper Table 2**:
- Farm: compliance, slca_lookup, spoilage_forecast (3)
- Processor: compliance, chain_query, calculator (3)
- Cooperative: slca_lookup, chain_query, spoilage_forecast, footprint (4)
- Distributor: compliance, slca_lookup, spoilage_forecast, recovery_capacity,
  calculator (5)
- Recovery: chain_query, slca_lookup, footprint (3)
**Code (`pirag/mcp/tool_dispatch.py`)**:
- Farm: check_compliance, slca_lookup, spoilage_forecast (3)
- Processor: check_compliance, policy_oracle, chain_query, calculator,
  yield_query (5)
- Cooperative: slca_lookup, chain_query, spoilage_forecast, footprint_query,
  yield_query (5)
- Distributor: check_compliance, slca_lookup, spoilage_forecast,
  recovery_capacity_check, calculator, yield_query (6)
- Recovery: chain_query, slca_lookup, footprint_query (3)
**Severity**: CRITICAL. Paper's "the distributor invokes five tools while the
farm invokes three" needs update to "six tools vs three".
**Fix plan**: regenerate Table 2 directly from `tool_dispatch.py`. Paper line
at Section 4.9 "200 tool calls per episode per scenario" needs a refresh from
the HPC run's actual tool-call count.

### B5, Smart contract name mapping

**Paper Section 3.15**: six contracts listed as "agent registry, decision
logger, policy store, **incentive contract**, **governance contract**,
provenance registry".
**Code** `contracts/hardhat/contracts/`: AgentRegistry.sol, DecisionLogger.sol,
PolicyStore.sol, ProvenanceRegistry.sol, **SLCARewards.sol**, **AgriDAO.sol**.
**Severity**: MEDIUM. Semantics match (SLCARewards = incentive, AgriDAO =
governance), only names differ.
**Fix plan**: one-line edit in paper: "incentive contract (SLCARewards.sol,
rewards proportional to SLCA performance), governance contract (AgriDAO.sol,
propose-vote-finalize-queue-execute sequence)".

### B6, Figure 1 caption: "EWM supply" vs Holt-Winters

**Paper Figure 1 caption**: "perception layer (PINN spoilage, LSTM demand,
**EWM supply**)".
**Code**: supply forecast is Holt-Winters double exponential smoothing
(`backend/src/models/yield_forecast.py`). Paper Section 3.4 later correctly
calls it "Holt-Winters exponential smoother" (α=0.5, β=0.2).
**Severity**: MEDIUM. Internal paper inconsistency (Fig 1 vs Section 3.4).
**Fix**: change Fig 1 caption to "Holt-Winters supply forecaster".

### B7, Cyber outage confidence narrative conflicts with the fix

**Paper Section 4.4, Figure 6c discussion** (lines 1357-1363 of snapshot):
"Once the outage engages, confidence stabilizes near 1.0: the policy commits
probability mass to redistribution, and because environmental inputs during
sustained outage repeat, entropy collapses at every step. The flat trace is
the behavior expected of a regime-aware contextual policy that has recognized
the shift and committed to a consistent response."
**Code post-fix (commit 59dbc1c)**: `select_action` now returns the Bernoulli
policy distribution `[1-p, p, 0]` during outage, not the sampled one-hot. For
agribrain (p=0.82) the confidence is 0.571 (above the 0.66 band into the amber
zone), not 1.0. The flat-at-1.0 trace in the original paper was a reporting
bug rationalised post hoc.
**Severity**: CRITICAL. Journal reviewers will see the new fig4 panel (c)
once the HPC run regenerates it, and the text will not match.
**Fix plan**: rewrite Section 4.4 Figure 6c paragraph:
- Drop "confidence stabilizes near 1.0 ... entropy collapses".
- Replace with: "During the outage, the softmax is replaced by a Bernoulli
  reroute-success policy. For agribrain the reroute success probability is
  0.82, giving a decision-confidence of 0.57 on every sampled step (above the
  0.66 high-confidence threshold in the amber band), versus 0.37 for
  hybrid_rl (p=0.55). The confidence trace reports the honest Bernoulli
  entropy of the reroute decision rather than a degenerate one-hot."
- This framing makes the paper MORE defensible: it shows the integrated
  system preserving meaningful (not over-confident) policy structure under
  channel degradation.

### B8, Cyber outage parameter description

**Paper Section 3.9**: "stochastic rerouting model with action-specific
success probabilities (cold chain 0.55, local redistribution 0.82, recovery
0.60), estimated as the fraction of each route's operational steps that
remain feasible".
**Code**: `CYBER_REROUTE_PROB` is mode-specific, not action-specific. Values
0.55, 0.82, 0.60 are assigned to `hybrid_rl`, `agribrain-family`, `no_slca`
respectively.
**Severity**: CRITICAL. Semantic mismatch misleads reviewers on what the
scenario actually tests.
**Fix plan**: rewrite Section 3.9 paragraph to mode-specific language:
"mode-dependent reroute success probabilities reflect each mode's autonomous
intelligence: hybrid_rl=0.55, no_slca=0.60, no_pinn=0.65, and the fully
context-enabled modes (agribrain, mcp_only, pirag_only, no_context, no_yield)
all 0.82 through shared edge infrastructure." Update Table 6 cyber_outage row
accordingly.

### B9, Knowledge base continuity window

**Paper Section 3.7**: "temporal context window of 20 entries (six-hour
horizon)".
**Arithmetic check**: 20 entries × 15-minute cadence = 300 minutes = 5 hours,
not 6. 24 entries would give 6 hours.
**Severity**: MINOR, arithmetic inconsistency.
**Fix**: either change the paper to "five-hour horizon" or change the window
to 24 entries in `backend/pirag/temporal_context.py`. The 24-entry version is
what `dynamic_knowledge.py` uses (`block_size = 24`), and matches the
"every 24 timesteps (6 h)" synthesis cadence elsewhere in the paper.
Recommended: align on 24 entries across the codebase, update the paper's "20
entries" to "24 entries".

### B10, hybrid_rl characterisation

**Paper Section 4.6 and Table 11 caption**: "Hybrid RL denotes the static-to-
RL switchover controller".
**Code** `action_selection.py:353-354`: hybrid_rl is a straight RL policy,
`logits = THETA @ phi + gamma * tau` (no switchover conditional). Static mode
is a separate mode entirely (always cold chain).
**Severity**: MEDIUM, textual misdescription.
**Fix**: replace "static-to-RL switchover controller" with "linear softmax RL
policy over the same six-feature state vector, without SLCA or PINN bonuses".
Switchover is between modes (static vs hybrid_rl vs agribrain in the ablation
design), not inside hybrid_rl.

### B11, "Surplus notification" message type

**Paper Section 3.5 and Section 4.9**: "five typed message classes:
*spoilage alerts*, *surplus notifications*, *capacity updates*, *reroute
requests*, and *acknowledgments*".
**Code**: zero occurrences of `surplus_notification` in any .py file (verified
in the pre-HPC audit). The message classes actually defined in the code are
different.
**Severity**: MEDIUM to CRITICAL depending on whether reviewers audit the
inter-agent messaging layer.
**Fix plan**: locate the real message enum in `backend/src/agents/` (likely
`base.py` or `roles.py`) and update Table/prose to match the actual class
names. If "surplus notification" is valuable narrative, rename the
corresponding code enum to match.

### B12, Paper H1 threshold, p < 0.10

**Paper Section 3.2**: "adjusted p < 0.10".
**Paper Table 7**: all reported p_adj values are "< 0.001".
**Severity**: MINOR, an internal paper inconsistency (not a paper-code issue).
**Fix**: update threshold to "p < 0.05" (matches the practically-
significant threshold implied by reported values). Or state both: pre-
registered threshold p < 0.10; observed p_adj < 0.001.

### B13, Frontend panel dimensionality and modes

**Paper Figure 13c**: MCP/piRAG panel shows "five context features ψ₀
through ψ₄".
**Code** `frontend/src/components/ExplainabilityPanel.jsx` and related: UI
exposes 5 features only (ψ₀..ψ₄). ψ₅ is not wired to the panel. Simulation-
mode selector does not include no_yield.
**Severity**: MEDIUM for Figure 13 accuracy, MINOR for live deployment.
**Fix**: add ψ₅ label + value pane to ExplainabilityPanel; add no_yield to
mode selector. Or, if the paper's Figure 13 was rendered before Path B,
regenerate the screenshot once the frontend is updated and state in the
caption "six context features after Path B integration".

### B14, Frontend API Literal mode coverage

Already fixed in commit `afccd7f` (added `no_yield` to `Literal` in
`decide.py` and `app.py`). Note as resolved.

---

## Category C, inconsistent but defensible (no change needed)

- **W_SCALE = 10.2976** vs paper `10.30`: rounding, fine.
- **W_ALPHA = 0.7339** vs paper `0.734`: rounding, fine.
- **Θ matrix** values match paper Section 3.9 exactly.
- **Θ_context** values for ψ₀..ψ₄ match paper exactly (only row 5 was added
  in Path B).
- **LSTM hyperparameters** (hidden=16, lookback=48, lr=0.005, epochs=80)
  match.
- **Holt-Winters hyperparameters** (α=0.5, β=0.2) match.
- **Route distances** (120/45/80 km) match.
- **PINN residual clamp** ±0.08 matches.
- **Governance override threshold** (z[0] < -2.0 AND z[1] > z[0] + 3.0)
  matches exactly.
- **Causal explanation components** all 5 implemented in
  `explain_decision.py` (BECAUSE, feature attribution, WITHOUT
  counterfactual, citations, Merkle root).
- **Three piRAG extensions** all implemented (query expansion, reranker
  clamped [0, 0.30], temporal τ_mod = 1.3 - 0.6κ).
- **Circuit breaker** (failure_threshold=3, reset 5 s, exponential backoff)
  matches.
- **Hybrid retriever** (BM25 0.6, TF-IDF 0.4, k=4) matches.
- **Dynamic knowledge feedback** (every 24 timesteps) matches paper Section 3.7
  body text; only the 20-entry temporal window line (B9) disagrees with 6h.
- **Six smart contracts**, Solidity ≥ 0.8.28, governance sequence
  propose→vote→finalize→queue→execute, provenance anchoring — all implemented
  faithfully; only the two name mappings (B5) need a line in the paper.

---

## Recommended commit sequence before HPC submission

All items below are code-only and can land in two commits before `sbatch`.
Paper updates are a separate track.

**Commit 1, statistical integrity**:
1. Fix A1 (BH-FDR per-scenario family).
2. Bump A3 resampling counts (n_boot=10_000, n_perm=10_000).
3. Decide A4 (BH vs Holm) and align code + paper.
4. Align A2 magnitude cap with paper (or doc the 2x bound).

**Commit 2, docstrings and safety**:
5. Fix A5 stale docstrings (3 line edits).
6. Tighten A6 silent default.
7. Add A8 end-to-end smoke tests for no_yield x cyber_outage and any other
   mode-scenario pair not covered by existing tests.
8. Optional, A7 move retrieval_guard into its own file for surface symmetry.

After these land the HPC run produces final numbers; paper edits (Category B)
are then one manuscript pass, not code changes.

---

## Paper revision checklist

One-pass edit of `AB_R4.docx`, grouped to minimise conflict risk:

**Structural updates driven by Path B**:
- [ ] B1 ψ 5→6 across Eq 10, Table 3, Fig 3 caption, Algorithm 1 steps 5c/5d,
      Θ_context matrix (Section 3.8).
- [ ] B2 modes 8→9 and episodes 800→900 across Section 3.13, Table 4,
      ablation Table 9.
- [ ] B3 seven tools → thirteen static + five runtime in Section 3.6, Fig 1
      caption, abstract if touched.
- [ ] B4 Table 2 regenerated from tool_dispatch.py.

**Narrative-only updates**:
- [ ] B5 contract names SLCARewards.sol / AgriDAO.sol.
- [ ] B6 Fig 1 caption EWM → Holt-Winters.
- [ ] B7 Section 4.4 rewrite confidence-trace paragraph (post-fix).
- [ ] B8 Section 3.9 cyber outage mode-specific wording.
- [ ] B9 temporal window 20 entries → 24 (or paper hour count to 5).
- [ ] B10 hybrid_rl description.
- [ ] B11 message-class names (check actual enum first).
- [ ] B12 H1 threshold phrasing.
- [ ] B13 Fig 13 screenshot refresh and caption.

**No change, defensible as is**: Category C items (W_SCALE rounding, Θ match,
LSTM hyperparameters, Holt-Winters, distances, PINN clamp, override rule,
explanation components, piRAG extensions, circuit breaker, retriever weights,
knowledge feedback cadence, contracts implementation depth).

---

## Authorial sign-off on scope

This review did not run any code. All verdicts come from file reads and grep
over the tracked tree at `afccd7f`. HPC numbers (Table 7, 8, 9, 11) cannot
be reproduced locally and will be regenerated by `sbatch hpc_run.sh`. The
text edits in Category B depend on those final numbers for a handful of
specific figures; the structural text edits (B1, B2, B3, B4, B5, B6, B8, B9,
B10) can be drafted today and spliced once the HPC numbers arrive.
