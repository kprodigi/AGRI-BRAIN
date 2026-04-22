# Path B — Manuscript Updates (tracked-changes copy)

Drafted from `final_implementation_prompt` Section F. Apply each block with tracked changes (`author="Reviewer"`) to the latest manuscript docx. Every number below is a placeholder to be filled verbatim from the Phase 2 / Phase 3 CSVs after the benchmark completes; do not paste drafts into the manuscript until the CSVs land.

Implementation status: code and tests committed in `330ff67`; benchmark Phases 2-3 pending.

---

## F.1 — Section 3.3 (perception layer)

Replace any sentence mentioning `EWM supply projection` or `exponentially weighted moving average` (grep returned zero such sentences in code; check the manuscript directly) with:

> The perception layer comprises three components: a physics-informed neural network (PINN) producing a spoilage-risk estimate ρ from temperature and humidity, an LSTM demand forecaster producing the one-step-ahead demand ŷ, and a Holt-Winters yield forecaster producing the one-step-ahead supply forecast and its confidence interval. The first two outputs populate the six-dimensional state vector φ(s) together with raw telemetry; the yield forecast is exposed through the MCP protocol as the `yield_query` tool and enters the routing policy via the supply-uncertainty component ψ₅ of the context vector (Section 3.7).

## F.2 — Section 3.5 (Channel 1)

Find-and-replace: `surplus notification` → `surplus alert`. The code uses `MessageType.SURPLUS_ALERT`; the manuscript is the only remaining place the stale term can appear.

## F.3 — Section 3.6 (Channel 2 / MCP)

Replace the existing "Seven tools are exposed..." or "Twelve tools are registered..." paragraph with:

> AGRI-BRAIN's MCP server exposes 18 tools at runtime, registered across two layers. The default registry contains 13 statically-registered tools that are available the moment the server starts: six domain tools that drive routing context (`check_compliance`, `spoilage_forecast`, `slca_lookup`, `chain_query`, `footprint_query`, `yield_query`); four protocol-support tools (`policy_oracle`, `calculator`, `convert_units`, `simulate`); and three explainability tools (`pirag_query`, `explain`, `context_features`). When the agent coordinator initialises with `context_enabled=True`, an additional five agent-capability tools are bound: capacity-management tools used by the distributor and recovery roles (for example, `recovery_capacity_check`) and coordinator-level introspection tools. The six domain tools in the first layer populate the context vector ψ consumed by the routing policy; all other tools provide protocol-level facilities used during decision construction, governance enforcement, and audit. Each invocation produces a JSON-RPC 2.0 request (method, typed parameters, request identifier) and receives a structured response. Role-specific dispatch patterns (Table 2) reflect different information needs at different lifecycle stages.

Update Table 2 so the distributor row lists `recovery_capacity_check` and `yield_query` alongside the existing entries; update the processor and cooperative rows to include `yield_query`.

## F.4 — Section 4 (context layer methodology)

Replace the 5-feature ψ paragraph with:

> The context vector aggregates evidence from MCP and piRAG into a six-dimensional feature vector ψ ∈ ℝ⁶ whose components are ψ₀ = compliance severity, ψ₁ = forecast urgency, ψ₂ = retrieval confidence, ψ₃ = regulatory pressure, ψ₄ = recovery saturation, and ψ₅ = supply uncertainty. The first two and the final two features originate from MCP tools (`check_compliance`, `spoilage_forecast`, `chain_query`, `yield_query`); ψ₂ and ψ₃ are extracted from piRAG retrieval metadata. ψ₅ is the coefficient of variation of the Holt-Winters one-step yield forecast, clamped to the unit interval: ψ₅ = min(σ / max(|μ|, 1), 1), where σ is the rolling standard deviation and μ is the point forecast.
>
> A sign-constrained weight matrix Θ_context ∈ ℝ³ˣ⁶ maps ψ to a logit modifier Δz ∈ [−1, +1]³ that is added to the policy logits before the softmax. The signs of Θ_context are domain-justified and preserved by the REINFORCE update through a sign-mask projection. The supply-uncertainty column carries +0.20 toward cold chain (preserve optionality), +0.05 toward local redistribution (release excess), and −0.15 toward recovery (avoid low-value commitment under uncertainty). These magnitudes are smaller than the compliance and forecast columns because supply uncertainty acts as a tiebreaker rather than a primary driver.

### F.4.a — Routing-path guard description (mandatory honesty fix)

If the manuscript claims that "three guards (dimensional analysis, feasibility, simulation verification) gate the routing modifier", correct to:

> Two retrieval paths are gated differently. The `/ask` path used for human queries aggregates three guards (dimensional consistency, parameter range feasibility, simulation cross-check) into a single boolean before returning a synthesised answer. The routing path used by the agent policy applies a lighter-weight retrieval-quality gate, requiring at least one citation with a top score above 0.15, before allowing the context modifier Δz to perturb the policy logits. The lighter gate reflects the routing path's tighter latency budget and the fact that any low-quality retrieval simply zeroes Δz rather than blocking the decision.

If Section G.1 is applied (unified three-guard routing gate), rewrite the paragraph to describe one unified gate.

## F.5 — Section 4 (Δz bound)

Grep the manuscript for `±0.30` or `+/-0.30` near Δz. If found, change to `±1.0` to match `_MODIFIER_CLAMP = 1.0`. Brief note: the bound was widened from the originally proposed ±0.30 to ±1.0 to give the modifier meaningful impact relative to the base policy logits.

## F.6 — Section 5 (results)

Update Tables 5, 6, 7 verbatim from Phase 2 CSVs. Add a new Table 8 (or subsection) reporting the `agribrain` − `no_yield` attribution from Phase 3.

Add one or two sentences describing ψ_5's mechanistic role:

> During overproduction and adaptive-pricing scenarios, the Holt-Winters confidence interval widens, ψ_5 grows, and Δz favours cold chain (optionality preservation) and local redistribution over recovery, reducing waste without committing inventory to low-value endpoints.

## F.7 — Figure 1 caption

Replace with:

> Figure 1. AGRI-BRAIN decision pipeline. Sensor telemetry feeds a perception layer (PINN spoilage producing ρ, LSTM demand producing ŷ, Holt-Winters yield forecaster producing supply forecasts to Channel 2), five role-specific agents, and three communication channels: direct messages (Channel 1), the MCP protocol layer with thirteen statically-registered tools and five additional agent-capability tools bound at coordinator initialisation (Channel 2; four domain tools shown), and piRAG retrieval (Channel 3, BM25 + TF-IDF over a 20-document knowledge base with a six-hour dynamic feedback cycle). MCP outputs ψ₀, ψ₁, ψ₄, ψ₅ and piRAG outputs ψ₂, ψ₃ converge into a six-dimensional context vector ψ that is mapped through a sign-constrained weight matrix Θ_context ∈ ℝ³ˣ⁶ into a logit modifier Δz ∈ [−1, +1]. The regime-aware contextual softmax routing policy selects among three routes (cold chain 120 km, redistribution 45 km, recovery 80 km) under a deterministic governance override. Routing outcomes are evaluated by four metrics (waste, carbon, SLCA, ARI) and logged to six Solidity smart contracts on a permissioned EVM testnet. Dashed arrows denote feedback loops: decision history (blockchain → Channel 3 piRAG knowledge base) and the sign-constrained REINFORCE update on Θ_context.

Replace the figure file in the manuscript with `C:\Users\Nahid\Downloads\Figure1_AGRI_BRAIN.pptx`. Re-insert the editable PPTX rather than editing inside Word.

---

## Verification grep list for the manuscript

Before submission, run these searches in the manuscript and confirm zero hits:

- `EWM`
- `exponentially weighted moving average` (other than in a legacy citation)
- `surplus notification`
- `Physics-informed reranker`
- `±0.30` or `+/-0.30` near Δz
- `recovery_capacity` without the `_check` suffix
- `Seven tools` or `Twelve tools` (both obsolete)
- AI phrasing: `furthermore`, `moreover`, `leveraging`, `harnessing`, `holistic`
- Em dashes (`—`): replace with commas, semicolons, or colons
