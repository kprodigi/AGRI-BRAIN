# Architecture

Technical reference for the AGRI-BRAIN decision pipeline. For installation
and reproduction instructions see the [README](../README.md) and
[HOW_TO_RUN.md](../HOW_TO_RUN.md); for the statistical methodology see
[STATISTICAL_METHODS.md](STATISTICAL_METHODS.md).

Information flows through four layers at different timescales: a
**perception layer** turns telemetry into a compact state estimate; a
**communication layer** brings in nonlocal information through three
channels (typed peer messages, MCP tool calls, physics-informed retrieval);
an **action layer** maps the combined signals to a routing decision through
a context-shifted softmax policy; and a **governance layer** records every
decision and its evidence on a Merkle-anchored ledger.

## Perception and forecasting

- **Arrhenius–Baranyi spoilage ODE with a physics-informed neural residual
  correction** trained under an ODE-residual penalty
  (`agribrain/backend/src/models/pinn_net.py`). The residual is clamped to
  ±0.08 so the estimate stays physically plausible under any network output.
- **LSTM demand forecaster** (numpy-only, 16 hidden units, truncated BPTT)
  with in-sample residual-standard-deviation prediction uncertainty. Holt's
  linear (double exponential smoothing) demand fallback is available via the
  `FORECAST_METHOD` env var.
- **Holt's linear yield/supply forecaster** (level + trend form of Holt
  1957, no seasonal component) for inventory projection, with matching
  residual-std uncertainty. Both forecasts feed symmetrically into the state
  vector φ(s) at indices 6–8 (supply point, supply uncertainty, demand
  uncertainty).
- **Softmax contextual policy** over a 10-dimensional state feature vector
  (perception + symmetric supply and demand forecast channels +
  demand-volatility price-pressure proxy) and a 5-dimensional institutional
  context modifier.

## Communication: MCP interoperability layer

- **14 statically registered tools** (13 register by default; the
  conditional `simulate` tool registers only when `SIM_API_BASE` is set) and
  **5 runtime role-capability tools** registered per coordinator (18 active
  at simulation time), plus 13 resources and 5 prompts. See
  [`TOOL_INVENTORY.md`](../agribrain/backend/pirag/mcp/TOOL_INVENTORY.md)
  for the per-tool listing.
- **JSON-RPC 2.0** dispatch with three transports: `InProcessTransport`
  (canonical; JSON-roundtrips inside the process; used by the simulator and
  the FastAPI `/mcp` endpoint), `StdioTransport` (newline-delimited JSON-RPC
  over pipes; pair with the shipped `python -m pirag.mcp.serve` entry
  point), and `HTTPTransport` (synchronous JSON-RPC over HTTP POST, with
  `SSETransport` retained as a backward-compatible alias).
- **Protocol recording**: recorded entries are real `(MCPMessage, response)`
  pairs through the in-process dispatcher; when wire-side serialization
  matters, `InProcessTransport` JSON-roundtrips on every send. The simulator
  runs MCP through `InProcessTransport` for deterministic per-step
  reproducibility.
- **MCP governance override** that mandates rerouting under simultaneous
  critical compliance violation and high spoilage forecast.

## Communication: physics-informed retrieval (piRAG)

- **20-document knowledge base** (regulatory, SOP, SLCA, contingency)
  retrieved by BM25 + TF-IDF hybrid retrieval (k=4, 20% retrieval ratio).
- **Physics-aware query expansion** when thermal thresholds are crossed, and
  **physics-aware reranking** (temperature-proximity scoring, spoilage-stage
  keyword density, urgency cues; the upstream Arrhenius decay rate is
  supplied as an input for query expansion rather than evaluated inside the
  reranker).
- **Keyword extraction** from retrieved passages (thresholds, regulatory
  references, required actions) for human-readable decision evidence.
- **Guards** (retrieval quality, dimensional analysis, feasibility) zero the
  context modifier when retrieval is weak, so poor evidence never pushes a
  decision below the no-context baseline.

## Context fusion, policy, and learning

- **5-agent coordinator** (Farm, Processor, Cooperative, Distributor,
  Recovery) dispatching decisions at lifecycle-stage boundaries.
- **Context feature integration** via a 5D institutional context vector
  (compliance severity, forecast urgency, retrieval confidence, regulatory
  pressure, recovery saturation) with a learned Θ_context ∈ ℝ^(3×5) weight
  matrix and SLCA bonus amplification.
- **Online REINFORCE learning** of context weights with sign constraints
  preserving domain-justified directions while adapting magnitudes to
  scenario conditions.
- **Circular-economy scoring** for composting, animal feed, and food bank
  pathways promotes reverse logistics to a first-class routing action.
- **Operational feasibility diagnostics** with decision latency and
  constraint-violation rates reported per scenario/method.

## Governance and explainability

- **Causal explanation engine** producing BECAUSE/WITHOUT reasoning with
  inline [KB:] citations, counterfactual probability comparisons, and
  Merkle-rooted provenance chains. Because the same context vector drives
  both the decision and the explanation, explanations are structural rather
  than post-hoc.
- **Off-chain Merkle audit ledger with optional on-chain anchoring** via
  Hardhat/Solidity smart contracts. Each episode produces a Merkle root over
  the routing decisions; anchoring on-chain is gated by `CHAIN_SUBMIT=1`.
  The published runs do not deploy to a permissioned EVM — the only
  configured network is `localhost`/Hardhat (see
  [contracts/README.md](../agribrain/contracts/README.md) for the production
  checklist). The contract suite (`AgentRegistry`, `DecisionLogger`,
  `PolicyStore`, `ProvenanceRegistry`, `SLCARewards`, `AgriDAO`)
  demonstrates role-gated agent registration, append-only provenance,
  persisted episode roots, a key-whitelisted policy store, and
  reentrancy-guarded governance. Three of the six contracts use role-based
  access control with separated functional roles — `DecisionLogger`
  (ADMIN_ROLE / LOGGER_ROLE), `ProvenanceRegistry` (ADMIN_ROLE /
  ANCHORER_ROLE), `SLCARewards` (ADMIN_ROLE / REWARDER_ROLE / SLASHER_ROLE) —
  with backward-compatible `setAuthorized`/`onlyOwner` shims on the first
  two; `AgentRegistry`, `PolicyStore`, and `AgriDAO` use owner or whitelist
  patterns. Anchored roots are verified by
  `mvp/simulation/analysis/verify_anchored_root.py`.

## Experimental design and statistics

- **8 canonical operating modes** plus 11 §4.7 sensitivity ablations
  (19 total): static, hybrid RL, no PINN, no SLCA, no context, MCP only,
  piRAG only, full AGRI-BRAIN, plus `agribrain_cold_start`,
  `agribrain_pert_{10,25,50}` (with-learning),
  `agribrain_pert_{10,25,50}_static` (no-learning), `agribrain_no_bonus`
  (SLCA-bonus zeroing), and `agribrain_theta_pert_{10,25,50}`
  (primary-policy-weight perturbation). The 8-mode count refers to the
  canonical publication ablation; the 19-mode set drives the §4.7
  supplementary analysis.
- **Dual-mode stochastic simulation** with 8 field-realistic uncertainty
  sources: sensor noise (tempC ±2.5 °C, RH ±7%), demand variability
  (CV 25%), inventory/yield uncertainty (CV 22%), transport distance jitter
  (CV 22%), spoilage model error (k_ref CV 20%, Ea_R CV 14%), scenario onset
  timing jitter (±6 h), policy weight perturbation (σ 0.15), and
  per-(mode, seed) policy-temperature heterogeneity (LogNormal σ 0.25).
  Set `DETERMINISTIC_MODE=true` for audit mode.
- **Robustness + significance toolkit**: multi-seed stress tests (sensor
  noise, missing telemetry, delay, MCP fault injection, compounded),
  pair-aware test selection (Wilcoxon signed-rank for paired comparisons,
  Mann–Whitney U for unpaired), Holm–Bonferroni primary-family correction,
  Benjamini–Yekutieli FDR for the secondary family, bootstrap CI on every
  Cohen's d, and Hedges' g small-sample correction.

## Backend API

```
GET  /health                 - Health check
POST /case/load              - Load spinach CSV into state
GET  /kpis                   - Computed KPIs from loaded data
GET  /telemetry              - Sensor time-series (tempC, RH, inventory, demand)
GET  /predictions            - Spoilage predictions, demand and yield forecasts
POST /decide                 - Run decision engine (softmax policy)
GET  /last-decision          - Most recent decision memo
GET  /decisions              - Decision feed
POST /scenarios/run          - Apply a scenario perturbation
POST /scenarios/reset        - Reset to baseline
GET  /scenarios/list         - List 5 available scenarios
GET  /governance/policy      - Current Policy object
POST /governance/policy      - Update policy parameters
GET  /governance/chain       - Blockchain configuration
GET  /audit/logs             - Audit log array
GET  /audit/memo.json        - Decision memo as JSON
GET  /audit/memo.pdf         - Decision memo as PDF
POST /results/generate       - Start simulation in background (returns immediately)
GET  /results/status         - Poll simulation job progress
GET  /results/summary        - Fetch last completed summary JSON
GET  /results/figures/{name} - Serve generated figure files
POST /mcp/mcp                - JSON-RPC 2.0 MCP endpoint (tools/call, resources/read, prompts/get)
GET  /mcp/resources          - List MCP resources
GET  /mcp/prompts            - List MCP prompts
POST /rag/ask                - Query the piRAG pipeline (physics-informed retrieval)
POST /rag/ingest             - Ingest documents into the piRAG knowledge base
POST /mcp/call               - Call an MCP tool (legacy)
WS   /stream                 - WebSocket real-time decision stream
```

## Frontend

Modern React dashboard built with shadcn/ui, featuring eight pages:

| Page | Description |
|------|-------------|
| **Operations** | KPI summary grid, real-time telemetry charts with temperature zones, spoilage & yield preview |
| **Quality** | Circular spoilage risk gauge, shelf-life countdown, IoT sensor charts, PINN vs ODE comparison |
| **Decisions** | Timeline view with role/action filters, decision cards with expandable MCP/piRAG explainability panels (causal BECAUSE/WITHOUT reasoning, 5-axis context feature radar chart, extracted keyword tags, Merkle-rooted provenance chains), analytics sidebar with pie chart, CSV/PDF export |
| **Map** | Leaflet map of South Dakota supply chain nodes with route overlays and live KPI popups |
| **Analytics** | Executive summary banner, interactive cross-scenario tables & charts, ablation study, radar profiles, scenario deep-dive gallery, carbon footprint analysis |
| **MCP/piRAG** | MCP protocol overview, context feature visualization, knowledge base browser, protocol traces, causal reasoning panel |
| **Demo** | Interactive system demo with live pipeline walkthrough and a step-by-step multi-agent run view |
| **Admin** | Seven tabs — Policy parameters, Blockchain status & config, Audit log, Scenario runner, Quick Decision, Runtime config, MCP Explorer (tool browser with 14 statically registered tools, live resource monitor, prompt template browser, live tool invocation with presets, piRAG knowledge base search, JSON-RPC protocol interaction log) |

**Tech stack:** React 18, React Router 7, shadcn/ui (Radix), Tailwind CSS,
Recharts, React-Leaflet, Framer Motion, Sonner toasts, Vite 7
