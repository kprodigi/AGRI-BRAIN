# HOW TO RUN — AGRI-BRAIN MVP

Step-by-step guide to run the full AGRI-BRAIN stack: backend API, frontend dashboard,
standalone simulation, and figure generation.

---

## Prerequisites

| Tool    | Minimum version | Check with         |
|---------|-----------------|--------------------|
| Python  | 3.10+           | `python3 --version`|
| pip     | 22+             | `pip --version`    |
| Node.js | 18+             | `node --version`   |
| npm     | 9+              | `npm --version`    |

> Optional: Hardhat (for on-chain features), Docker (not required).

---

## 1. Clone and navigate

```bash
git clone https://github.com/kprodigi/AGRI-BRAIN.git AGRI-BRAIN
cd AGRI-BRAIN
```

Throughout this guide ``AGRI-BRAIN`` is the repository root. If you
cloned into a different directory name (for example ``AgriBrain`` on
case-sensitive filesystems or ``agri-brain``), substitute that name
wherever ``AGRI-BRAIN`` appears in the commands below.

The repository layout:

```
AGRI-BRAIN/
├── README.md
├── HOW_TO_RUN.md
├── docs/screenshots/           # Frontend screenshots
├── agribrain/
│   ├── backend/                # FastAPI backend (port 8100)
│   │   ├── src/                # Application code
│   │   │   ├── app.py          # Main FastAPI app
│   │   │   ├── models/         # PINN spoilage, forecast, SLCA, policy
│   │   │   ├── routers/        # API route handlers
│   │   │   ├── agents/         # Multi-agent coordinator (5 roles)
│   │   │   └── chain/          # Blockchain integration
│   │   ├── pirag/              # PiRAG pipeline (RAG, MCP, provenance)
│   │   ├── experiments/        # Policy experiment scripts
│   │   ├── static/             # Swagger branding assets
│   │   └── pyproject.toml
│   ├── frontend/               # React + Vite dashboard (port 5173)
│   │   └── src/
│   │       ├── pages/          # Ops, Quality, Decisions, Map, Analytics, Admin
│   │       ├── components/ui/  # shadcn/ui component library
│   │       ├── components/explainability/ # ExplainabilityPanel (causal reasoning, radar, keywords, provenance)
│   │       ├── components/mcp/            # McpTab (tool browser, resource monitor, invocation, piRAG search)
│   │       ├── layouts/        # MainLayout (sidebar, header, theme toggle)
│   │       ├── hooks/          # useTheme, useWebSocket
│   │       ├── lib/            # Utility functions
│   │       └── mvp/            # API configuration
│   └── contracts/              # Solidity smart contracts (Hardhat)
├── mvp/
│   └── simulation/             # Standalone simulation & figure scripts
│       ├── generate_results.py # Scenario x mode simulation runner
│       ├── generate_figures.py # Publication figure generator
│       ├── stochastic.py       # 8-source stochastic perturbation engine (+ orthogonal telemetry-lag channel)
│       ├── reproduce_core.py   # One-command full reproduction pipeline
│       ├── benchmarks/         # Multi-seed benchmark & stress suites
│       ├── validation/         # Result validation & regression guards
│       ├── analysis/           # Diagnostics & paper evidence export
│       └── tests/              # Stochastic & benchmark test suites
└── .gitignore
```

---

## 2. Backend setup

### 2a. Create a virtual environment (recommended)

```bash
cd AGRI-BRAIN
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2b. Install backend dependencies

For interactive development (loose pyproject ranges):

```bash
pip install -e "agribrain/backend[dev]"
```

For paper-grade reproducibility (pinned versions matching the
artifact-manifest commit):

```bash
pip install -r agribrain/backend/requirements-lock.txt
pip install -e agribrain/backend --no-deps      # editable wiring only
```

The `[dev]` extra adds `httpx` and `pytest-asyncio`, required by the
TestClient-based integration tests. The `requirements-lock.txt`
header documents the Python version used to generate it; if your
Python differs from the lockfile baseline, run `docs/RELEASE.md`
step 2 to regenerate from a clean venv. Without the lockfile,
dependencies float within their pyproject ranges (e.g.
`numpy>=2.1,<3`); with the lockfile, every transitive dep is pinned
for bit-comparable reproduction.

The pyproject declares: fastapi, uvicorn, pydantic, numpy, pandas,
matplotlib, reportlab, orjson, requests, web3, python-multipart,
pyyaml, scipy. The `[dev]` extra adds pytest, httpx, pytest-asyncio.

### 2b-extra. Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_METHOD` | `lstm` | Demand forecaster: `lstm` (numpy LSTM) or `holt_winters` (legacy env alias selecting Holt's linear, level + trend; the implementation is not seasonal Holt-Winters) |
| `ONLINE_LEARNING` | `false` | Enable REINFORCE policy gradient updates |
| `LLM_PROVIDER` | `template` | RAG answer engine: `template` or `api` |
| `DATA_CSV` | (auto) | Override path to spinach sensor CSV |
| `RAG_CONTEXT_ENABLED` | `true` | Enable MCP/piRAG context integration in agribrain mode |
| `DETERMINISTIC_MODE` | `false` | `true` = exact reproducibility (audit), `false` = 8-source stochastic perturbations |
| `STOCH_TEMP_STD_C` | `2.5` | Source 1: temperature sensor noise sigma (°C) |
| `STOCH_RH_STD` | `7.0` | Source 1: humidity sensor noise sigma (%) |
| `STOCH_DEMAND_FRAC_STD` | `0.25` | Source 2: demand multiplicative CV (daily retail variability) |
| `STOCH_INVENTORY_FRAC_STD` | `0.22` | Source 3: inventory/yield multiplicative CV (shrinkage, weather) |
| `STOCH_TRANSPORT_KM_STD` | `0.22` | Source 4: transport distance jitter CV (detours, traffic, loading) |
| `STOCH_K_REF_STD` | `0.20` | Source 5: Arrhenius decay rate k_ref CV (batch-to-batch biological variability) |
| `STOCH_EA_R_STD` | `0.14` | Source 5: Arrhenius activation energy Ea/R CV |
| `STOCH_ONSET_JITTER_H` | `6.0` | Source 6: scenario onset timing jitter ±hours (uniform) |
| `STOCH_THETA_NOISE_STD` | `0.15` | Source 7: policy weight THETA noise sigma (per element) |
| `STOCH_POLICY_TEMP_STD` | `0.25` | Source 8: policy-temperature LogNormal sigma in log-space (operator softmax-temperature heterogeneity) |
| `STOCH_DELAY_PROB` | `0.10` | Orthogonal to the 8 sources: telemetry lag probability per step (intermittent dropouts) |

> The defaults above are the single source of truth for the published
> 20-seed HPC benchmark calibration. They are emitted by
> `mvp.simulation.stochastic.canonical_defaults()` and asserted by the
> CI guard `tests/test_doc_stoch_defaults.py`. If you edit the table,
> either update `canonical_defaults()` to match (and rerun the HPC
> benchmark) or revert.
| `BENCHMARK_SEEDS` | `42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515` | Comma-separated seeds for multi-seed benchmark (default: 20 seeds) |

Security/runtime flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `dev` | Runtime mode (`dev`/`prod`) |
| `REQUIRE_API_KEY` | `false` in dev, `true` in prod | Require `x-api-key` on API routes |
| `APP_API_KEY` | (empty) | Global API key value when auth is enabled |
| `ALLOW_LOCAL_WITHOUT_API_KEY` | `true` in dev, `false` in prod | Allow loopback bypass for local development |
| `ENABLE_DEBUG_ROUTES` | `true` in dev, `false` in prod | Enables `/debug/*` routes |
| `WS_REQUIRE_API_KEY` | `false` in dev, `true` in prod | Require websocket API key |
| `WS_API_KEY` | falls back to `APP_API_KEY` | Websocket auth key |
| `CORS_ORIGINS` | `*` in dev, `http://localhost:5173` in prod | Comma-separated allowed origins |
| `PROTECT_DOCS` | `false` in dev, `true` outside dev | Gate `/docs`, `/redoc`, `/openapi.json` behind the API-key middleware. Production deployments should leave this on so an unauthenticated `GET /openapi.json` cannot enumerate the route schema. Disable only when docs are terminated upstream by a reverse proxy / IP allowlist. |
| `GOVERNANCE_API_KEY` | falls back to `APP_API_KEY` | Scoped key accepted for the `/governance/*` routes (in addition to `APP_API_KEY`). Use to limit blast radius of a single leaked credential. |
| `CHAIN_API_KEY`      | falls back to `APP_API_KEY` | Scoped key accepted for `POST /chain/config`. |
| `PHASE_API_KEY`      | falls back to `APP_API_KEY` | Scoped key accepted for the `/phase/*` deployment-phase routes. |
| `MCP_API_KEY`        | falls back to `APP_API_KEY` | Scoped key accepted for the `/mcp/*` JSON-RPC surface. |
| `MCP_RATE_LIMITS`    | `transport` (default) / `enabled` / `disabled` | Per-tool token-bucket policy from `pirag/configs/policy.yaml`. `transport` (default): enforced only at the public MCP HTTP/JSON-RPC boundary so the simulator's in-process registry calls bypass the bucket. `enabled` / `on` / `true`: enforce on every tool invocation including the simulator hot path. `disabled` / `off` / `false`: skip enforcement entirely. |
| `STRICT_VALIDATION`  | `1`           | `mvp/simulation/validation/validate_results.py` exits non-zero on missing tables or range violations. Set to `0` to downgrade to advisory-only for local debugging. |
| `STRICT_SMOKE`       | `1`           | `mvp/simulation/tests/test_stochastic_quick.py` exits non-zero on `BROKEN` / `NEEDS TUNING` verdicts. Set to `0` to make the smoke test advisory-only. |
| `ALLOW_MISSING_BASELINE` | `0` outside CI | When `1`, `run_regression_guard.py` treats a missing `baseline_snapshot.json` as a SKIP rather than a failure. CI sets this to `1` so the drift gate fires only on real drift, never on first-run-on-a-fresh-branch. |
| `CHAIN_REQUIRE_PRIVKEY` | `true` | Require `CHAIN_PRIVKEY` to be set before chain-anchoring code paths run. The signing key is **only** loaded from this env var; `POST /chain/config` no longer accepts it in the request body (rejected with 422). |
| `CHAIN_BEST_EFFORT`  | `false` in prod | When `false`, on-chain submission failures raise; when `true`, they log at WARN and `submit_onchain` returns `None`. Production must keep this `false` so operators do not silently believe an anchor happened when it did not. |
| `DYNAMIC_KB_FEEDBACK` | `true` (dev), `false` (published runs) | Enables the piRAG dynamic-knowledge re-ingestion loop. Disable for HPC publication runs because the loop re-ingests the agent's own actions as documents, which biases ablation comparisons. |

> Production hardening checklist: `APP_ENV=prod`, `REQUIRE_API_KEY=true`,
> `APP_API_KEY=<strong>`, `PROTECT_DOCS=true`, `ALLOW_LOCAL_WITHOUT_API_KEY=false`,
> `ENABLE_DEBUG_ROUTES=false`, `WS_REQUIRE_API_KEY=true`, `WS_API_KEY=<strong>`,
> `CORS_ORIGINS=https://your-frontend.example.com`, `CHAIN_REQUIRE_PRIVKEY=true`,
> `CHAIN_BEST_EFFORT=false`. The `.env.prod.example` file at the repo root tracks
> these settings for you.

### 2c. Start the backend

```bash
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100
```

You should see:

```
[startup] spinach CSV loaded
INFO:     Uvicorn running on http://127.0.0.1:8100
```

### 2d. Verify the backend is running

Open a new terminal and run:

```bash
curl http://127.0.0.1:8100/health
```

Expected: `{"ok":true}`

### 2e. Load sensor data

```bash
curl -X POST http://127.0.0.1:8100/case/load
```

Expected: `{"ok":true,"records":288}`

This step is required before the frontend dashboard will display data.

Browse the interactive API docs at: **http://127.0.0.1:8100/docs**

---

## 3. Frontend setup

Open a new terminal:

```bash
cd AGRI-BRAIN/agribrain/frontend
npm install
npm run dev
```

You should see:

```
VITE v7.x.x  ready
➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser to see the AGRI-BRAIN dashboard.

The frontend connects to the backend at `http://127.0.0.1:8100`.
Make sure the backend is running and data is loaded before using the dashboard.

### Frontend pages

| URL | Page | Description |
|-----|------|-------------|
| `/` | Operations | KPI cards (records, temperature, anomalies, waste rate), real-time telemetry line charts with safe/warning/critical temperature zones, spoilage & yield area chart |
| `/quality` | Quality | Circular spoilage risk gauge, shelf-life countdown timer, current sensor readings, IoT temperature/humidity charts, PINN vs ODE spoilage comparison |
| `/decisions` | Decisions | Decision timeline with action badges, filters by role and action type, search, decision analytics sidebar with pie chart, CSV export, PDF report. Each decision card has an expandable Explainability Panel showing: causal BECAUSE/WITHOUT reasoning with highlighted keywords, 5-axis context feature radar chart (compliance, forecast urgency, retrieval confidence, regulatory pressure, recovery saturation) with logit adjustment bars, categorized keyword tags (thresholds, regulations, required actions), and Merkle-rooted provenance chain with SHA-256 evidence hashes |
| `/map` | Map | Leaflet map centered on South Dakota with 4 supply chain nodes (farm, processor, cooperative, recovery) and route overlays showing cold chain, redistribution, and recovery paths |
| `/analytics` | Analytics | Executive summary with 5 hero metrics, Table 1 (cross-scenario) and Table 2 (ablation study), grouped bar charts, radar chart, method comparison, scenario deep-dive gallery with figures, carbon footprint analysis, full simulation runner |
| `/mcp-pirag` | MCP/piRAG | MCP protocol overview, context feature visualization, knowledge base browser, protocol traces, causal reasoning panel, **H3 Mechanism tab (§5.8)** — cross-seed per-channel logit-contribution stats from `decision_ledger_aggregate.json` with the supplementary Table S1 rendered live (MCP vs piRAG median Δlogit per scenario, super-additivity fraction, per-feature mean absolute contribution) |
| `/demo` | Demo | Interactive system demo with live pipeline walkthrough and agent decision theater |
| `/admin` | Admin Panel | Seven tabs: Policy (routing/carbon/SLCA parameters), Blockchain (RPC status, config), Audit (searchable log table with expandable rows), Scenarios (5 scenario cards with intensity slider), Quick Decision (role selector + instant decision), Runtime config, MCP Explorer (tool browser with 14 statically registered tools, live resource monitor with 5s auto-refresh, prompt template browser with parameter forms, live tool invocation with presets for compliance/piRAG/explain, piRAG knowledge base search with physics-informed retrieval, JSON-RPC protocol interaction log) |

### Features

- **Explainability Panel**: Each decision card on the Decisions page has a "Show explanation" button that reveals causal reasoning (BECAUSE/WITHOUT), a 5-axis context feature radar chart (ψ_0..ψ_4, the institutional context vector), categorized keyword tags from piRAG retrieval, and a Merkle-rooted provenance chain.
- **MCP Explorer**: The Admin panel's MCP tab provides an interactive tool browser, live resource monitor, prompt template expander, tool invocation console with presets, piRAG knowledge base search, and a JSON-RPC protocol interaction log.
- **H3 Mechanism panel (§5.8)**: `/mcp-pirag` → "H3 Mechanism" tab renders the cross-seed channel-attribution Table S1 directly from `decision_ledger_aggregate.json`. Use this as the centrepiece of a research demo — it shows the per-channel decomposition (MCP-derived vs piRAG-derived ψ-features) is *complementary, not redundant*, and quantifies the super-additivity fraction (the ~20 % of decisions where the integrated condition exceeds the better single channel by > 0.005 ARI-equivalent logit shift).
- **Showcase banner**: A dismissable hero banner on the Operations dashboard surfaces the three canonical paper claims (H1 integration superiority, H2 communication robustness, H3 component complementarity) with their effect sizes and links into the supporting frontend pages. Dismissal is persisted to `localStorage.ops.showcaseBanner.dismissed` so a returning operator does not see it on every reload.
- **Dark mode**: Toggle via the sun/moon icon in the header. Persists in localStorage.
- **WebSocket**: Real-time connection indicator ("Live" badge) in the header. Auto-reconnects.
- **Notifications**: Bell icon in header shows decision events from the WebSocket stream.
- **Responsive**: Sidebar collapses on mobile with bottom navigation.

### Showcase demo flow (for visiting academics or industry guests)

Five-minute walkthrough that takes a guest from headline claims to verifiable evidence:

1. **`/` (Ops dashboard)** — the showcase banner is the first thing visible. It quotes the three hypotheses with their effect sizes from the canonical 20-seed run.
2. **`/mcp-pirag` → "H3 Mechanism (§5.8)"** — click the headline link from the banner. The panel loads `decision_ledger_aggregate.json` live and renders the supplementary Table S1 plus the pooled-perturbed pooled stats (n = 23,040 agribrain-mode decisions). This is the freshest paper-grade evidence.
3. **`/demo` (interactive pipeline)** — walk through the 12-phase decision pipeline (IoT → PINN spoilage → LSTM forecast → agent dispatch → MCP tools → piRAG retrieval → ψ vector → Θ_context × ψ logit modifier → policy → action → SLCA → causal explanation → Merkle root → on-chain anchor).
4. **`/decisions`** — expand any decision card's Explainability Panel. Show the BECAUSE/WITHOUT counterfactual, the ψ radar, and the Merkle root. Copy the Merkle root and demonstrate that it can be verified independently against the on-chain `ProvenanceRegistry.sol` anchor.
5. **`/analytics`** — full benchmark tables (Table 1 cross-scenario, Table 2 ablation), per-scenario deep-dive figure gallery, paired ΔARI Wilcoxon-CI whiskers. This is the corroborating quantitative evidence.
6. **`/admin` → "Blockchain"** — surface the deployed contract addresses, the policy-anchor history, and the DAO governance flow (`propose` → `vote` → `queue` → `execute`) that update Θ_context on-chain.

The full sequence runs from a fresh terminal in under 10 minutes assuming the backend is already serving on `:8100` and the frontend on `:5173`.

---

## 4. Test all backend API endpoints

With the backend running on port 8100, test each endpoint:

### Core data endpoints

```bash
# Health check
curl http://127.0.0.1:8100/health

# Load CSV data (triggers PINN spoilage computation)
curl -X POST http://127.0.0.1:8100/case/load

# Get KPIs (records, avg temp, waste rates, etc.)
curl http://127.0.0.1:8100/kpis

# Get telemetry time-series (tempC, RH, inventory, demand)
curl http://127.0.0.1:8100/telemetry

# Get predictions (shelf_left, spoilage_risk, yield forecast)
curl http://127.0.0.1:8100/predictions
```

### Decision engine

```bash
# Make a decision (regime-aware softmax policy)
# Required fields: agent_id (str), role (str). Optional: mode, step, deterministic.
curl -X POST http://127.0.0.1:8100/decide \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"farm","role":"farm"}'

# Get the last decision
curl http://127.0.0.1:8100/last-decision

# Get decision feed
curl http://127.0.0.1:8100/decisions
```

### Policy and governance

```bash
# Read current policy
curl http://127.0.0.1:8100/governance/policy

# Update a policy parameter
curl -X POST http://127.0.0.1:8100/governance/policy \
  -H "Content-Type: application/json" \
  -d '{"carbon_per_km": 0.15}'

# Read blockchain config
curl http://127.0.0.1:8100/governance/chain
```

### Scenarios

```bash
# List available scenarios
curl http://127.0.0.1:8100/scenarios/list

# Run a scenario (heatwave, overproduction, cyber_outage, adaptive_pricing, baseline)
curl -X POST http://127.0.0.1:8100/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{"name":"heatwave","intensity":1.0}'

# Reset to baseline
curl -X POST http://127.0.0.1:8100/scenarios/reset
```

### Audit and reporting

```bash
# Get audit logs
curl http://127.0.0.1:8100/audit/logs

# Get decision memo as JSON
curl http://127.0.0.1:8100/audit/memo.json

# Download decision memo as PDF (requires reportlab)
curl http://127.0.0.1:8100/audit/memo.pdf -o memo.pdf
```

### Simulation results (via backend)

```bash
# Start simulation in background (5 scenarios × 19 modes, typically 60-90 min in deterministic mode)
curl -X POST http://127.0.0.1:8100/results/generate

# Poll progress
curl http://127.0.0.1:8100/results/status

# Fetch summary once complete
curl http://127.0.0.1:8100/results/summary

# Fetch a generated figure
curl http://127.0.0.1:8100/results/figures/fig2_heatwave.png -o fig2.png
```

### MCP Protocol (JSON-RPC 2.0)

```bash
# Initialize MCP handshake
curl -X POST http://127.0.0.1:8100/mcp/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"test-client","version":"1.0.0"}}}'

# List available tools (14 statically registered tools including pirag_query, explain, context_features, yield_query, demand_query)
curl -X POST http://127.0.0.1:8100/mcp/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

# Check compliance via MCP
curl -X POST http://127.0.0.1:8100/mcp/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"check_compliance","arguments":{"temperature":14.0,"humidity":85.0,"product_type":"spinach"}}}'

# Query the piRAG knowledge base via MCP
curl -X POST http://127.0.0.1:8100/mcp/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"pirag_query","arguments":{"query":"FDA temperature violation corrective action","k":4,"temperature":14.0,"rho":0.4}}}'

# Get a causal explanation via MCP
curl -X POST http://127.0.0.1:8100/mcp/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"explain","arguments":{"action":"local_redistribute","role":"farm","rho":0.4,"temperature":14.0,"scenario":"heatwave"}}}'

# List MCP resources (telemetry, quality, context features)
curl http://127.0.0.1:8100/mcp/resources

# List MCP prompts (regulatory, SOP, governance query templates)
curl http://127.0.0.1:8100/mcp/prompts
```

### piRAG Knowledge Base

```bash
# Query the knowledge base directly
# Required field: question (str). Optional: k (int), anchor_on_chain (bool).
curl -X POST http://127.0.0.1:8100/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"FDA cold chain requirements for spinach temperature excursion","k":4}'

# Ingest a custom document
curl -X POST http://127.0.0.1:8100/rag/ingest \
  -H "Content-Type: application/json" \
  -d '[{"id":"custom_doc","text":"Custom regulatory guidance for temperature monitoring...","metadata":{"source":"manual"}}]'
```

### Debug

```bash
# List all registered routes
curl http://127.0.0.1:8100/debug/routes
```

---

## 5. Run all scenarios interactively

Using the frontend dashboard:

1. Navigate to **Admin Panel** → **Scenarios** tab
2. Select a scenario card (e.g., "Heatwave")
3. Adjust the intensity slider and click **Run**
4. Switch to **Operations** or **Quality** pages to see updated telemetry
5. Go to **Decisions** page and click **Take Decision**
6. Click **Show explanation** on the decision card to see the causal reasoning, context feature radar, keywords, and provenance chain
7. View the audit trail under **Admin** → **Audit** tab
8. Navigate to **Admin** → **MCP** tab to explore tools, resources, prompts, and invoke them live
9. Try the **piRAG Search** sub-tab to query the knowledge base with physics-informed retrieval

Or via the API (run each scenario and make a decision):

```bash
for scenario in heatwave overproduction cyber_outage adaptive_pricing baseline; do
  echo "=== $scenario ==="
  curl -s -X POST http://127.0.0.1:8100/scenarios/run \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$scenario\",\"intensity\":1.0}"
  echo ""
  curl -s -X POST http://127.0.0.1:8100/decide \
    -H "Content-Type: application/json" \
    -d '{"agent_id":"farm","role":"farm"}'
  echo -e "\n"
done
```

---

## 6. Run standalone simulation and generate figures

The standalone simulation runs all 5 scenarios × 19 modes (95 episodes
single-seed: 8 canonical modes + 11 §4.7 sensitivity ablations) and
produces publication-quality results. The state vector phi(s) is
10-dimensional: six perception features (freshness, inventory pressure,
demand point forecast, thermal stress, spoilage urgency, interaction),
three forecast-channel features (supply point, supply uncertainty,
demand uncertainty) that treat the supply and demand forecasters
symmetrically, and a demand-volatility Bollinger z-score
(``price_signal``) that proxies market pressure. The context vector
psi is 5-dimensional and carries the institutional / coordination
signals from MCP and piRAG.

```bash
cd AGRI-BRAIN/mvp/simulation

# Generate results (CSV tables)
python generate_results.py

# Generate figures (PDF + PNG)
python generate_figures.py
```

### 6a. Re-render figures from cached results (no simulator run)

Once a 20-seed HPC run has completed and the per-seed JSONs +
`benchmark_summary.json` + `benchmark_significance.json` are on
disk, you can re-render every figure without re-running the
simulator. Use this whenever you want to tweak a figure's styling
(line weights, colors, axis ranges) without paying the ~80-minute
`run_all()` cost:

```bash
cd AGRI-BRAIN
python mvp/simulation/regenerate_figures_from_cache.py
python mvp/simulation/analysis/build_artifact_manifest.py
```

The script reads:

  * `mvp/simulation/results/benchmark_summary.json` — 20-seed
    bootstrap means / CIs for every scalar metric (drives fig 6 /
    fig 7 / fig 8 panel B / fig 9 / fig 10).
  * `mvp/simulation/results/benchmark_seeds/seed_*.json` — per-seed
    envelope `{seed, scenarios, traces}`. Per-step `traces` arrays
    drive every line plot and window-aggregated panel
    (fig 2 / 3 / 4 / 5 / 8 panel A). The figure code's own
    `_load_per_seed_traces` helper still consumes the full
    multi-seed envelope where it needs cross-seed CIs (fig 2 panel
    D mean line, fig 4 panels B/C/D bars).

By default the script picks seed 42 as the "single-seed
representative" for the line-plot panels; if seed 42 isn't on disk
it falls back to the smallest available seed. Total runtime:
seconds, not hours.

**Trace-fields contract.** `mvp/simulation/benchmarks/run_single_seed.py`
defines `TRACE_FIELDS` — the per-step fields each seed dumps. As
of 2026-05 it covers every field the figure code reads, so a
fresh HPC run produces a self-contained cache. If you re-render
against an older HPC run that pre-dates a particular trace field,
the affected figure will fail with a `KeyError` and the script
will report which trace was missing; the rest still re-render.

### Output files

Results are saved to `mvp/simulation/results/`:

| File                   | Description                                       |
|------------------------|---------------------------------------------------|
| `table1_summary.csv`     | Per-scenario metrics (3 methods x 5 scenarios)              |
| `table2_ablation.csv`    | Full ablation study (19 modes x 5 scenarios — 8 canonical + 11 §4.7 sensitivity ablations) |
| `benchmark_summary.json` | Multi-seed benchmark means/std/CI                           |
| `benchmark_significance.json` | Permutation-test p-values and effect sizes            |
| `stress_summary.json`    | Stress-suite per-scenario robustness output                 |
| `stress_degradation.csv` | Delta metrics under stressors                               |
| `artifact_manifest.json` | SHA-256 reproducibility manifest                            |
| `traces_*.json`          | Decision traces with keywords, causal reasoning, provenance |
| `mcp_protocol_*.json`    | Genuine MCP JSON-RPC interaction logs                       |
| `fig2_heatwave.png/pdf`  | Heatwave scenario: env exposure, spoilage trajectory, AgriBrain action probs, per-step (1-waste)*SLCA policy-quality factor (2x2) |
| `fig3_overproduction.png/pdf` | Overproduction & reverse logistics: inventory vs demand, waste reduction, RLE trajectory, SLCA components (2x2) |
| `fig4_cyber.png/pdf`     | Cyber outage: ARI per step (a), action distribution shift pre/during outage (b), behavior shift = per-method reroute rate (c), outage-window levels = absolute ARI / Waste / Service during the outage per method (d). Panels B/C/D prefer multi-seed cross-seed-SE error bars when per-seed action/waste traces are present in `benchmark_seeds/`; fall back to single-seed Wald-binomial / step-level SE locally. Causality reads top-down + left-right. (2x2, redesigned 2026-05; panel D switched from pre/during deltas to absolute during-window levels in 2026-05 to remove a saturation artefact that inverted the Service ranking.) |
| `fig5_pricing.png/pdf`   | Adaptive pricing: demand + Bollinger triggers, routing distribution, price equity, per-step reward comparison across modes (2x2) |
| `fig6_cross.png/pdf`     | Cross-scenario performance comparison: ARI / RLE / Waste / SLCA grouped bars across the 4 stress scenarios x 3 methods (2x2) |
| `fig7_ablation.png/pdf`  | Ablation study: ARI / Waste / RLE grouped bars across 4 stress scenarios × 8 canonical modes (1x3) |
| `fig8_green_ai.png/pdf`  | Green AI and carbon footprint: cumulative CO2 trajectory + total carbon bar chart (1x2) |
| `fig9_robustness.png/pdf` | Performance gain over baselines and context channel: (a) Cohen's d heatmap (5 scenarios × 5 baselines), (b) % ARI improvement forest plot with min-max range whiskers, (c) **context-influence rate** per scenario for {AgriBrain, MCP only, piRAG only} with paired-bootstrap CIs (1x3). The legacy honor-rate metric is retained alongside in `benchmark_summary.json` for the supplementary methods table. |
| `fig10_latency_quality_frontier.png/pdf` | Latency vs ARI frontier: (a) lightweight methods, (b) context-aware methods with broken x-axis + overhead arrow, (c) **paired ΔARI vs No Context** bars per scenario (added 2026-05) with Wilcoxon signed-rank 95% CI whiskers from `benchmark_significance.json`. (1x3.) |

Each figure is also saved as PDF for LaTeX inclusion.

---

## 6b. Run the 20-seed benchmark on HPC (SLURM)

**Posture:** the HPC pipeline runs the **stochastic** benchmark with
**20 seeds** by default and only ever publishes stochastic numbers.
`DETERMINISTIC_MODE=false` is enforced at three layers (orchestrator,
seed array task, and aggregation task); a stale `true` in the cluster
env aborts the submit. The deterministic regression-guard snapshot is a
separate manual step (see "Regenerate the regression-guard snapshot"
below) and is intentionally NOT part of the published-results path.

The full benchmark is 5 scenarios × 19 modes × 20 seeds = 1,900 episodes
(8 canonical paper modes + 11 §4.7 sensitivity ablations:
`agribrain_cold_start`, `agribrain_pert_{10,25,50}`,
`agribrain_pert_{10,25,50}_static`, `agribrain_no_bonus`,
`agribrain_theta_pert_{10,25,50}`). Aggregation, stress suite, figures,
explainability metrics, and the paper-evidence pipeline run in the
single dependent aggregator job. Three scripts live in `hpc/`:

| Script | Role |
|---|---|
| `hpc/hpc_run.sh` | Orchestrator run on the login node. Sets up `.venv`, computes `RUN_TAG`, submits the seed array and the dependent aggregation job. |
| `hpc/hpc_seed.sh` | SLURM array task (`--array=0-19`). One seed per task. Writes `mvp/simulation/results/benchmark_seeds/<RUN_TAG>/seed_<N>.json`. 6 h / 8 GB / 4 CPU per task. |
| `hpc/hpc_aggregate.sh` | Single SLURM task chained via `--dependency=afterok`. Runs Stages 1-10 (base tables, validation, both aggregators, stress suite, figures, paper evidence, manifest). 8 h / 16 GB / 4 CPU. |

### Submit

From the HPC login node, in the repo root:

```bash
bash hpc/hpc_run.sh
```

That script will:

1. Create `.venv` if absent; `pip install -e agribrain/backend`.
2. Run the login-node policy-shape load assertion (fails fast if the resolver
   pulled a broken package combination).
3. Compute `RUN_TAG=$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M)`.
4. Submit `hpc/hpc_seed.sh`, then `hpc/hpc_aggregate.sh` with
   `--dependency=afterok:<seed_job>` so aggregation runs only if every
   seed task succeeded.

### Monitor

```bash
squeue -u $USER
tail -f logs/seed_<job_id>_0.out
tail -f logs/aggregate_<job_id>.out
```

### Output

On success the aggregator writes `hpc_results_<RUN_TAG>.tar.gz` in the
repo root, including the canonical summary/significance JSONs, Table 1
and Table 2 CSVs, stress suite outputs, Figure 2-10 PNG+PDF, the
explainability assessment metrics (`explainability_metrics.json`, the
§4.10 100/100/100 numbers), the artifact manifest, and the full
`decision_ledger/` directory. Transfer back with:

```bash
scp <hpc-host>:$PWD/hpc_results_<RUN_TAG>.tar.gz .
tar xzf hpc_results_<RUN_TAG>.tar.gz
```

### Regenerate the regression-guard snapshot (deterministic, manual, post-HPC)

`mvp/simulation/baseline_snapshot.json` is a *deterministic* fixture
that catches code-drift on later deterministic runs; it is not a
stochastic publication artefact. The regression guard skips itself in
stochastic mode (the default), so the snapshot only matters when a
maintainer explicitly runs `DETERMINISTIC_MODE=true`. Regenerate after
each HPC publication run so the deterministic baseline matches the
shipped code, then commit the resulting JSON:

```bash
DETERMINISTIC_MODE=true REGRESSION_GUARD_INIT=true \
    python -m mvp.simulation.validation.run_regression_guard

git add mvp/simulation/baseline_snapshot.json
git commit -m "regenerate baseline_snapshot.json after HPC run <RUN_TAG>"
```

### Wall time

Each array task is ~2 h on a modern HPC node (5× laptop speedup assumed,
45 cells per seed × ~159 s/cell). Array wall-clock is scheduler-limited
but typically 2-4 h with all 20 tasks running concurrently. Aggregation
runs ~3-4 h. End-to-end: 6-10 h from `bash hpc/hpc_run.sh` to the archive.

### Pre-HPC verification

Before submitting, run the pre-HPC check locally:

```bash
# default CI-speed tests
cd agribrain/backend && pytest -q

# Opt-in full mode x scenario matrix (slow; ~10 min). The default
# `addopts = "-m 'not slow'"` in pyproject.toml hides slow tests in
# the standard `pytest` invocation; to run them you have to override
# the addopts so the explicit `-m slow` selector wins.
pytest --override-ini="addopts=" -m slow
```

The canonical pre-HPC verification lives in
`mvp/simulation/validation/`. Run the validator and manifest verifier
before submitting:

```bash
python mvp/simulation/validation/validate_results.py        # range checks (strict by default)
python mvp/simulation/analysis/verify_manifest.py --strict-commit --allow-missing
python mvp/simulation/validation/validate_publication_artifacts.py
```

---

## 7. On-chain features (recommended)

The §1 / §4.13 claim "blockchain verification of every routing decision"
requires a running EVM. The localhost Hardhat node below is enough to
satisfy that claim end-to-end on a single workstation. A fresh clone
that follows §2 + §3 produces decisions that show **"On-chain anchor
not attempted — chain not configured"** in the Explainability panel;
following the steps in this section flips them to a real `0x…`
transaction hash.

### 7a. One-command quickstart

From the repo root:

```bash
bash agribrain/contracts/hardhat/scripts/start_localhost_chain.sh
```

That script (see file for details) installs Hardhat dependencies,
starts a node on `127.0.0.1:8545`, deploys all six contracts via
`scripts/deploy.js`, writes the addresses to
`agribrain/backend/runtime/chain/deployed-addresses.localhost.json`
(the backend auto-loads them), and prints the `CHAIN_PRIVKEY` env var
the backend needs.

Then **export the private key the script printed** and start the
backend:

```bash
export CHAIN_PRIVKEY=0xac0974…  # printed by the script
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100
```

The next decision you trigger (via the Admin Quick Decision tab or
`POST /decide`) anchors on chain. Confirm with:

```bash
curl -s http://127.0.0.1:8100/last-decision | jq .memo.tx_hash
# -> "0x…"  (real hash; not "0x0" and not null)
```

### 7b. Manual setup (same effect, more visible)

```bash
cd AGRI-BRAIN/agribrain/contracts/hardhat
npm install                     # one-time
npx hardhat node                # leave running

# In a second terminal, from the repo root:
cd AGRI-BRAIN/agribrain/contracts/hardhat
npx hardhat run scripts/deploy.js --network localhost
# -> writes deployed-addresses.localhost.json into backend/runtime/chain/
```

Configure the chain in the Admin panel (Blockchain tab) or via:

```bash
curl -X POST http://127.0.0.1:8100/governance/chain \
  -H "Content-Type: application/json" \
  -d '{"rpc":"http://127.0.0.1:8545","chain_id":31337}'
```

### Slither (optional, match CI locally)

The GitHub Actions **contract-analysis** job runs [Slither](https://github.com/crytic/slither) on `agribrain/contracts/hardhat` with `--exclude-informational --exclude-low` and `fail-on: medium` (version **0.11.4** in CI; matches the `slither-version` pin in `.github/workflows/ci.yml`). To reproduce outside CI, install Slither in a Python environment (or use a container image that includes it), then from the Hardhat directory:

```bash
cd agribrain/contracts/hardhat
npm install
slither . --exclude-informational --exclude-low
```

If this reports nothing at medium+ severity, you align with the filtered CI gate. Informational and low findings are excluded on purpose; see `.github/workflows/ci.yml` for the exact flags.

### `chain_query` tool and the MCP Reliability figure

Figure 9(b) of the paper reports envelope vs tool error counts across a full benchmark run. One non-trivial source of tool errors is `chain_query`: this tool reads the live FastAPI app state (the running REST service's in-memory `state["log"]` list) to return the most recent on-chain routing decisions. Under the simulator-benchmark path (`mvp/simulation/generate_results.py`), the FastAPI app module is importable but *not running*, so `state["log"]` is never populated. The tool correctly surfaces this as a structured `_status: "error"` payload with `_error_kind: "state_unavailable"`, which the protocol-recorder counts as a tool-level error in fig9(b). This is by design: `chain_query` is intended for the live REST deployment path, where the FastAPI app populates the audit trail as decisions happen. In the benchmark it is correctly reporting that the live trail is not reachable from the simulator subprocess. The paper's fig9(b) caption should note this explicitly.

---

## 8. Complete walkthrough (all-in-one)

Run everything end-to-end in order:

```bash
# --- Terminal 1: Backend ---
cd AGRI-BRAIN
source .venv/bin/activate  # if using venv
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100

# --- Terminal 2: Frontend ---
cd AGRI-BRAIN/agribrain/frontend
npm install && npm run dev

# --- Terminal 3: Tests and simulation ---
cd AGRI-BRAIN

# 1. Verify backend health
curl http://127.0.0.1:8100/health

# 2. Load data
curl -X POST http://127.0.0.1:8100/case/load

# 3. Check KPIs
curl http://127.0.0.1:8100/kpis

# 4. Run a decision
curl -X POST http://127.0.0.1:8100/decide \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"farm","role":"farm"}'

# 5. Run all scenarios
for s in heatwave overproduction cyber_outage adaptive_pricing baseline; do
  curl -s -X POST http://127.0.0.1:8100/scenarios/run \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$s\"}"
  echo " -> $s applied"
done

# 6. Generate simulation results
curl -X POST http://127.0.0.1:8100/results/generate

# 7. Generate standalone figures
cd mvp/simulation
python generate_results.py
python generate_figures.py

# 8. Open browser: http://localhost:5173
```

---

## 9. Reproduce core research outputs (one command)

From repo root:

```bash
python mvp/simulation/reproduce_core.py
```

This runs, in order:
- results generation
- validation checks
- regression guard check (initialize once with `REGRESSION_GUARD_INIT=true`)
- stress robustness suite (noise, missing data, telemetry delay, MCP faults)
- within-trace temporal stability check (early/mid/late thirds of the same trace; not external validity in the methodological sense)
- per-seed benchmark runs (`benchmarks/run_single_seed.py`)
- canonical multi-seed aggregation (`benchmarks/aggregate_seeds.py`) with CIs + paired stats
- figure generation
- paper evidence export
- artifact manifest (SHA-256 hashes + exact git commit for reproducibility)
- publication artifact schema validation (`validation/validate_publication_artifacts.py`)

For publication reporting policy, see:
- `docs/METHODS_REPRO_APPENDIX.md`
- `docs/STATISTICAL_METHODS.md`
- `docs/CLAIMS_TO_EVIDENCE.md`

A pre-generated lockfile ships at `agribrain/backend/requirements-lock.txt`.
For paper-grade reproducibility install with that file (see §2b above);
to regenerate from scratch, follow `docs/RELEASE.md` step 2.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Make sure the editable install succeeded: `pip install -e "agribrain/backend[dev]"`. Since the 2026-05 packaging fix the `--app-dir` flag is **not** required and the uvicorn invocation should be just `python -m uvicorn src.app:API --port 8100`. |
| `ModuleNotFoundError: No module named 'pirag'` | Same: `pip install -e "agribrain/backend[dev]"`. The packaging fix exposes both `src` and `pirag` to the editable install. |
| Port 8100 already in use | Kill the existing process: `lsof -ti:8100 \| xargs kill` |
| Frontend CORS errors | Ensure backend is on port 8100 and frontend on port 5173 |
| Charts show skeleton loaders | Ensure `/case/load` was called first |
| Leaflet map tiles not loading | Check internet connection; map requires OpenStreetMap tile access |
| Dark mode not working | Clear localStorage (`localStorage.removeItem('agri-brain-theme')`) and refresh |
| WebSocket disconnected | Ensure backend is running; the header badge shows "Live" or "Offline" |
| `reportlab` not found (PDF route) | `pip install reportlab` |
| `matplotlib` not found (figures) | `pip install matplotlib` |
| Figures directory empty | Run `python generate_results.py` then `python generate_figures.py` |
| Hardhat errors | Run `npm install` in `contracts/hardhat/` first |

---

## Port reference

| Service       | Port  | URL                        |
|---------------|-------|----------------------------|
| Backend API   | 8100  | http://127.0.0.1:8100      |
| Frontend      | 5173  | http://localhost:5173       |
| Swagger docs  | 8100  | http://127.0.0.1:8100/docs |
| Hardhat node  | 8545  | http://127.0.0.1:8545      |
