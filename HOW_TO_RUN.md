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

The repository layout:

```
AGRI-BRAIN/
├── README.md
├── HOW_TO_RUN.md
├── docs/screenshots/           # Frontend screenshots
├── agri-brain-mvp-1.0.0/
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
│       ├── stochastic.py       # 7-source stochastic perturbation engine
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

```bash
pip install -e agri-brain-mvp-1.0.0/backend
```

This installs all dependencies listed in `pyproject.toml`:
fastapi, uvicorn, pydantic, numpy, pandas, matplotlib, reportlab, orjson, requests, web3, python-multipart, pyyaml.

### 2b-extra. Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_METHOD` | `lstm` | Demand forecaster: `lstm` (numpy LSTM) or `holt_winters` |
| `ONLINE_LEARNING` | `false` | Enable REINFORCE policy gradient updates |
| `LLM_PROVIDER` | `template` | RAG answer engine: `template` or `api` |
| `DATA_CSV` | (auto) | Override path to spinach sensor CSV |
| `RAG_CONTEXT_ENABLED` | `true` | Enable MCP/piRAG context integration in agribrain mode |
| `DETERMINISTIC_MODE` | `false` | `true` = exact reproducibility (audit), `false` = 7-source stochastic perturbations |
| `STOCH_TEMP_STD_C` | `1.5` | Source 1: temperature sensor noise sigma (°C) |
| `STOCH_RH_STD` | `5.0` | Source 1: humidity sensor noise sigma (%) |
| `STOCH_DEMAND_FRAC_STD` | `0.18` | Source 2: demand multiplicative CV (daily retail variability) |
| `STOCH_INVENTORY_FRAC_STD` | `0.15` | Source 3: inventory/yield multiplicative CV (shrinkage, weather) |
| `STOCH_TRANSPORT_KM_STD` | `0.15` | Source 4: transport distance jitter CV (detours, traffic, loading) |
| `STOCH_K_REF_STD` | `0.15` | Source 5: Arrhenius decay rate k_ref CV (batch-to-batch biological variability) |
| `STOCH_EA_R_STD` | `0.10` | Source 5: Arrhenius activation energy Ea/R CV |
| `STOCH_ONSET_JITTER_H` | `4.0` | Source 6: scenario onset timing jitter ±hours (uniform) |
| `STOCH_THETA_NOISE_STD` | `0.03` | Source 7: policy weight THETA noise sigma (per element) |
| `STOCH_DELAY_PROB` | `0.05` | Telemetry lag probability per step (intermittent dropouts) |
| `BENCHMARK_SEEDS` | `42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515` | Comma-separated seeds for multi-seed benchmark (default: 20 seeds) |

Security/runtime flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `dev` | Runtime mode (`dev`/`prod`) |
| `REQUIRE_API_KEY` | `false` in dev | Require `x-api-key` on API routes |
| `APP_API_KEY` | (empty) | API key value when auth is enabled |
| `ALLOW_LOCAL_WITHOUT_API_KEY` | `true` in dev | Allow loopback bypass for local development |
| `ENABLE_DEBUG_ROUTES` | `true` in dev | Enables `/debug/*` routes |
| `WS_REQUIRE_API_KEY` | `false` in dev | Require websocket API key |
| `WS_API_KEY` | falls back to `APP_API_KEY` | Websocket auth key |
| `CORS_ORIGINS` | `*` in dev | Comma-separated allowed origins |

### 2c. Start the backend

```bash
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100 --app-dir agri-brain-mvp-1.0.0/backend
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
cd AGRI-BRAIN/agri-brain-mvp-1.0.0/frontend
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
| `/decisions` | Decisions | Decision timeline with action badges, filters by role and action type, search, decision analytics sidebar with pie chart, CSV export, PDF report. Each decision card has an expandable Explainability Panel showing: causal BECAUSE/WITHOUT reasoning with highlighted keywords, 6-axis context feature radar chart (compliance, forecast, retrieval, regulatory, recovery saturation, supply uncertainty) with logit adjustment bars, categorized keyword tags (thresholds, regulations, required actions), and Merkle-rooted provenance chain with SHA-256 evidence hashes |
| `/map` | Map | Leaflet map centered on South Dakota with 4 supply chain nodes (farm, processor, cooperative, recovery) and route overlays showing cold chain, redistribution, and recovery paths |
| `/analytics` | Analytics | Executive summary with 5 hero metrics, Table 1 (cross-scenario) and Table 2 (ablation study), grouped bar charts, radar chart, method comparison, scenario deep-dive gallery with figures, carbon footprint analysis, full simulation runner |
| `/mcp-pirag` | MCP/piRAG | MCP protocol overview, context feature visualization, knowledge base browser, protocol traces, causal reasoning panel |
| `/demo` | Demo | Interactive system demo with live pipeline walkthrough and agent decision theater |
| `/admin` | Admin Panel | Seven tabs: Policy (routing/carbon/SLCA parameters), Blockchain (RPC status, config), Audit (searchable log table with expandable rows), Scenarios (5 scenario cards with intensity slider), Quick Decision (role selector + instant decision), Runtime config, MCP Explorer (tool browser with 13 statically registered tools, live resource monitor with 5s auto-refresh, prompt template browser with parameter forms, live tool invocation with presets for compliance/piRAG/explain, piRAG knowledge base search with physics-informed retrieval, JSON-RPC protocol interaction log) |

### Features

- **Explainability Panel**: Each decision card on the Decisions page has a "Show explanation" button that reveals causal reasoning (BECAUSE/WITHOUT), a 6-axis context feature radar chart (ψ_0..ψ_5, including Path B supply uncertainty), categorized keyword tags from piRAG retrieval, and a Merkle-rooted provenance chain.
- **MCP Explorer**: The Admin panel's MCP tab provides an interactive tool browser, live resource monitor, prompt template expander, tool invocation console with presets, piRAG knowledge base search, and a JSON-RPC protocol interaction log.
- **Dark mode**: Toggle via the sun/moon icon in the header. Persists in localStorage.
- **WebSocket**: Real-time connection indicator ("Live" badge) in the header. Auto-reconnects.
- **Notifications**: Bell icon in header shows decision events from the WebSocket stream.
- **Responsive**: Sidebar collapses on mobile with bottom navigation.

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
# Start simulation in background (5 scenarios x 9 modes, typically 4-15 min)
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

# List available tools (13 static tools including pirag_query, explain, context_features, yield_query)
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

The standalone simulation runs all 5 scenarios x 9 modes (45 episodes)
and produces publication-quality results. The ninth mode is the Path B
ablation (`no_yield`) that suppresses the ψ_5 supply-uncertainty feature
while keeping the rest of the context pipeline intact, so the
agribrain-vs-no_yield delta isolates the contribution of the Holt-Winters
yield forecaster to the routing policy.

```bash
cd AGRI-BRAIN/mvp/simulation

# Generate results (CSV tables)
python generate_results.py

# Generate figures (PDF + PNG)
python generate_figures.py
```

### Output files

Results are saved to `mvp/simulation/results/`:

| File                   | Description                                       |
|------------------------|---------------------------------------------------|
| `table1_summary.csv`     | Per-scenario metrics (3 methods x 5 scenarios)              |
| `table2_ablation.csv`    | Full ablation study (9 modes x 5 scenarios, including no_yield) |
| `benchmark_summary.json` | Multi-seed benchmark means/std/CI                           |
| `benchmark_significance.json` | Permutation-test p-values and effect sizes            |
| `stress_summary.json`    | Stress-suite per-scenario robustness output                 |
| `stress_degradation.csv` | Delta metrics under stressors                               |
| `artifact_manifest.json` | SHA-256 reproducibility manifest                            |
| `traces_*.json`          | Decision traces with keywords, causal reasoning, provenance |
| `mcp_protocol_*.json`    | Genuine MCP JSON-RPC interaction logs                       |
| `fig2_heatwave.png/pdf`  | Heatwave scenario comparison (2x2 panel)                    |
| `fig3_overproduction.png/pdf` | Overproduction scenario comparison (2x2 panel)         |
| `fig4_cyber.png/pdf`     | Cyber outage scenario comparison (1x3 panel)                |
| `fig5_pricing.png/pdf`   | Adaptive pricing scenario comparison (2x2 panel)            |
| `fig6_cross.png/pdf`     | Cross-scenario performance comparison (2x2 grouped bars)    |
| `fig7_ablation.png/pdf`  | Ablation study (1x3 grouped bars, 9 modes)                  |
| `fig8_green_ai.png/pdf`  | Green AI and carbon footprint (1x2 panel)                   |
| `fig9_mcp_pirag_robustness.png/pdf` | MCP/piRAG robustness and benchmark CIs         |
| `fig10_latency_quality_frontier.png/pdf` | Quality-latency operational frontier       |

Each figure is also saved as PDF for LaTeX inclusion.

---

## 6b. Run the 20-seed benchmark on HPC (SLURM)

The stochastic multi-seed benchmark (5 scenarios × 9 modes × 20 seeds =
900 episodes, plus aggregation, stress suite, figures, and the paper
evidence pipeline) is built for a SLURM cluster. Three scripts live at
the repo root:

| Script | Role |
|---|---|
| `hpc_run.sh` | Orchestrator run on the login node. Sets up `.venv`, computes `RUN_TAG`, submits the seed array and the dependent aggregation job. |
| `hpc_seed.sh` | SLURM array task (`--array=0-19`). One seed per task. Writes `mvp/simulation/results/benchmark_seeds/<RUN_TAG>/seed_<N>.json`. 6 h / 8 GB / 4 CPU per task. |
| `hpc_aggregate.sh` | Single SLURM task chained via `--dependency=afterok`. Runs Stages 1-10 (base tables, validation, both aggregators, stress suite, figures, paper evidence, manifest). 8 h / 16 GB / 4 CPU. |

### Submit

From the HPC login node, in the repo root:

```bash
bash hpc_run.sh
```

That script will:

1. Create `.venv` if absent; `pip install -e agri-brain-mvp-1.0.0/backend`.
2. Run the login-node Path B load assertion (fails fast if the resolver
   pulled a broken package combination).
3. Compute `RUN_TAG=$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M)`.
4. Submit `hpc_seed.sh`, then `hpc_aggregate.sh` with
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
and Table 2 CSVs, stress suite outputs, Figure 2-10 PNG+PDF, artifact
manifest, and the per-seed JSON directory. Transfer back with:

```bash
scp <hpc-host>:$PWD/hpc_results_<RUN_TAG>.tar.gz .
tar xzf hpc_results_<RUN_TAG>.tar.gz
```

### Wall time

Each array task is ~2 h on a modern HPC node (5× laptop speedup assumed,
45 cells per seed × ~159 s/cell). Array wall-clock is scheduler-limited
but typically 2-4 h with all 20 tasks running concurrently. Aggregation
runs ~3-4 h. End-to-end: 6-10 h from `bash hpc_run.sh` to the archive.

### Pre-HPC verification

Before submitting, run the pre-HPC check locally:

```bash
# default CI-speed tests
cd agri-brain-mvp-1.0.0/backend && pytest -q

# opt-in full mode x scenario matrix (slow; ~10 min)
pytest -m slow
```

See `docs/path_b/final_pre_hpc_check_2026-04-22.md` for the 42-check
report recorded at the last successful verification.

---

## 7. On-chain features (optional)

If you want to test blockchain integration with Hardhat:

```bash
cd AGRI-BRAIN/agri-brain-mvp-1.0.0/contracts/hardhat

# Install Hardhat dependencies
npm install

# Start a local Hardhat node
npx hardhat node

# In another terminal, deploy contracts
npx hardhat run scripts/deploy.js --network localhost
```

Configure the chain in the Admin panel (Blockchain tab) or via:

```bash
curl -X POST http://127.0.0.1:8100/governance/chain \
  -H "Content-Type: application/json" \
  -d '{"rpc":"http://127.0.0.1:8545","chain_id":31337}'
```

---

## 8. Complete walkthrough (all-in-one)

Run everything end-to-end in order:

```bash
# --- Terminal 1: Backend ---
cd AGRI-BRAIN
source .venv/bin/activate  # if using venv
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100 --app-dir agri-brain-mvp-1.0.0/backend

# --- Terminal 2: Frontend ---
cd AGRI-BRAIN/agri-brain-mvp-1.0.0/frontend
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
- external validity holdout check (early/mid/late windows)
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

Recommended environment lock (for archival reproducibility):

```bash
python -m pip freeze > requirements-lock.txt
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Use `--app-dir agri-brain-mvp-1.0.0/backend` flag or run uvicorn from that directory |
| `ModuleNotFoundError: No module named 'pirag'` | Run `pip install -e agri-brain-mvp-1.0.0/backend` |
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
