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
git clone <repo-url> AGRI-BRAIN
cd AGRI-BRAIN
```

The repository layout:

```
AGRI-BRAIN/
├── README.md               # Project overview and quick start
├── HOW_TO_RUN.md           # This file
├── .gitignore
├── agri-brain-mvp-1.0.0/
│   ├── backend/            # FastAPI backend (port 8100)
│   │   ├── src/            # Application code (app.py, models/, routers/, agents/)
│   │   ├── pirag/          # PiRAG integration
│   │   │   ├── ingestion/  # Document parser, TF-IDF embedder, vector store
│   │   │   ├── inference/  # LLM abstraction (template + API engines)
│   │   │   ├── knowledge_base/  # SOPs, regulatory docs, SLCA guides
│   │   │   └── mcp/        # MCP tool server (compliance, slca, chain query)
│   │   ├── pyproject.toml
│   │   └── static/         # Swagger branding assets
│   ├── frontend/           # React + Vite dashboard (port 5173)
│   └── contracts/          # Solidity smart contracts (Hardhat)
├── mvp/
│   └── simulation/         # Standalone simulation & figure scripts
│       ├── generate_results.py
│       ├── generate_figures.py
│       └── results/        # Output CSVs and figures
└── pirag_patch_v1_fullcode/ # Standalone PiRAG reference code
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

### 2c. Start the backend

```bash
cd agri-brain-mvp-1.0.0/backend
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100 --reload
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
Make sure the backend is running before using the dashboard.

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
curl -X POST http://127.0.0.1:8100/decide \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"farm","role":"grower","step":0}'

# Get the last decision
curl http://127.0.0.1:8100/last-decision
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
# Run full simulation (5 scenarios x 5 modes, ~1 min)
curl -X POST http://127.0.0.1:8100/results/generate

# Fetch a generated figure
curl http://127.0.0.1:8100/results/figures/fig2_heatwave.png -o fig2.png
```

### Debug

```bash
# List all registered routes
curl http://127.0.0.1:8100/debug/routes
```

---

## 5. Run all scenarios interactively

Using the frontend dashboard:

1. Navigate to the **Scenarios** tab
2. Select a scenario from the list (e.g., "Climate-Induced Heatwave")
3. Click **Run** — the backend applies perturbations to the sensor data
4. Switch to **Ops** or **Quality** tabs to see updated telemetry and predictions
5. Go to **Decisions** tab and click **Take Decision** to run the policy engine
6. View the audit trail in the **Admin** tab

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
    -d '{"agent_id":"farm","role":"grower","step":0}'
  echo -e "\n"
done
```

---

## 6. Run standalone simulation and generate figures

The standalone simulation runs all 5 scenarios x 5 modes (50 episodes each)
and produces publication-quality results.

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
| `table1_summary.csv`   | Per-scenario metrics across all 5 modes           |
| `table2_ablation.csv`  | Ablation study (agribrain vs no_pinn, no_slca)    |
| `fig2_heatwave.png`    | Heatwave scenario comparison                      |
| `fig3_reverse.png`     | Overproduction (reverse logistics) comparison     |
| `fig4_cyber.png`       | Cyber outage scenario comparison                  |
| `fig5_pricing.png`     | Adaptive pricing scenario comparison              |
| `fig6_cross.png`       | Cross-scenario radar chart                        |
| `fig7_ablation.png`    | Ablation bar chart                                |
| `fig8_green.png`       | Green footprint analysis                          |

Each figure is also saved as PDF for LaTeX inclusion.

### Expected metric ordering

For each scenario, the modes should rank approximately:

```
agribrain > no_pinn > hybrid_rl > no_slca > static
```

in terms of ARI (AGRI-BRAIN Resilience Index) and SLCA composite scores.

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

The deployed addresses are auto-synced to the backend via
`backend/runtime/chain/deployed-addresses.localhost.json`.

Configure the chain in the Admin panel or via:

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
cd agri-brain-mvp-1.0.0/backend
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100 --reload

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
  -d '{"agent_id":"farm","role":"grower"}'

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

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run uvicorn from `agri-brain-mvp-1.0.0/backend/` directory |
| `ModuleNotFoundError: No module named 'pirag'` | Run `pip install -e agri-brain-mvp-1.0.0/backend` |
| Port 8100 already in use | Kill the existing process: `lsof -ti:8100 \| xargs kill` |
| Frontend CORS errors | Ensure backend is on port 8100, not 8111 |
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
