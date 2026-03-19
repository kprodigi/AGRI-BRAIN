# AGRI-BRAIN

An adaptive supply-chain intelligence system combining PINN-based spoilage
prediction, LSTM demand forecasting, Social Life-Cycle Assessment (SLCA),
multi-agent coordination, and regime-aware contextual policy for sustainable
food logistics.

## Screenshots

| Operations Dashboard | Quality Monitoring | Supply Chain Map |
|:---:|:---:|:---:|
| ![Ops](docs/screenshots/ops-dashboard-light.png) | ![Quality](docs/screenshots/quality-tab-light.png) | ![Map](docs/screenshots/map-view-light.png) |

| Decisions Timeline | Analytics & Validation | Admin Panel |
|:---:|:---:|:---:|
| ![Decisions](docs/screenshots/decisions-timeline-light.png) | ![Analytics](docs/screenshots/analytics-overview-light.png) | ![Admin](docs/screenshots/admin-policy-light.png) |

| Analytics Tables | Analytics Figures | Admin Blockchain |
|:---:|:---:|:---:|
| ![Tables](docs/screenshots/analytics-tables-light.png) | ![Figures](docs/screenshots/analytics-figures-light.png) | ![Blockchain](docs/screenshots/admin-blockchain-light.png) |

| Admin Scenarios |
|:---:|
| ![Scenarios](docs/screenshots/admin-scenarios-light.png) |

## Architecture Highlights

- **LSTM demand forecaster** (numpy-only, 16 hidden units, truncated BPTT)
  with Holt-Winters fallback controlled by `FORECAST_METHOD` env var
- **Holt-Winters yield/supply forecaster** for inventory projection
- **5-agent coordinator** (Farm, Processor, Cooperative, Distributor, Recovery)
  dispatching decisions at lifecycle-stage boundaries
- **PiRAG pipeline** with TF-IDF ingestion, hybrid BM25+dense retrieval,
  guard checks, Merkle provenance, and LLM abstraction layer
- **MCP tool server** exposing compliance, SLCA lookup, and chain query tools
- **Circular economy scoring** for composting, animal feed, food bank pathways
- **PINN-enhanced Arrhenius spoilage model** with Baranyi lag phase
- **Softmax contextual policy** with 6-dimensional feature vector
- **On-chain governance** via Hardhat/Solidity smart contracts

## Frontend

Modern React dashboard built with shadcn/ui, featuring six pages:

| Page | Description |
|------|-------------|
| **Operations** | KPI bento grid, real-time telemetry charts with temperature zones, spoilage & yield preview |
| **Quality** | Circular spoilage risk gauge, shelf-life countdown, IoT sensor charts, PINN vs ODE comparison |
| **Decisions** | Timeline view with role/action filters, decision cards, analytics sidebar with pie chart, CSV/PDF export |
| **Map** | Leaflet map of South Dakota supply chain nodes with route overlays and live KPI popups |
| **Analytics** | Executive summary banner, interactive cross-scenario tables & charts, ablation study, radar profiles, scenario deep-dive gallery, carbon footprint analysis |
| **Admin** | Five tabs — Policy parameters, Blockchain status & config, Audit log, Scenario runner, Quick Decision |

**Tech stack:** React 18, React Router 7, shadcn/ui (Radix), Tailwind CSS, Recharts, React-Leaflet, Framer Motion, Sonner toasts, Vite 7

## Quick Start

### Backend (port 8100)

```bash
cd AGRI-BRAIN
python -m venv .venv && source .venv/bin/activate
pip install -e agri-brain-mvp-1.0.0/backend
python -m uvicorn src.app:API --port 8100 --app-dir agri-brain-mvp-1.0.0/backend
```

### Frontend (port 5173)

```bash
cd agri-brain-mvp-1.0.0/frontend
npm install
npm run dev
```

### Load data and verify

```bash
curl -X POST http://localhost:8100/case/load    # Load sensor CSV
curl http://localhost:8100/health                # {"ok":true}
```

- Dashboard: http://localhost:5173
- Admin panel: http://localhost:5173/admin
- API docs: http://localhost:8100/docs

### Simulation

```bash
cd mvp/simulation
python generate_results.py    # 5 scenarios x 5 modes
python generate_figures.py    # Publication figures
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_METHOD` | `lstm` | Demand forecaster: `lstm` or `holt_winters` |
| `ONLINE_LEARNING` | `false` | Enable REINFORCE policy gradient updates |
| `LLM_PROVIDER` | `template` | RAG answer engine: `template` or `api` |
| `DATA_CSV` | (auto) | Override path to spinach sensor CSV |

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
POST /results/generate       - Run full simulation, return summary JSON
GET  /results/figures/{name} - Serve generated figure files
POST /rag/ask                - Query the PiRAG pipeline
POST /mcp/call               - Call an MCP tool
WS   /stream                 - WebSocket real-time decision stream
```

## Project Structure

```
AGRI-BRAIN/
  README.md
  HOW_TO_RUN.md
  docs/screenshots/             # Frontend screenshots (light theme)
  agri-brain-mvp-1.0.0/
    backend/
      src/
        app.py                  # FastAPI application
        models/                 # Spoilage, LSTM/HW forecast, SLCA, policy,
                                #   reverse logistics, policy learner, footprint
        routers/                # API route handlers
        chain/                  # Blockchain integration (Hardhat)
        agents/                 # Multi-agent coordinator (5 roles), runtime, bus
      pirag/                    # PiRAG integration
        ingestion/              # Document parser, TF-IDF embedder, vector store
        inference/              # LLM abstraction (template + API engines)
        knowledge_base/         # SOPs, regulatory docs, IoT specs, SLCA guides
        mcp/                    # MCP tool server (compliance, slca, chain query)
        pyrag/                  # Hybrid BM25+dense retriever
        guards/                 # Unit and feasibility guards
        provenance/             # Merkle tree + on-chain anchoring
    frontend/
      src/
        pages/                  # Ops, Quality, Decisions, Map, Analytics, Admin
        components/ui/          # shadcn/ui component library
        layouts/                # MainLayout (sidebar, header, theme, notifications)
        hooks/                  # useTheme, useWebSocket
        lib/                    # Utility functions (cn, fmt, jget, jpost)
        mvp/                    # API configuration and helpers
    contracts/                  # Solidity smart contracts
  mvp/
    simulation/
      generate_results.py       # Scenario x mode simulation runner
      generate_figures.py       # Publication figure generator
      results/                  # Generated outputs (CSV, PNG, PDF)
```
