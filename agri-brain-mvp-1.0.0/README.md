# AGRI-BRAIN MVP

An adaptive supply-chain intelligence system for sustainable food logistics.

## Prerequisites

- **Python** >= 3.10 with pip
- **Node.js** >= 18 with npm

## Quick Start

### 1. Backend

```bash
cd AGRI-BRAIN
python -m venv .venv && source .venv/bin/activate
pip install -e agri-brain-mvp-1.0.0/backend
python -m uvicorn src.app:API --port 8100 --app-dir agri-brain-mvp-1.0.0/backend
```

Load sensor data (required for dashboard):

```bash
curl -X POST http://localhost:8100/case/load
```

API docs at http://127.0.0.1:8100/docs

### 2. Frontend

```bash
cd agri-brain-mvp-1.0.0/frontend
npm install
npm run dev
```

Open http://localhost:5173 — dashboard with eight pages:

- **Operations** — KPI cards, real-time telemetry charts, spoilage & yield preview
- **Quality** — Spoilage risk gauge, shelf-life countdown, IoT sensor charts, PINN vs ODE comparison
- **Decisions** — Timeline view with filters, decision analytics sidebar, CSV/PDF export
- **Map** — Leaflet map of South Dakota supply chain nodes with route overlays
- **Analytics** — Cross-scenario tables, charts, radar profiles, scenario deep-dive gallery
- **MCP/piRAG** — Protocol overview, context features, knowledge base, traces, causal reasoning
- **Demo** — Interactive system demo with pipeline walkthrough and agent decision theater
- **Admin** — Policy, Blockchain, Audit, Scenarios, Quick Decision, Runtime, MCP tabs

Features: dark mode toggle, WebSocket live indicator, notification bell, responsive sidebar.

### 3. Simulation

```bash
cd mvp/simulation
python generate_results.py
python generate_figures.py
```

Results saved to `mvp/simulation/results/`.

## Key API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /case/load | Load spinach CSV into state |
| GET | /kpis | Computed KPIs |
| GET | /telemetry | Sensor time-series (tempC, RH, inventory, demand) |
| GET | /predictions | Spoilage predictions and yield forecast |
| POST | /decide | Run decision engine |
| GET | /decisions | Decision feed |
| POST | /scenarios/run | Apply a scenario |
| POST | /scenarios/reset | Reset to baseline |
| GET | /scenarios/list | List available scenarios |
| GET | /governance/policy | Current policy parameters |
| POST | /governance/policy | Update policy |
| GET | /governance/chain | Blockchain config |
| GET | /audit/logs | Audit log |
| POST | /results/generate | Run full simulation |
| GET | /results/status | Poll simulation progress |
| GET | /results/summary | Fetch completed results |
| GET | /results/figures/{name} | Serve generated figures |
| POST | /mcp/mcp | MCP JSON-RPC 2.0 endpoint |
| GET | /mcp/resources | List MCP resources |
| GET | /mcp/prompts | List MCP prompts |
| POST | /rag/ask | Query piRAG knowledge base |
| POST | /rag/ingest | Ingest documents into piRAG |
| WS | /stream | Real-time decision stream |
| WS | /stream | Real-time decision stream |

## Tech Stack

**Backend:** FastAPI, uvicorn, numpy, pandas, matplotlib, reportlab, web3

**Frontend:** React 18, React Router 7, shadcn/ui (Radix), Tailwind CSS, Recharts, React-Leaflet, Framer Motion, Sonner, Vite 7

## Project Structure

```
agri-brain-mvp-1.0.0/
  backend/
    src/
      app.py                # FastAPI application
      models/               # PINN spoilage, forecast, SLCA, footprint, policy
      routers/              # API route handlers
      chain/                # Blockchain integration (Hardhat)
      agents/               # Agent runtime and WebSocket bus
    pirag/                  # PiRAG integration (RAG, MCP, provenance)
    experiments/            # Policy experiment scripts and outputs
  frontend/
    src/
      pages/                # Ops, Quality, Decisions, Map, Analytics, Admin
      components/ui/        # shadcn/ui component library
      layouts/              # MainLayout (sidebar, header, theme, notifications)
      hooks/                # useTheme, useWebSocket
      lib/                  # Utility functions (cn, fmt, jget, jpost)
      mvp/                  # API configuration and helpers
  contracts/                # Solidity smart contracts
```
