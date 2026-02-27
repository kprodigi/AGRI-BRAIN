# AGRI-BRAIN MVP

An adaptive supply-chain intelligence system for sustainable food logistics.

## Prerequisites

- **Python** >= 3.10 with pip
- **Node.js** >= 18 with npm

## Quick Start

### 1. Backend

```bash
cd agri-brain-mvp-1.0.0/backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m uvicorn src.app:API --port 8100 --reload
```

API docs at http://127.0.0.1:8100/docs

### 2. Frontend

```bash
cd agri-brain-mvp-1.0.0/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 — main dashboard with Operations, Quality, Decisions tabs.
Open http://127.0.0.1:5173/admin — admin panel with Policy, Blockchain, Scenarios, Results.

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
| POST | /decide | Run decision engine |
| GET | /last-decision | Most recent decision memo |
| GET | /decisions | Decision feed |
| POST | /scenarios/run | Apply a scenario |
| POST | /scenarios/reset | Reset to baseline |
| GET | /scenarios/list | List available scenarios |
| GET | /governance/policy | Current policy parameters |
| POST | /governance/policy | Update policy |
| GET | /governance/chain | Blockchain config |
| GET | /audit/logs | Audit log |
| GET | /audit/memo.pdf | Decision memo as PDF |
| POST | /results/generate | Run full simulation |
| GET | /results/figures/{name} | Serve generated figures |

## Project Structure

```
agri-brain-mvp-1.0.0/
  backend/
    src/
      app.py                # FastAPI application
      state.py              # Global state with thread-safe accessors
      models/               # PINN spoilage, forecast, SLCA, footprint, policy
      routers/              # API route handlers
      chain/                # Blockchain integration (Hardhat)
      agents/               # Agent runtime and WebSocket bus
    pirag/                  # PiRAG integration (RAG, MCP, provenance)
  frontend/
    src/
      ui/                   # Main app (Ops, Quality, Decisions tabs)
      mvp/                  # Admin panel (Policy, Scenarios, Results)
  contracts/                # Solidity smart contracts
mvp/
  simulation/
    generate_results.py     # Scenario x mode simulation runner
    generate_figures.py     # Publication figure generator
    results/                # Generated outputs (CSV, PNG, PDF)
```
