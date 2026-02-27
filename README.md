# AGRI-BRAIN

An adaptive supply-chain intelligence system combining PINN-based spoilage
prediction, Social Life-Cycle Assessment (SLCA), and regime-aware contextual
policy for sustainable food logistics.

## Quick Start

### Backend (port 8100)

```bash
cd agri-brain-mvp-1.0.0/backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m uvicorn src.app:API --port 8100 --reload
```

### Frontend (port 5173)

```bash
cd agri-brain-mvp-1.0.0/frontend
npm install
npm run dev
```

- Main app: http://127.0.0.1:5173 (Operations, Quality, Decisions)
- Admin panel: http://127.0.0.1:5173/admin (Policy, Scenarios, Results)
- API docs: http://127.0.0.1:8100/docs

### Simulation

```bash
cd mvp/simulation
python generate_results.py    # 5 scenarios x 5 modes
python generate_figures.py    # Publication figures
```

## Backend API

```
POST /case/load              - Load spinach CSV into state
GET  /kpis                   - Computed KPIs from loaded data
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
GET  /audit/memo.pdf         - Decision memo as PDF
POST /results/generate       - Run full simulation, return summary JSON
GET  /results/figures/{name} - Serve generated figure files
```

## Project Structure

```
agri-brain-mvp-1.0.0/
  backend/
    src/
      app.py              # FastAPI application
      models/             # PINN spoilage, forecast, SLCA, policy, footprint
      routers/            # API route handlers
      chain/              # Blockchain integration (Hardhat)
      agents/             # Agent runtime and WebSocket bus
    pirag/                # PiRAG integration (RAG, MCP, provenance)
  frontend/
    src/
      ui/                 # Main app (Ops, Quality, Decisions tabs)
      mvp/                # Admin panel (Policy, Scenarios, Results)
  contracts/              # Solidity smart contracts
mvp/
  simulation/
    generate_results.py   # Scenario x mode simulation runner
    generate_figures.py   # Publication figure generator
    results/              # Generated outputs (CSV, PNG, PDF)
```
