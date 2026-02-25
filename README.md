# AGRI-BRAIN

An adaptive supply-chain intelligence system combining PINN-based spoilage
prediction, Social Life-Cycle Assessment (SLCA), and regime-aware contextual
policy for sustainable food logistics.

## Backend API

The backend exposes simulation results via REST:

```
POST /results/generate           → run simulation, return summary JSON
GET  /results/figures/{filename} → serve generated figure files
```

## Project Structure

```
agri-brain-mvp-1.0.0/
  backend/
    src/
      app.py              # FastAPI application
      models/             # PINN spoilage, forecast, SLCA, policy
      routers/            # API route handlers
mvp/
  simulation/
    generate_results.py   # Scenario × mode simulation runner
    generate_figures.py   # Publication figure generator
    results/              # Generated outputs (CSV, PNG, PDF)
```
