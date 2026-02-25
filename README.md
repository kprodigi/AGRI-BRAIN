# AGRI-BRAIN

Adaptive supply-chain intelligence system combining PINN-based spoilage
prediction, Social Life-Cycle Assessment (SLCA), and regime-aware contextual
policy for sustainable food logistics.

## Generating Paper Results

```bash
cd mvp/simulation
pip install numpy pandas matplotlib scipy
python generate_results.py    # runs all scenarios, saves tables
python generate_figures.py    # generates publication-quality figures
```

Results are written to `mvp/simulation/results/` including:
- `table1_summary.csv` — Scenario × Method summary
- `table2_ablation.csv` — Full ablation study
- 7 publication figures (PNG + PDF at 300 DPI)

See `mvp/simulation/README.md` for full methodology documentation.

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
