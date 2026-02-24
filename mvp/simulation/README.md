# AGRI-BRAIN Simulation & Results Generation

This directory contains the simulation framework used to reproduce all paper
results for the AGRI-BRAIN system.

## Overview

The simulation runs 5 **scenarios** × 5 **modes** to evaluate the AGRI-BRAIN
adaptive supply-chain intelligence system against baselines and ablation
variants.

### Scenarios

| ID | Description |
|----|-------------|
| `heatwave` | 72 h climate-induced heatwave: +20 °C ramp (hours 24–48) with exponential tail; +10 % RH |
| `overproduction` | Inventory multiplied 2.5× during hours 12–60; triggers redistribution |
| `cyber_outage` | Yield drops to 15 % and inventory to 25 % from hour 24 onward |
| `adaptive_pricing` | Demand oscillation (amplitude 45, period 60) plus Gaussian noise (σ=14) |
| `baseline` | Original sensor data with no perturbation |

### Modes

| Mode | Description |
|------|-------------|
| `static` | Always selects cold-chain routing — no intelligence |
| `hybrid_rl` | Softmax policy with regime tilt but no SLCA logit correction |
| `no_pinn` | Degraded spoilage estimate (no PINN integration) |
| `no_slca` | Full PINN but uniform social scores (no SLCA optimisation) |
| `agribrain` | Full system: PINN spoilage + SLCA-aware routing + regime tilt |

## Models Used

All models are imported from the backend (`backend/src/models/`):

- **Spoilage (PINN)**: First-order ODE decay `dC/dt = -k(T,H) · C` with
  parameters `k₀=0.04`, `α=0.12`, `T₀=4.0 °C`, `β=0.25`
- **Forecast**: Exponential smoothing with horizon tiling (Section 4.2.2)
- **SLCA**: 4-component Social Life-Cycle Assessment
  (Carbon, Labour, Resilience, Price transparency)
- **Policy**: Contextual softmax policy with θ matrix (3 actions × 6 features)
  and Bollinger volatility regime tilt

## Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| ARI | `(1 − waste) · SLCA · shelf_quality` | Adaptive Resilience Index |
| RLE | `routed_at_risk / total_at_risk` | Reverse Logistics Efficiency |
| Waste | `ρ − saved` (intervention model) | Effective produce loss rate |
| SLCA | `w_c·C + w_l·L + w_r·R + w_p·P` | Social LCA composite score |
| Carbon | `Σ(km × carbon_per_km)` | Total CO₂ emissions (kg) |
| Equity | `1 − |price − MSRP| / MSRP` | Price fairness index |

## Reproducing Results

```bash
cd mvp/simulation
pip install numpy pandas matplotlib scipy
python generate_results.py    # runs all scenarios, saves tables
python generate_figures.py    # generates all 7 publication figures
```

All outputs are saved to `mvp/simulation/results/`:
- `table1_summary.csv` — Scenario × Method (static, hybrid_rl, agribrain)
- `table2_ablation.csv` — Scenario × Variant (all 5 modes)
- `fig2_heatwave.png/.pdf` — Heatwave deep-dive (2×2)
- `fig3_reverse.png/.pdf` — Overproduction reverse logistics (2×2)
- `fig4_cyber.png/.pdf` — Cyber outage analysis (1×3)
- `fig5_pricing.png/.pdf` — Demand volatility & pricing (2×2)
- `fig6_cross.png/.pdf` — Cross-scenario comparison (2×2 bars)
- `fig7_ablation.png/.pdf` — Ablation study (1×3 bars)
- `fig8_green.png/.pdf` — Green AI carbon footprint (1×2)

## Expected Approximate Results (AGRI-BRAIN mode)

| Scenario | ARI | Waste | RLE | SLCA |
|----------|-----|-------|-----|------|
| Heatwave | ~0.18 | ~0.18 | ~1.00 | ~0.87 |
| Overproduction | ~0.19 | ~0.18 | ~0.98 | ~0.86 |
| Cyber Outage | ~0.19 | ~0.18 | ~1.00 | ~0.87 |
| Price Volatility | ~0.19 | ~0.18 | ~1.00 | ~0.87 |

### Ablation Impact (largest to smallest)

1. **Removing SLCA** has the largest impact — drops SLCA to 0.50 and ARI to ~0.09
2. **Removing PINN** degrades waste detection — waste increases to ~0.42
3. **Hybrid RL** (no SLCA logit bonus) — slightly less optimal routing

## Seed & Reproducibility

All runs use `seed=42` via `numpy.random.default_rng(42)`.
Deterministic seeding ensures exact reproducibility across platforms.

## API Integration

The simulation is also accessible via the backend API:

```
POST /results/generate    → runs simulation, returns summary JSON
GET  /results/figures/{filename}  → serves generated figure files
```
