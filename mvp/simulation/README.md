# AGRI-BRAIN Simulation & Results Generation

This directory contains the simulation framework used to reproduce all paper
results for the AGRI-BRAIN system.

## Overview

The simulation runs 5 **scenarios** x 19 **modes** (95 episodes: 8 canonical + 11 §4.7 sensitivity ablations) to evaluate
the AGRI-BRAIN adaptive supply-chain intelligence system against baselines and
ablation variants, including MCP/piRAG context integration ablations. The
state vector phi(s) has 10 dimensions: six perception features, three
forecast-channel features (supply point, supply uncertainty, demand
uncertainty) that treat the LSTM demand and Holt's linear (level + trend)
supply forecasts symmetrically, and one demand-volatility price-pressure
proxy. The context vector psi remains 5-dimensional and carries
institutional / coordination signals only.

### Scenarios

| ID | Description |
|----|-------------|
| `heatwave` | 72 h climate-induced heatwave: +20 C ramp (hours 24-48) with exponential tail; +10 % RH |
| `overproduction` | Inventory multiplied 2.5x during hours 12-60; triggers redistribution |
| `cyber_outage` | Demand drops to 15 % from hour 24 onward; +10 C refrigeration degradation |
| `adaptive_pricing` | Demand oscillation (amplitude 45, period 60) plus Gaussian noise (sigma=14) |
| `baseline` | Original sensor data with no perturbation |

### Modes

| Mode | Description |
|------|-------------|
| `static` | Always selects cold-chain routing; no intelligence |
| `hybrid_rl` | Softmax policy with regime tilt but no SLCA logit correction |
| `no_pinn` | Degraded spoilage estimate (no PINN integration) |
| `no_slca` | Full PINN but uniform social scores (no SLCA optimisation) |
| `no_context` | Full agribrain policy but MCP/piRAG context disabled |
| `mcp_only` | MCP tool outputs only (compliance, forecast urgency, recovery saturation); piRAG features zeroed |
| `pirag_only` | piRAG retrieval only (regulatory pressure, retrieval confidence); MCP features zeroed |
| `agribrain` | Full system: PINN + SLCA + MCP tools + piRAG retrieval + online learning |

In addition to these 8 canonical modes, `generate_results.py` runs 11
§4.7 sensitivity-ablation modes (`agribrain_cold_start`,
`agribrain_pert_{10,25,50}` with/without `_static`, `agribrain_no_bonus`,
`agribrain_theta_pert_{10,25,50}`) for the prior- and weight-sensitivity
story — 19 modes total.

All four agribrain-logit modes (`agribrain`, `no_context`, `mcp_only`,
`pirag_only`) share the same per-scenario `ablation_seed` for the
paired-design statistics. `no_context` is **not** context-enabled —
its MCP/piRAG context layer is switched off and its logits run with a
zeroed modifier — but it shares the seed so that ARI differences
against `agribrain` reflect only the context injection rather than
stochastic noise. The other agribrain-logit modes (`mcp_only`,
`pirag_only`, `agribrain`) keep MCP/piRAG context on.

Stochastic perturbations apply to `tempC`, `RH`, `demand_units`, and
`inventory_units`. Decision latency is recorded as observed wall-clock
execution time (deterministic observation, not synthetically perturbed).
`STOCH_DELAY_PROB` (default `0.05`) is a separate temporal-lag mechanism that
acts on whole time series by carrying forward the previous sample.

## Models Used

All models are imported from the backend (`backend/src/models/`):

- **Spoilage (PINN)**: First-order ODE decay `dC/dt = -k_eff(t,T,H) * C` with
  Arrhenius parameters `k_ref=0.0021 h^-1`, `Ea/R=8000 K`, `T_ref=277.15 K`, `beta=0.25`
  and Baranyi lag phase `lambda=12.0 h`
- **Forecast**: LSTM demand forecaster (default) or Holt's linear
  (level + trend, no seasonal indices) demand fallback, controlled by
  `FORECAST_METHOD`. Yield/supply forecasting always uses Holt's
  linear via `query_yield` (`pirag.mcp.tools.yield_query`).
- **SLCA**: 4-component Social Life-Cycle Assessment
  (Carbon, Labor, Resilience, Price transparency)
- **Policy**: Contextual softmax policy with weight matrix
  Theta of shape (3 actions x 10 features) and Bollinger volatility
  regime tilt. The 5-dimensional institutional context vector psi has
  its own learned weight matrix Theta_context of shape (3, 5) which
  produces an additive logit modifier.

## MCP/piRAG Context Integration

Each agent step invokes role-specific MCP tools and piRAG knowledge retrieval:

- **MCP tools** (JSON-RPC 2.0): 14 statically registered tools (the
  live registry length is `len(get_default_registry().list_tools())`,
  13 by default) — compliance check, spoilage forecast, SLCA lookup,
  chain query, policy oracle, calculator, footprint query,
  convert_units, pirag_query, explain, context_features, simulate,
  yield_query, and demand_query; the coordinator adds 5 runtime
  role-capability tools (18 at simulation time, since `simulate` is
  unregistered when `SIM_API_BASE` is empty, as in the simulator
  subprocess).
- **piRAG pipeline**: 20-document knowledge base with BM25+TF-IDF hybrid
  retrieval at `k=4` for the simulator/benchmark call sites in
  `pirag/context_builder.py` (the underlying `PiRAGPipeline.ask` default
  is `k=6`; the simulation/benchmark always passes `k=4`), physics-
  informed reranking, scenario-discriminative query expansion.
- **State features**: 10D feature vector phi(s) = [freshness, inventory pressure, demand point forecast, thermal stress, spoilage urgency, interaction, supply point, supply uncertainty, demand uncertainty, price signal] with policy weight matrix Theta of shape (3, 10). Supply and demand forecast uncertainties are residual-std prediction-error estimates (Hyndman & Athanasopoulos 2018, Ch. 8.7); the price signal is a demand-volatility Bollinger z-score clipped to [-1, +1] that proxies market pressure.
- **Context features**: 5D feature vector psi = [compliance severity, forecast urgency, retrieval confidence, regulatory pressure, recovery saturation] with learned Theta_context weight matrix of shape (3, 5)
- **Governance override**: Deterministic redistribution when policy probability of cold-chain falls below the calibration-derived ceiling (5th percentile of pi(cold_chain) over benchmark rollouts) AND local-redistribute dominates cold-chain by the calibrated median gap
- **Online REINFORCE learning**: Sign-constrained shrinkage-prior gradient updates on Theta_context, on a (3, 10) delta added to Theta (PolicyDeltaLearner), and on the reward-shaping vectors SLCA_BONUS, SLCA_RHO_BONUS, NO_SLCA_OFFSET (RewardShapingLearner). Each learnable delta is zero-initialised and capped at 25 percent of its hand-calibrated initial magnitude per entry so learning anchors on domain priors.
- **Causal explanation engine**: BECAUSE/WITHOUT reasoning with [KB:] citations and Merkle provenance
- **Protocol recording**: Every MCP JSON-RPC interaction is captured as genuine protocol traffic

## Multi-Agent Architecture

The simulation dispatches each timestep to a role-specific supply chain
agent via an `AgentCoordinator`.  Agents are mapped to lifecycle stages
based on hours since harvest:

| Agent | Role | Stage (hours) | Bias [CC, LR, Rec] | Mandate |
|-------|------|---------------|---------------------|---------|
| FarmAgent | farm | [0, 18) | [+0.08, -0.03, -0.05] | Preserve freshness |
| ProcessorAgent | processor | [18, 36) | [-0.02, +0.06, -0.04] | Processing efficiency |
| CooperativeAgent | cooperative | [12, 30) overlay | [0.00, +0.04, -0.04] | Governance coordination |
| DistributorAgent | distributor | [36, 54) | [-0.05, +0.10, -0.05] | Community redistribution |
| RecoveryAgent | recovery | [54, +inf) | [-0.06, -0.02, +0.08] | Waste valorization |

**Message types**: `SPOILAGE_ALERT`, `SURPLUS_ALERT`, `CAPACITY_UPDATE`,
`REROUTE_REQUEST`, `ACK`.

Role biases are small relative to the THETA-phi logits and mode-specific
bonuses, nudging decisions toward each stage's mandate without overriding
the global policy.

## Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| ARI | `(1 - waste) * SLCA * shelf_quality` | Adaptive Resilience Index |
| RLE | `routed_at_risk / total_at_risk` | Reverse Logistics Efficiency |
| Waste | `(k_inst * W_SCALE)^W_ALPHA * (1 + surplus_penalty)` then `* (1 - save_factor)` | Effective produce loss rate |
| SLCA | `w_c*C + w_l*L + w_r*R + w_p*P` | Social LCA composite score |
| Carbon | `sum(km * carbon_per_km)` | Total CO2 emissions (kg) |
| Equity | `1 - sigma(SLCA_values)` | SLCA uniformity index (Gini-inspired) |

## Directory Layout

```
mvp/simulation/
├── generate_results.py          # Scenario x mode simulation runner
├── generate_figures.py          # Publication figure generator (PNG + PDF)
├── stochastic.py                # 8-source stochastic perturbation engine
├── reproduce_core.py            # One-command full reproduction pipeline
├── benchmarks/                  # Multi-seed benchmark & stress suites
│   ├── run_benchmark_suite.py   # Full multi-seed benchmark runner
│   ├── run_stress_suite.py      # Noise/missing-data/fault stress tests
│   ├── run_temporal_stability.py # Within-trace temporal stability (early/mid/late thirds of one trace; NOT external validity)
│   ├── run_single_seed.py       # Single-seed benchmark episode
│   └── aggregate_seeds.py       # Canonical multi-seed aggregation + stats
├── validation/                  # Result validation & regression guards
│   ├── validate_results.py      # Deterministic result validation
│   ├── run_regression_guard.py  # Regression drift guard
│   ├── validate_publication_artifacts.py  # Publication artifact schema check
│   └── verify_context_integration.py     # MCP/piRAG context integration check
├── analysis/                    # Diagnostics & paper evidence export
│   ├── build_artifact_manifest.py     # SHA-256 manifest + git commit pinning
│   ├── channel_saturation_analysis.py # §5.8 saturation/redundancy analysis
│   ├── explainability_metrics.py      # §4.10 explainability assessment
│   ├── export_paper_evidence.py       # Paper evidence trace export
│   ├── verify_anchored_root.py        # On-chain Merkle-root verifier
│   └── verify_manifest.py             # Manifest integrity check
├── tests/                       # Stochastic & benchmark test suites
│   ├── test_stochastic_feasibility.py  # Full stochastic feasibility tests
│   ├── test_stochastic_quick.py        # Quick stochastic validation tests
│   ├── stochastic_benchmark_check.py   # Benchmark seed variance check
│   └── stochastic_rank_check.py        # Method rank stability check
└── results/                     # Generated outputs (CSV, PNG, PDF, JSON)
```

## Reproducing Results

```bash
cd mvp/simulation
pip install -e ../../agribrain/backend

# Quick: core tables and figures only
python generate_results.py    # runs all scenarios, saves tables
python generate_figures.py    # generates publication figures (PNG + PDF)

# Full: one-command reproduction pipeline (recommended)
python reproduce_core.py
```

All outputs are saved to `mvp/simulation/results/`:
- `table1_summary.csv` -- Scenario x Method (static, hybrid_rl, agribrain)
- `table2_ablation.csv` -- Scenario x Variant (all 19 modes: 8 canonical + 11 §4.7 ablations)
- `benchmark_summary.json` -- Multi-seed means/std/CI
- `benchmark_significance.json` -- Permutation-test p-values + effect sizes
- `stress_summary.json` -- Stress-suite robustness outputs
- `stress_degradation.csv` -- Delta metrics under stressors
- `artifact_manifest.json` -- SHA-256 reproducibility manifest
- `traces_*.json` -- Decision traces with keywords, causal reasoning, provenance per scenario
- `mcp_protocol_*.json` -- Genuine MCP JSON-RPC interaction logs per scenario
- `heatwave`/`overproduction`/`cyber_outage`/`adaptive_pricing`/`cross_scenario`/`ablation`/`green_ai_carbon` plus the H1/H2/H3 paper figures `performance_efficiency`/`context_value`/`stress_robustness` (`.png/.pdf`) -- publication figures

## Expected Approximate Results (AGRI-BRAIN mode)

| Scenario | ARI | Waste | RLE | SLCA |
|----------|-----|-------|-----|------|
| Heatwave | 0.549 | 0.031 | 0.806 | 0.686 |
| Overproduction | 0.575 | 0.041 | 0.854 | 0.666 |
| Cyber Outage | 0.607 | 0.037 | 0.713 | 0.691 |
| Adaptive Pricing | 0.661 | 0.029 | 0.738 | 0.727 |
| Baseline | 0.685 | 0.024 | 0.780 | 0.743 |

### Ablation Impact (ARI, largest to smallest drop from agribrain)

1. **Removing SLCA** (`no_slca`) -- drops SLCA to ~0.60-0.65 and ARI to ~0.50-0.58
2. **Removing PINN** (`no_pinn`) -- slight waste increase; ARI ~0.57-0.71
3. **Hybrid RL only** -- no SLCA logit bonus; ARI ~0.55-0.66
4. **No context** (`no_context`) -- full policy but no MCP/piRAG; ARI ~0.59-0.73
5. **MCP only** (`mcp_only`) -- compliance/forecast tools without piRAG retrieval; ARI +0.003-0.017 over no_context
6. **piRAG only** (`pirag_only`) -- knowledge retrieval without MCP tools; ARI +0.007-0.017 over no_context
7. **Full AGRI-BRAIN** (`agribrain`) -- all components; ARI +0.012-0.032 over no_context

MCP-only matches or exceeds piRAG-only in ARI in 4 of 5 scenarios (piRAG leads only in cyber_outage); piRAG's advantage appears at the decision-necessity level, and full AGRI-BRAIN exceeds both in every scenario.

### Context Diagnostics (agribrain mode)

| Scenario | MCP calls | piRAG queries | Governance overrides | Learner delta-theta norm |
|----------|-----------|---------------|---------------------|--------------------------|
| Heatwave | ~450 | 288 | ~68 | 0.023 |
| Overproduction | ~680 | 288 | 0 | 0.032 |
| Cyber Outage | ~430 | 288 | ~159 | 0.024 |
| Adaptive Pricing | ~490 | 288 | 0 | 0.042 |
| Baseline | ~430 | 288 | 0 | 0.048 |

## Seed & Reproducibility

Single-run scripts default to `seed=42`, while benchmark summaries are computed over
20 fixed seeds by default (see `BENCHMARK_SEEDS` in `.env.example`).
Set `DETERMINISTIC_MODE=true` for strict reproducibility checks and snapshot guards.
Wall-clock latency metrics remain runtime-dependent and may vary by machine load.

## API Integration

The simulation is also accessible via the backend API:

```
POST /results/generate    -> starts simulation job (background)
GET  /results/status      -> poll job status
GET  /results/summary     -> fetch latest completed summary
GET  /results/figures/{filename}  -> serves generated figure files
```
