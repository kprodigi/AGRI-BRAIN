<div align="center">

# AGRI-BRAIN

**A Protocol-Mediated, Physics-Informed Multi-Agent Framework for
Explainable Perishable Supply Chains**

[![CI](https://github.com/kprodigi/AGRI-BRAIN/actions/workflows/ci.yml/badge.svg)](https://github.com/kprodigi/AGRI-BRAIN/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Overview](#overview) •
[Key Results](#key-results) •
[Quick Start](#quick-start) •
[Reproducing the Paper](#reproducing-the-paper) •
[Documentation](#documentation) •
[Citation](#citation)

</div>

![AGRI-BRAIN architecture: protocol-mediated physics-informed multi-agent decision pipeline](docs/figures/architecture.jpg)

## Overview

Perishable agri-food supply chains stay fragile because sensing,
forecasting, compliance, and routing are optimized in isolation. AGRI-BRAIN
treats **communication as an explicit decision variable**: five role agents
(farm, processor, cooperative, distributor, recovery) fuse three channels —
typed peer messages, **Model Context Protocol (MCP)** tool calls, and
**physics-informed retrieval-augmented generation (piRAG)** — into a
five-dimensional context vector that directly shifts a softmax routing
policy through a learned, sign-constrained logit modifier. A
physics-informed (Arrhenius–Baranyi + bounded neural residual) spoilage
estimator supplies the physical state, online REINFORCE adapts the context
weights, and every decision is serialized to a Merkle-anchored audit ledger,
so each explanation is structural rather than post-hoc.

The framework is evaluated on a 72-hour fresh-spinach cold chain under four
documented disruption scenarios (heatwave, overproduction surge, cyber
outage, adaptive-pricing shock) plus an unperturbed baseline, against an
8-mode architecture ablation.

## Highlights

- **Physics-informed spoilage perception** — Arrhenius–Baranyi kinetics with
  a bounded (±0.08) neural residual trained under an ODE-residual penalty.
- **MCP interoperability layer** — 14 statically registered tools + 5
  runtime role-capability tools behind one JSON-RPC 2.0 surface with three
  transports ([tool inventory](agribrain/backend/pirag/mcp/TOOL_INVENTORY.md)).
- **Physics-informed retrieval (piRAG)** — hybrid BM25 + TF-IDF over a
  20-document institutional corpus with thermal query expansion,
  physics-aware reranking, and quality guards.
- **Context-fused routing policy** — a 5D context vector shifts the softmax
  logits through a sign-constrained, online-REINFORCE-learned modifier; a
  deterministic governance override guards joint crisis evidence.
- **Explainability by construction** — BECAUSE/WITHOUT causal narratives,
  per-component attribution, counterfactuals, and Merkle-rooted provenance
  from the same vector that drove the decision.
- **Statistically defended benchmark** — 800-episode crossed design with
  8 stochastic perturbation sources, paired permutation tests, BCa
  bootstrap, and Holm–Bonferroni correction, validated end-to-end in CI.



## Key Results

Evaluated on 5 perturbation scenarios × 8 routing modes × 20 stochastic
seeds × 288 hourly steps = 230,400 decisions per HPC re-run. Source:
[`benchmark_summary.json`](mvp/simulation/results/benchmark_summary.json)
and [`paper_benchmark_table.json`](mvp/simulation/results/paper_benchmark_table.json)
(BCa-bootstrap CIs, Holm–Bonferroni FWER control, paired permutation tests).

| Hypothesis | Effect | Significance |
|---|---|---|
| **H1 — Integration superiority.** AGRI-BRAIN ARI beats no-context across all 5 scenarios. | ΔARI **+0.012 to +0.032** | Cohen's d_pooled **0.96–2.01**, p_adj < 0.001 |
| **H2 — Channel complementarity.** piRAG is the dominant standalone router; MCP integrates **synergistically** and adds an exclusive discrete-safety layer (governance overrides, compliance reroutes, outage resilience). | Context decisive on **10.3%** of decisions (41% where active); piRAG-necessary **7.7%**, MCP-necessary **1.7%**, MCP-necessity doubling on its governed events | **Non-redundancy 75.0%** (95% CI [72.2, 77.7]); necessity coupling **φ = +0.26** (p < 10⁻³); each channel beats no-context (p_adj < 0.001) |
| **H3 — Communication robustness.** Performance degrades < 1% under sensor noise, missing data, telemetry delay, and MCP tool fault. | \|ΔARI\| < **0.01** all five stressors (worst single cell 0.0091) | Pre-specified ≤ 0.01 threshold met |

<details>
<summary><b>H2 in full — the decision-level channel decomposition</b></summary>

The linear logit modifier (m_MCP + m_piRAG ≡ m_full) cannot be
super-additive, so channel value is measured at the non-linear argmax via
per-decision drop-one counterfactuals. Context is decisive on **10.3%** of
agribrain decisions (up to **33%** under cyber outage) — and on **41%** of
the decisions where the context layer is *active*, with the influence highly
concentrated (Gini **0.804**; the decisive 10.3% carry **48.3%** of all
decision movement). piRAG-necessary **7.7%**, MCP-necessary **1.7%** (almost
entirely synergistic — MCP-only **0.1%** of context-changed decisions),
emergent synergy **1.7%** of all decisions (n = 19,200, 20 seeds).

The **non-redundancy index is 75.0%** (bootstrap 95% CI [72.2, 77.7]) — most
context-changed decisions are attributable to a single channel or to synergy
(> 0.5); note this does **not** exceed the channel-independence baseline
(0.79; permutation p = 1.0), so the channels are not *more* separable than
chance. The significant structure is the **positive necessity coupling**
(φ = +0.26, permutation p < 10⁻³) and each channel independently beating
no-context (p_adj < 0.001). Each channel adds ARI (mcp-only 0.592,
pirag-only 0.591 vs no-context 0.581; full 0.598). MCP's discrete value:
governance overrides, compliance-decisive reroutes (10.9% of compliance
events), MCP-necessity doubling on its governed events (compliance ψ₀>0:
**1.7%→3.0%**; cyber-outage **5.3%→9.9%**), and cyber-outage edge resilience
(reroute P 0.63→0.73). Source:
[`channel_attribution_aggregate.json`](mvp/simulation/results/channel_attribution_aggregate.json),
[`channel_complementarity_test.json`](mvp/simulation/results/channel_complementarity_test.json).

</details>

## Repository Structure

```
AGRI-BRAIN/
├── README.md                       # You are here
├── HOW_TO_RUN.md                   # Full operating manual (env vars, HPC, validation)
├── LICENSE                         # MIT
├── docs/
│   ├── ARCHITECTURE.md             # Layer-by-layer technical reference + API + frontend
│   ├── METHODS_REPRO_APPENDIX.md   # Step-by-step reproduction appendix
│   ├── STATISTICAL_METHODS.md      # Tests, corrections, CI machinery
│   ├── RELEASE.md                  # Release notes
│   ├── figures/                    # Architecture diagram
│   └── screenshots/                # Dashboard gallery (18 views)
├── agribrain/
│   ├── backend/
│   │   ├── src/                    # FastAPI app, models (PINN, LSTM, SLCA, policy),
│   │   │                           #   routers, chain integration, 5-agent coordinator
│   │   ├── pirag/                  # piRAG pipeline, context fusion/learning, MCP
│   │   │   ├── mcp/                #   protocol, registry, transports, 14 tools
│   │   │   ├── knowledge_base/     #   20 domain documents
│   │   │   ├── guards/             #   unit, feasibility, retrieval-quality guards
│   │   │   ├── provenance/         #   Merkle tree + on-chain anchoring
│   │   │   └── tests/              #   MCP/piRAG test suite
│   │   └── tests/                  # Backend test suite
│   ├── frontend/                   # React 18 + shadcn/ui dashboard (8 pages)
│   └── contracts/                  # Solidity suite (6 contracts) + Hardhat
├── mvp/simulation/
│   ├── generate_results.py         # Scenario × mode simulation runner
│   ├── generate_figures.py         # Publication figure generator
│   ├── stochastic.py               # 8-source perturbation engine
│   ├── reproduce_core.py           # One-command full reproduction
│   ├── benchmarks/                 # Multi-seed benchmark, stress, attribution suites
│   ├── validation/                 # Result validation + regression guards
│   ├── analysis/                   # Diagnostics, manifest builder, evidence export
│   ├── tests/                      # Stochastic & benchmark tests
│   └── results/                    # Canonical committed artifacts (see Data Availability)
└── hpc/                            # SLURM orchestration (run → seed array → aggregate)
```

## Quick Start

The commands below assume the repository is cloned as ``AGRI-BRAIN``
(the default ``git clone`` directory). If you cloned into a different
directory name, substitute that name wherever ``AGRI-BRAIN`` appears.

### Backend (port 8100)

```bash
cd AGRI-BRAIN
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (cmd)
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
pip install -e agribrain/backend
python -m uvicorn src.app:API --port 8100
```

### Frontend (port 5173)

```bash
cd agribrain/frontend
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

## Reproducing the Paper

### Single-machine simulation

```bash
cd mvp/simulation
python generate_results.py    # 5 scenarios × 19 modes (95 episodes: 8 canonical + 11 §4.7 ablations)
python generate_figures.py    # publication figures (PNG + PDF)
```

Or run the one-command end-to-end pipeline:

```bash
python mvp/simulation/reproduce_core.py
```

### HPC benchmark (20 seeds)

The 20-seed stochastic benchmark (5 scenarios × 8 canonical modes × 20 seeds
= 800 episodes for the headline ablation, plus the 11 §4.7 sensitivity modes
which add another 5 × 11 × 20 = 1,100 episodes when run, plus aggregation and
figure generation) is submitted through three SLURM scripts in the
`hpc/` directory. From the HPC login node, in the repo root:

```bash
bash hpc/hpc_run.sh
```

This orchestrator:

1. Creates `.venv` if absent, installs the backend package, and runs a
   Policy-shape load assertion (fails fast if the resolver pulled a broken
   combination).
2. Computes `RUN_TAG=$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M)`.
3. Submits `hpc/hpc_seed.sh` as a 20-task array, one seed per task
   (`--time=06:00:00`, `--mem=8G`, `--cpus-per-task=4`).
4. Submits `hpc/hpc_aggregate.sh` with `--dependency=afterok:<seed_job>`
   (`--time=08:00:00`, `--mem=16G`). The aggregator runs Stages 1-10:
   base table generation, validation, both context-ablation and canonical
   multi-seed aggregators, stress suite, figures, paper-evidence export,
   manifest, and final validation.
5. Writes `hpc_results_<RUN_TAG>.tar.gz` on completion. Transfer with
   `scp <hpc-host>:$PWD/hpc_results_<RUN_TAG>.tar.gz .` and untar into the
   results tree.

End-to-end wall time is typically 6-10 h with scheduler queueing. The
canonical pre-HPC validation lives in `mvp/simulation/validation/` —
run `python mvp/simulation/validation/validate_results.py` and
`python mvp/simulation/analysis/verify_manifest.py --strict-commit
--allow-missing` before submitting the HPC run.

### Verifying the committed evidence

Every published number can be checked on a fresh clone without running
anything:

```bash
python mvp/simulation/validation/validate_publication_artifacts.py
```

This validates the artifact manifest (SHA-256 per file), the significance
fields, the stress pass/fail schema, and the per-claim threshold assertions.

## Interactive Dashboard

A React dashboard ships with the framework for operating the live system —
eight pages covering operations, quality, decision explainability, mapping,
analytics, MCP/piRAG inspection, demos, and admin (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#frontend) for the full page and
tech-stack reference).

| System Walkthrough | Decision Explainability | Operations |
|:---:|:---:|:---:|
| ![Pipeline](docs/screenshots/demo-pipeline.gif) | ![Explainability](docs/screenshots/decisions-explainability-light.png) | ![Ops](docs/screenshots/ops-dashboard-light.png) |

<details>
<summary><b>Full screenshot gallery (18 views)</b></summary>

| Operations Dashboard | Quality Monitoring | Supply Chain Map |
|:---:|:---:|:---:|
| ![Ops](docs/screenshots/ops-dashboard-light.png) | ![Quality](docs/screenshots/quality-tab-light.png) | ![Map](docs/screenshots/map-view-light.png) |

| Decisions Timeline | Analytics | Admin Panel |
|:---:|:---:|:---:|
| ![Decisions](docs/screenshots/decisions-timeline-light.png) | ![Analytics](docs/screenshots/analytics-light.png) | ![Admin](docs/screenshots/admin-policy-light.png) |

| Admin Blockchain | Admin Scenarios |
|:---:|:---:|
| ![Blockchain](docs/screenshots/admin-blockchain-light.png) | ![Scenarios](docs/screenshots/admin-scenarios-light.png) |

| Explainability Panel | MCP Tools | MCP Resources |
|:---:|:---:|:---:|
| ![Explainability](docs/screenshots/decisions-explainability-light.png) | ![MCP Tools](docs/screenshots/admin-mcp-tools-light.png) | ![MCP Resources](docs/screenshots/admin-mcp-resources-light.png) |

| MCP Invocation | piRAG Search |
|:---:|:---:|
| ![MCP Invoke](docs/screenshots/admin-mcp-invoke-light.png) | ![piRAG](docs/screenshots/admin-mcp-pirag-light.png) |

| MCP/piRAG Overview | Context Features | Knowledge Base |
|:---:|:---:|:---:|
| ![Overview](docs/screenshots/mcp-pirag-overview-light.png) | ![Features](docs/screenshots/mcp-pirag-features-light.png) | ![KB](docs/screenshots/mcp-pirag-knowledge-light.png) |

| Protocol & Traces | Causal Reasoning |
|:---:|:---:|
| ![Protocol](docs/screenshots/mcp-pirag-protocol-light.png) | ![Causal](docs/screenshots/mcp-pirag-causal-light.png) |

| Heatwave | Overproduction | Cyber Outage |
|:---:|:---:|:---:|
| ![Heatwave](docs/screenshots/multi-agent-run-heatwave.gif) | ![Overproduction](docs/screenshots/multi-agent-run-overproduction.gif) | ![Cyber Outage](docs/screenshots/multi-agent-run-cyber_outage.gif) |

| Adaptive Pricing | Baseline |
|:---:|:---:|
| ![Adaptive Pricing](docs/screenshots/multi-agent-run-adaptive_pricing.gif) | ![Baseline](docs/screenshots/multi-agent-run-baseline.gif) |

</details>

## Documentation

| Document | Contents |
|----------|----------|
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | Full operating manual: every entry point, env var, validation gate, and HPC stage |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layer-by-layer technical reference, backend API, frontend pages |
| [docs/METHODS_REPRO_APPENDIX.md](docs/METHODS_REPRO_APPENDIX.md) | Ordered reproduction steps mapped to paper sections |
| [docs/STATISTICAL_METHODS.md](docs/STATISTICAL_METHODS.md) | Statistical tests, corrections, and CI machinery |
| [agribrain/backend/pirag/mcp/TOOL_INVENTORY.md](agribrain/backend/pirag/mcp/TOOL_INVENTORY.md) | Per-tool MCP inventory (static + runtime) |
| [agribrain/contracts/README.md](agribrain/contracts/README.md) | Smart-contract suite, access model, production checklist |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_METHOD` | `lstm` | Demand forecaster: `lstm` or `holt_winters` |
| `ONLINE_LEARNING` | `false` | Enable REINFORCE policy gradient updates |
| `LLM_PROVIDER` | `template` | RAG answer engine: `template` or `api` |
| `DATA_CSV` | (auto) | Override path to spinach sensor CSV |
| `RAG_CONTEXT_ENABLED` | `true` | Enable MCP/piRAG context integration in agribrain mode |
| `SIM_API_BASE` | (empty) | Base URL for the optional simulation API; unset by default, which leaves the `simulate` MCP tool unregistered |
| `DETERMINISTIC_MODE` | `false` | `true` = exact reproducibility (audit), `false` = 8-source stochastic perturbations (see HOW_TO_RUN.md for the canonical default values) |

### Security/ops flags

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `dev` | Runtime mode (`dev`/`prod`) |
| `REQUIRE_API_KEY` | `false` in dev | Require `x-api-key` header on all routes (except `/health`, `/docs`, `/static`) |
| `APP_API_KEY` | (empty) | API key value when `REQUIRE_API_KEY=true` |
| `ALLOW_LOCAL_WITHOUT_API_KEY` | `true` in dev only | Skip key check for loopback requests (disabled behind reverse proxies via X-Forwarded-For) |
| `ENABLE_DEBUG_ROUTES` | `true` in dev | Enables `/debug/routes` and `/debug/config` |
| `WS_REQUIRE_API_KEY` | `false` in dev | Require websocket auth via `x-api-key` header or `api_key` query param |
| `WS_API_KEY` | (empty) | WebSocket API key (falls back to `APP_API_KEY` if unset) |
| `CORS_ORIGINS` | `*` in dev | Comma-separated allowed origins |
| `CHAIN_REQUIRE_PRIVKEY` | `true` | Require private key for on-chain transactions |

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for the complete variable reference,
including the HPC pipeline toggles and the canonical `STOCH_*` stochastic
defaults.

## System Requirements

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| CPU cores | 4 | 8 | 8 (SLURM HPC) |
| RAM | 16 GB | 32 GB | 32 GB |
| Storage | 2 GB | 5 GB | 5 GB |
| Python | 3.11 | 3.11+ | 3.11 |
| Node.js | 18 | 22 | 22 |

**Execution time estimates (8-core CPU):**

| Task | Time |
|------|------|
| Quick validation run (`DETERMINISTIC_MODE=true`, 1 seed) | ~5 min |
| Single full run (5 scenarios × 19 modes — 8 canonical + 11 §4.7 ablations) | ~80 min (deterministic) |
| Full 20-seed benchmark pipeline | ~90 min (local) / 3-5 h (HPC array) |
| Complete reproduction including stress tests | ~2 h (local) / 6-10 h (HPC end-to-end) |

## Dataset

The system uses IoT sensor telemetry from fresh spinach cold-chain storage, included at
`agribrain/backend/src/data_spinach.csv` (288 records, no preprocessing required).

| Column | Description |
|--------|-------------|
| `timestamp` | ISO 8601 datetime (UTC) |
| `tempC` | Refrigeration temperature (°C) |
| `RH` | Relative humidity (%) |
| `shockG` | Mechanical shock (g) |
| `ambientC` | Ambient temperature (°C) |
| `inventory_units` | Current inventory level (units) |
| `demand_units` | Demand rate (units / step) |
| `quality_preference` | Buyer-side quality preference signal |
| `regulatory_temp_max` | FDA cold-chain temperature ceiling (°C) |

## Data Availability

All artifacts needed to verify the paper's claims are in this repository or
regenerable from it:

- **Committed evidence (verify on a fresh clone, no run needed):** the canonical
  tables, figures, and aggregates under `mvp/simulation/results/` —
  `table1_summary.csv`, `table2_ablation.csv`, `benchmark_summary.json`,
  `benchmark_significance.json`, `paper_benchmark_table.json`, the figure
  `*.png/*.pdf`, `stress_*`, `temporal_stability_*`, and the channel-analysis
  JSONs (`channel_attribution_aggregate.json`, `channel_complementarity_test.json`,
  `channel_saturation_analysis.json`), each with SHA-256 provenance in
  `mvp/simulation/results/artifact_manifest.json`.
- **Regenerable runtime data (gitignored, large):** the per-seed decision ledgers
  (`decision_ledger_h2/`, `decision_ledger_per_seed/`, ~510 MB). The spinach
  sensor dataset is tracked at `agribrain/backend/src/data_spinach.csv`.
  `mvp/simulation/results/` is scratch
  space — treat the committed allowlisted files above as canonical. Reproduce the
  full set with `python mvp/simulation/reproduce_core.py` (end-to-end) or
  `bash hpc/hpc_run.sh` (20-seed benchmark), pinning `PYTHONHASHSEED=0`; the
  §5.8 ledgers come from `mvp/simulation/_run_h2_all.py` then
  `mvp/simulation/benchmarks/aggregate_channel_attribution.py`.
- **Reproduction guide:** see [`HOW_TO_RUN.md`](HOW_TO_RUN.md) and
  [`docs/METHODS_REPRO_APPENDIX.md`](docs/METHODS_REPRO_APPENDIX.md). The
  large runtime artifacts and dataset are available from the authors on request.

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use AGRI-BRAIN in your research, please cite it via
[`CITATION.cff`](CITATION.cff) or the BibTeX below. For paper-grade
reproducibility, also report the exact `git_commit` recorded in
[`mvp/simulation/results/artifact_manifest.json`](mvp/simulation/results/artifact_manifest.json).

```bibtex
@software{sarker2026agribrain,
  title   = {{AGRI-BRAIN}: A Protocol-Mediated Physics-Informed Interoperable
             Multi-Agent Framework for Explainable Perishable Supply Chains},
  author  = {Sarker, Nahid and Kazi, Monzure-Khoda},
  year    = {2026},
  url     = {https://github.com/kprodigi/AGRI-BRAIN},
  version = {1.2.0},
  license = {MIT}
}
```
