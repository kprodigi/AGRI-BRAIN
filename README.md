# AGRI-BRAIN

An adaptive supply-chain intelligence system combining PINN-based spoilage
prediction, LSTM demand forecasting, Social Life-Cycle Assessment (SLCA),
multi-agent coordination, MCP-mediated tool interoperability, physics-informed
RAG knowledge retrieval, and regime-aware contextual policy with online
REINFORCE learning for sustainable food logistics.

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

| System Demo | Live Pipeline Walkthrough |
|:---:|:---:|
| ![Demo Page](docs/screenshots/demo-page-light.png) | ![Pipeline](docs/screenshots/demo-pipeline.gif) |

| Agent Decision Theater | Heatwave Scenario |
|:---:|:---:|
| ![Theater](docs/screenshots/theater-page-light.png) | ![Heatwave](docs/screenshots/agent-theater-heatwave.gif) |

## Architecture Highlights

- **MCP interoperability layer** with 12 registered tools, 12 resources, and 5 prompts
  accessible through JSON-RPC 2.0 protocol with InProcess, Stdio, and SSE transports
- **Physics-informed RAG (piRAG)** with 20-document knowledge base, BM25+TF-IDF hybrid
  retrieval (k=4, 20% retrieval ratio), physics-aware query expansion, and Arrhenius-based
  reranking that surfaces different documents under different physical conditions
- **Causal explanation engine** producing BECAUSE/WITHOUT reasoning with inline [KB:]
  citations, counterfactual probability comparisons, and Merkle-rooted provenance chains
- **8 operating modes** for systematic ablation: static, hybrid RL, no PINN, no SLCA,
  no context, MCP only, piRAG only, and full AGRI-BRAIN
- **LSTM demand forecaster** (numpy-only, 16 hidden units, truncated BPTT)
  with Holt-Winters fallback controlled by `FORECAST_METHOD` env var
- **Holt-Winters yield/supply forecaster** for inventory projection
- **5-agent coordinator** (Farm, Processor, Cooperative, Distributor, Recovery)
  dispatching decisions at lifecycle-stage boundaries
- **Context feature integration** via 5D feature vector (compliance severity, forecast
  urgency, retrieval confidence, regulatory pressure, recovery saturation) with learned
  THETA_CONTEXT weight matrix and SLCA bonus amplification
- **MCP governance override** that mandates rerouting under simultaneous critical
  compliance violation and high spoilage forecast
- **Online REINFORCE learning** of context weights with sign constraints preserving
  domain-justified directions while adapting magnitudes to scenario conditions
- **Keyword extraction** from piRAG passages (thresholds, regulatory references,
  required actions) for human-readable decision evidence
- **Protocol recording** of genuine MCP JSON-RPC interactions during simulation
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
| **Decisions** | Timeline view with role/action filters, decision cards with expandable MCP/piRAG explainability panels (causal BECAUSE/WITHOUT reasoning, 5-axis context feature radar chart, extracted keyword tags, Merkle-rooted provenance chains), analytics sidebar with pie chart, CSV/PDF export |
| **Map** | Leaflet map of South Dakota supply chain nodes with route overlays and live KPI popups |
| **Analytics** | Executive summary banner, interactive cross-scenario tables & charts, ablation study, radar profiles, scenario deep-dive gallery, carbon footprint analysis |
| **Admin** | Six tabs — Policy parameters, Blockchain status & config, Audit log, Scenario runner, Quick Decision, MCP Explorer (tool browser with 12 tools, live resource monitor, prompt template browser, live tool invocation with presets, piRAG knowledge base search, JSON-RPC protocol interaction log) |

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
python generate_results.py    # 5 scenarios x 8 modes (40 episodes)
python generate_figures.py    # 7 publication figures (PNG + PDF)
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
POST /mcp/mcp                - JSON-RPC 2.0 MCP endpoint (tools/call, resources/read, prompts/get)
GET  /mcp/mcp/resources      - List MCP resources
GET  /mcp/mcp/prompts        - List MCP prompts
POST /rag/ask                - Query the piRAG pipeline (physics-informed retrieval)
POST /rag/ingest             - Ingest documents into the piRAG knowledge base
POST /mcp/call               - Call an MCP tool (legacy)
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
        agent_pipeline.py       # Main piRAG pipeline (ingest, retrieve, answer)
        context_builder.py      # Role-specific query construction with scenario terms
        context_to_logits.py    # 5D context features + THETA_CONTEXT weight matrix
        context_learner.py      # Online REINFORCE learning of context weights
        context_eval.py         # Counterfactual evaluator for context impact
        context_provider.py     # Unified context provider interface
        explain_decision.py     # Causal explanation engine (BECAUSE/WITHOUT/citations)
        keyword_extractor.py    # Extract thresholds, regulations, required actions
        physics_reranker.py     # Physics-informed document reranking
        temporal_context.py     # Temporal context window and continuity scoring
        message_enrichment.py   # Enrich inter-agent messages with piRAG context
        dynamic_knowledge.py    # Periodic decision history ingestion into KB
        trace_exporter.py       # Decision trace capture and paper evidence export
        ingestion/              # Document parser, TF-IDF embedder, vector store
        inference/              # LLM abstraction (template + API engines)
        knowledge_base/         # 20 domain documents (regulatory, SOP, SLCA, contingency)
        mcp/                    # MCP implementation
          protocol.py           # JSON-RPC 2.0 MCPServer with tools/resources/prompts
          registry.py           # Tool registry with capability-based discovery
          tool_dispatch.py      # Role-specific tool workflow composition
          context_sharing.py    # Inter-agent shared context store
          transport.py          # InProcess, Stdio, SSE transport layers
          protocol_recorder.py  # Genuine MCP interaction recording
          agent_capabilities.py # Per-agent capability declarations
          resources.py          # MCP resource definitions (telemetry, context, quality)
          prompts.py            # Parameterized query templates with scenario terms
          server.py             # FastAPI REST wrapper for MCP protocol
          tools/                # 12 MCP tool implementations
            compliance.py       # FDA temperature/humidity compliance check
            slca_lookup.py      # SLCA weight and score lookup
            chain_query.py      # Blockchain audit trail query
            spoilage_forecast.py # Arrhenius-Baranyi forward integration
            footprint_query.py  # Energy and water footprint
            pirag_query.py      # Physics-informed KB retrieval via MCP
            explain_tool.py     # Causal explanation generation via MCP
            context_features.py # Context feature vector readout via MCP
            calculator.py       # Safe arithmetic evaluation
            units.py            # Unit conversion
            simulator.py        # Forward simulation proxy
            policy_oracle.py    # Governance access check
        pyrag/                  # Hybrid BM25+dense retriever
        guards/                 # Unit and feasibility guards
        provenance/             # Merkle tree + on-chain anchoring
        tests/                  # 56 tests covering all MCP/piRAG components
    frontend/
      src/
        pages/                  # Ops, Quality, Decisions, Map, Analytics, Admin
        components/ui/          # shadcn/ui component library
        components/explainability/ # ExplainabilityPanel (causal reasoning, radar, keywords, provenance)
        components/mcp/         # McpTab (tool browser, resource monitor, invocation, piRAG search)
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
