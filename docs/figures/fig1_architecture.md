# fig1 — AGRI-BRAIN architecture overview

This file is the **source** for figure 1 in the paper. The Mermaid
diagram below is rendered to SVG and PDF (see `docs/figures/README.md`)
for inclusion in the manuscript. Edit only this file; the rendered
artifacts are regenerated.

```mermaid
flowchart TD
    %% =========================================================
    %% Input layer
    %% =========================================================
    subgraph DATA["Input"]
        IOT[IoT cold-chain telemetry<br/>tempC / RH / shockG / inventory]
        DEMAND[Retail demand stream]
    end

    %% =========================================================
    %% Layer 1: Physics-and-forecast models
    %% =========================================================
    subgraph LAYER1["Layer 1 — Physics & Forecast"]
        PINN[PINN spoilage model<br/>Arrhenius + Baranyi]
        LSTM[LSTM demand forecast<br/>+ residual std]
        HOLT[Holt's linear yield forecast<br/>+ residual std]
    end

    %% =========================================================
    %% Layer 2: Context interoperability
    %% =========================================================
    subgraph LAYER2["Layer 2 — Context Interoperability"]
        MCP[MCP JSON-RPC layer<br/>14 statically registered tools]
        PIRAG[piRAG retrieval<br/>BM25 + TF-IDF<br/>physics-aware reranker]
        EXPLAIN[Explanation engine<br/>BECAUSE / WITHOUT<br/>5-axis context vector]
    end

    %% =========================================================
    %% Layer 3: Decision policy
    %% =========================================================
    subgraph LAYER3["Layer 3 — Decision Policy"]
        AGENTS[5-agent coordinator<br/>farm / processor / cooperative<br/>distributor / recovery]
        POLICY[Regime-aware softmax policy<br/>10-dim phi(s)<br/>+ context modifier]
        REINFORCE[Online REINFORCE learner<br/>sign-constrained THETA updates]
    end

    %% =========================================================
    %% Layer 4: Anchoring and audit
    %% =========================================================
    subgraph LAYER4["Layer 4 — Anchoring & Audit"]
        DAO[AgriDAO governance contracts<br/>SLCARewards / PolicyStore]
        LOGGER[DecisionLogger<br/>per-decision tx hashes]
        MERKLE[ProvenanceRegistry<br/>per-episode Merkle anchors]
    end

    %% =========================================================
    %% Layer 5: Frontend / operator surface
    %% =========================================================
    subgraph LAYER5["Layer 5 — Operator Surface"]
        DASH[FastAPI / React dashboard<br/>Operations / Quality / Decisions<br/>Map / Analytics / Admin / MCP-piRAG]
    end

    %% Data flow
    IOT --> PINN
    DEMAND --> LSTM
    IOT --> HOLT
    PINN --> POLICY
    LSTM --> POLICY
    HOLT --> POLICY
    POLICY --> AGENTS
    MCP --> POLICY
    PIRAG --> MCP
    MCP --> EXPLAIN
    POLICY --> EXPLAIN
    AGENTS --> LOGGER
    AGENTS --> MERKLE
    DAO -.->|policy params| POLICY
    LOGGER --> DASH
    EXPLAIN --> DASH
    POLICY --> REINFORCE
    REINFORCE -.->|delta theta| POLICY

    %% Styling
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a
    classDef physics fill:#dcfce7,stroke:#166534,color:#14532d
    classDef context fill:#fef9c3,stroke:#a16207,color:#713f12
    classDef policy fill:#fce7f3,stroke:#9d174d,color:#831843
    classDef chain fill:#e0e7ff,stroke:#3730a3,color:#312e81
    classDef ui fill:#f3e8ff,stroke:#6b21a8,color:#581c87

    class IOT,DEMAND input
    class PINN,LSTM,HOLT physics
    class MCP,PIRAG,EXPLAIN context
    class AGENTS,POLICY,REINFORCE policy
    class DAO,LOGGER,MERKLE chain
    class DASH ui
```

## Caption (for the paper)

> Figure 1. AGRI-BRAIN system architecture. Cold-chain IoT telemetry
> and retail demand feed Layer-1 physics and forecast models (PINN
> spoilage, LSTM demand, Holt's linear yield). Layer 2 mediates
> context via the Model Context Protocol (MCP, 14 statically
> registered tools) and physics-informed RAG (piRAG, BM25+TF-IDF
> with a temperature-aware reranker), exposing a 5-axis institutional
> context vector to both the decision policy and the explanation
> engine. Layer 3 dispatches decisions through a 5-agent coordinator
> over a 10-dimensional state vector phi(s); an online REINFORCE
> learner refines the routing weights between episodes. Layer 4
> anchors decisions and per-episode Merkle roots on an EVM chain via
> the AgriDAO + DecisionLogger + ProvenanceRegistry contract suite.
> Layer 5 surfaces the full pipeline (real-time telemetry, decision
> queue with explainability panels, audit trail) to operators
> through the FastAPI / React dashboard.

## Provenance

The diagram is editable text; do not commit only the rendered
artifacts. Regenerate via `docs/figures/README.md` after edits. The
artifact-manifest SHA-256 chain hashes the rendered SVG so changes
to the diagram propagate into the publication trace.
