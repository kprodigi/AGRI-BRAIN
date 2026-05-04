# MCP tool inventory

The README advertises "14 statically registered tools and 5
additional runtime role-capability tools (19 at simulation time)".
This document lists each tool and where it is defined, so a reviewer
can audit the count without grepping the codebase.

If a tool is added or removed, edit this file in the same commit;
the test `agribrain/backend/tests/test_mcp_tool_inventory.py`
fails CI when the documented set drifts from the registry.

## Statically registered tools (14)

These are registered unconditionally by
`build_default_registry()` in `agribrain/backend/pirag/mcp/registry.py`.

| # | Tool name              | Purpose                                                       |
|---|------------------------|---------------------------------------------------------------|
| 1 | `check_compliance`     | FDA cold-chain envelope check (temperature, humidity).        |
| 2 | `slca_lookup`          | Pull SLCA component scores for a product type.                |
| 3 | `chain_query`          | Read recent on-chain DecisionLogged events (live REST only).  |
| 4 | `policy_oracle`        | Allowlist-gate per-tool access; reads `configs/policy.yaml`.  |
| 5 | `calculator`           | Bounded numeric expression evaluator (surplus, deltas).       |
| 6 | `convert_units`        | Unit conversion (mass / volume / temperature).                |
| 7 | `spoilage_forecast`    | Integrate Arrhenius-Baranyi ODE forward.                      |
| 8 | `footprint_query`      | Cumulative energy / water / carbon counters.                  |
| 9 | `pirag_query`          | piRAG retrieval with physics-aware reranking.                 |
|10 | `explain`              | Causal BECAUSE/WITHOUT explanation engine.                    |
|11 | `context_features`     | Extract the 5-axis institutional context vector.              |
|12 | `yield_query`          | Holt's linear yield/supply forecast + residual std.           |
|13 | `demand_query`         | LSTM (or Holt's linear fallback) demand forecast + std.       |
|14 | `simulate`             | Forward-simulation HTTP call (only when `SIM_API_BASE` is set; otherwise registered as a known by-design absence so `mcp_registration_status()` reports the gap). |

## Runtime role-capability tools (5)

Registered by `register_role_capabilities()` in
`agribrain/backend/pirag/mcp/agent_capabilities.py` when the agent
coordinator boots. Each is keyed to a role profile and scopes the
tool to that role's reachable state.

| # | Tool name                           | Role        | Purpose                                               |
|---|-------------------------------------|-------------|-------------------------------------------------------|
| 1 | `farm_freshness_assessment`         | farm        | Per-batch freshness band classification.              |
| 2 | `recovery_capacity_check`           | recovery    | Composting / animal-feed / food-bank capacity probe.  |
| 3 | `cooperative_coordination_status`   | cooperative | Inter-role coordination state (vetoes, vouchers).     |
| 4 | `processor_throughput_status`       | processor   | Active throughput vs. nominal capacity.               |
| 5 | `distributor_route_feasibility`     | distributor | Route-level feasibility (km, fuel, weather).          |

## Total

14 static + 5 runtime = **19 at simulation time** (matches README
"Architecture Highlights"). The live FastAPI count is whatever
`get_default_registry().list_tools()` reports at the time
`/mcp/registry/status` is queried; the static minimum is 13 when
`SIM_API_BASE` is empty (the documented production posture for the
simulator subprocess).
