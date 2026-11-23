# MCP tool inventory

The default registry contains 14 tools. An optional fifteenth (`simulate`)
registers only when `SIM_API_BASE` is configured. Five runtime role-capability
tools bring the publication simulator total to 19.
This document lists each tool and where it is defined, so a reviewer
can audit the count without grepping the codebase.

If a tool is added or removed, edit this file in the same commit;
the test `agribrain/backend/tests/test_mcp_tool_inventory.py`
fails CI when the documented set drifts from the registry.

## Statically registered tools (14 default plus one optional)

These are registered by `get_default_registry()` in
`agribrain/backend/pirag/mcp/registry.py`; all except `simulate`
register unconditionally, and `simulate` registers only when
`SIM_API_BASE` is set (so 14 register by default).

| # | Tool name              | Purpose                                                       |
|---|------------------------|---------------------------------------------------------------|
| 1 | `check_compliance`     | Declared synthetic operating-envelope check (legacy name).    |
| 2 | `slca_lookup`          | Pull declared social-performance priors (legacy key).          |
| 3 | `chain_query`          | Read recent decisions from the active same-episode or local audit ledger (legacy tool name; no blockchain query). |
| 4 | `policy_oracle`        | Allowlist-gate per-tool access; reads `configs/policy.yaml`.  |
| 5 | `calculator`           | Bounded numeric expression evaluator (surplus, deltas).       |
| 6 | `convert_units`        | Unit conversion (mass / volume / temperature).                |
| 7 | `spoilage_forecast`    | Integrate Arrhenius ODE with the declared rational lag factor. |
| 8 | `footprint_query`      | Cumulative energy / water / carbon counters.                  |
| 9 | `pirag_query`          | Institutional retrieval with physics-aware reranking.        |
|10 | `explain`              | Recorded calculation trace plus context-ablation delta.       |
|11 | `context_features`     | Extract the 5-axis institutional context vector.              |
|12 | `yield_query`          | Persistence-default supply-proxy forecast; Holt-linear diagnostic available explicitly. |
|13 | `demand_query`         | Holt-linear demand forecast by confirmatory default; LSTM diagnostic available explicitly. |
|14 | `destination_capacity` | Published capacity of the redistribution outlets and the recovery facility, and the share of the current dispatch each can take. Called by every role under the revised context design. |
|15 | `simulate`             | Forward-simulation HTTP call (only when `SIM_API_BASE` is set; otherwise registered as a known by-design absence so `mcp_registration_status()` reports the gap). |

## Runtime role-capability tools (5)

Registered by the per-role `register_farm_capabilities()`,
`register_recovery_capabilities()`,
`register_cooperative_capabilities()`,
`register_processor_capabilities()`, and
`register_distributor_capabilities()` functions (wrapped by
`register_all_agent_capabilities()`) in
`agribrain/backend/pirag/mcp/agent_capabilities.py` when the agent
coordinator boots. Each is keyed to a role profile and scopes the
tool to that role's reachable state.

| # | Tool name                           | Role        | Purpose                                               |
|---|-------------------------------------|-------------|-------------------------------------------------------|
| 1 | `farm_freshness_assessment`         | farm        | Summarize the farm agent's handled and at-risk counts. |
| 2 | `recovery_capacity_check`           | recovery    | Report the recovery agent's remaining broadcast allowance. |
| 3 | `cooperative_coordination_status`   | cooperative | Report cooperative broadcast allowance and handled steps. |
| 4 | `processor_throughput_status`       | processor   | Summarize handled steps and cumulative simulated waste. |
| 5 | `distributor_route_feasibility`     | distributor | Summarize distributor routing and at-risk counts.     |

## Total

15 static definitions + 5 runtime; at simulation time `SIM_API_BASE`
is empty so `simulate` is absent, giving 14 + 5 = **19 active** (20
only when `SIM_API_BASE` is set). The live FastAPI count is whatever
`get_default_registry().list_tools()` reports at the time
`/mcp/registry/status` is queried; the static minimum is 14 when
`SIM_API_BASE` is empty (the documented production posture for the
simulator subprocess).
