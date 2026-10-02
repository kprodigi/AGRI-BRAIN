# AGRI-BRAIN architecture

This page summarizes the decision pipeline that was evaluated. The paths below point to the code that implements each step.

## Evaluated modes

| Mode | Behavior | Tools and retrieval | Peer messages |
|---|---|---|---|
| `static` | Always continues the cold chain | No | No |
| `no_context` | Adaptive policy from observations, forecasts, base-policy learning and social shaping | No | No |
| `agribrain` | `no_context` plus the context package: tool outputs, standard retrieval, peer messages, context-matrix learning and the context-linked guard | Yes | Yes |

The definitions are in `agribrain/backend/src/models/mode_capabilities.py`. The mode registry in the full source tree also lists earlier ablation modes; the study evaluates only the three above, and `reproduction/source` restricts its registry to them.

## Decision flow

1. A synthetic 72-hour case supplies temperature, humidity, shock, inventory and demand at 15-minute steps (288 decisions per episode). Distances are 120 km for cold-chain continuation, 45 km for local redistribution and 80 km for recovery (`src/models/policy.py`).
2. A mechanistic Arrhenius-type spoilage model, corrected by a frozen bounded neural residual, gives the policy's remaining-quality estimate. A separate noise-free synthetic process scores the outcome, so paired modes share the scored trajectory.
3. Four decision-owner roles (farm, processor, distributor, recovery) exchange typed peer messages through an in-process coordinator; a cooperative role contributes during hours 12 to 30. The summed peer bias is clipped to 0.30 per action (`src/agents/message.py`).
4. Operating-envelope, spoilage-forecast and ledger-query tools supply three components of a five-component context vector (envelope severity, forecast urgency, recovery saturation). Retrieval over 20 fixed documents (BM25 and TF-IDF, reciprocal rank fusion) supplies retrieval-rank strength and a source-labeled guidance flag, which pass a retrieval gate with floor 0.0246 (`pirag/context_to_logits.py`).
5. The context-weight matrix maps the vector to a bounded adjustment of the three routing logits. The initial matrix is `THETA_CONTEXT` (`pirag/context_to_logits.py`, lines 59 to 64).
6. During hours 12 to 30 the applied adjustment blends 0.7 of the primary role's modifier with 0.3 of the cooperative's; if only the cooperative's operating-envelope check is critical, a fixed offset of [-0.20, +0.20, 0.00] is added (`src/agents/coordinator.py`).
7. The policy samples a route from the softmax of the base logits plus the adjustment. A governance rule replaces the sampled action with local redistribution when the cold-chain probability is below a ceiling and redistribution leads cold chain by more than a minimum gap (`governance_override_applies`, `src/models/action_selection.py`). Both thresholds are synthetic hyperparameters.
8. Each decision record stores the state, base logits, raw and gated context, peer bias, probabilities, the random draw, the sampled and final actions and a SHA-256 leaf. Leaves form a Merkle root per episode. Ledger anchoring is optional and was disabled in the study.

## Learning

Adaptive modes adapt for three episodes and are evaluated in a fourth, frozen episode. Context-matrix updates are sign-constrained and bounded; `no_context` has no context matrix, so only its base-policy and reward-shaping learners adapt.

## Outputs

Adaptive Resilience Index, modeled waste, modeled transport emissions, social performance, Reverse Logistics Efficiency and coordinator decision latency. These are simulation outputs, not field measurements.

## Source layout

- `agribrain/backend/src/`: API, case state, models, agent runtime and optional chain integration.
- `agribrain/backend/pirag/`: retrieval, context construction, guards, MCP tools and provenance.
- `agribrain/frontend/`: dashboard.
- `agribrain/contracts/`: optional Solidity prototypes with Hardhat tests.
- `mvp/simulation/`: simulator, spoilage-model assets and the primary-run scripts.
- `hpc/`: Slurm scripts and validators for the primary run.
- `reproduction/`: harness for the weight-sensitivity study.
