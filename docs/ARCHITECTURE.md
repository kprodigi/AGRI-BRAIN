# AGRI-BRAIN architecture

## Decision flow

1. The synthetic case supplies temperature, humidity, shock, ambient
   conditions, inventory, demand, quality preference, and a scenario-specific
   temperature threshold.
2. A noise-free independent synthetic DGP produces the common spoilage-risk
   outcome used to score every paired arm. A mechanistic Arrhenius first-order
   model produces the policy-side estimate; residual-enabled modes add the
   frozen PINN correction, while No-PINN retains the mechanistic estimate.
3. Four decision-owner roles exchange typed peer messages through an
   in-process coordinator. A cooperative role contributes as an overlapping
   advisory layer rather than a fifth sequential decision owner.
4. Structured MCP tools expose an author-declared operating-envelope signal,
   forecast urgency, recovery saturation, conversion, and local audit
   functions. The operating-envelope output is a simulation feature, not a
   legal or regulatory determination.
5. Institutional retrieval requests the top four items from a constructed
   20-document corpus using BM25 and TF-IDF, expands thermal queries, and
   applies physics-aware reranking and author-declared integrity guards. The
   numeric feasibility
   screen accepts the inclusive range `[-1e9, +1e9]`; this deliberately
   permissive check is an integrity/schema screen, not substantive physical
   validation.
6. Tool and retrieval outputs are normalized into a five-dimensional context
   vector: operating-envelope severity, forecast urgency, normalized fused-rank
   strength, a source-labelled guidance flag, and recovery saturation.
   MCP-derived coordinates and retrieval-derived coordinates remain separate
   through the logit map.
7. The pre-context logits are

   `z_base = Theta phi(s) + b_tau tau + b_SLCA(rho) + b_role + b_peer + b_knee`.

   Here `b_tau = [0.25, 0.05, -0.25]` is action-specific, not a
   softmax-invariant scalar. `b_peer` is computed from the active agent's typed
   inbox using the declared per-message map and is clipped to ±0.30 per action.
   The no-peer secondary arm structurally disables message generation,
   delivery, consumption, and this inbox bias.
8. The external-context modifier is

   `m = clip(S[Theta_MCP psi_MCP + g_r g_p tau_c Theta_RAG psi_RAG], -1, +1)`.

   An author-declared RRF-floor gate `g_r` and temporal continuity `tau_c`
   scale only the retrieval contribution; they do not attenuate MCP. The optional
   hard physics-consistency gate is disabled in the locked confirmatory run, so
   `g_p = 1`; physical consistency is still used for institutional-retrieval
   reranking (internal identifier: piRAG).
   Standard-RAG keeps the author-declared RRF-floor gate but removes query expansion,
   physical-consistency reranking, and the temporal multiplier. No-external-context
   disables both MCP and retrieval while retaining peer communication.
9. During `12 <= hour < 30`, the cooperative advisory role independently
   constructs `m_coop` from the channels enabled for that arm. Ordinarily the
   applied modifier is `clip(0.70 m_primary + 0.30 m_coop, -1, +1)`. If the
   cooperative MCP operating-envelope result contains a critical synthetic
   violation and the primary MCP result does not, the applied modifier becomes
   `clip(m_coop + [-0.20, +0.20, 0.00], -1, +1)`. The recorded legacy
   `cooperative_veto` key denotes only this author-declared benchmark rule; it
   is not a legal, regulatory, or calibrated safety determination.
10. The policy samples from `softmax(z_base + m)`. An author-declared
   probability-gap rule may override the sampled action when its condition is
   met. This action rule is separate from the cooperative modifier composition
   above and is an internal benchmark guardrail, not external governance.
11. The instrumented decision record stores inputs, contextual contributions,
   logits, action, calculation trace, provenance leaves, and a Merkle root.

## Learning boundary

Online REINFORCE updates bounded context, role-policy, and reward-shaping
components for the declared learned modes. Learned comparators receive equal
four-episode budgets: adaptation on episodes 0--2 and no updates during the
retained episode 3. No-external-context keeps the external modifier
structurally absent, so it has no context-matrix updates, while its role-policy
and reward-shaping learners still adapt on episodes 0--2. Explicitly frozen
sensitivity modes record zero updates throughout.

The public calculation trace covers the mapping from retained context
quantities to the policy calculation. It is not a causal account of the
upstream forecaster, spoilage model, or retrieval process.

The independent DGP outcome and the policy-observed spoilage estimate are
stored in separate ledger fields and reconstructed independently. Paired modes
must have the same latent-environment hash; No-PINN must differ from
residual-enabled modes only in its policy-observed spoilage path and downstream
decisions.

## Evaluation outputs

The benchmark reports:

- Adaptive Resilience Index;
- waste fraction per routing opportunity;
- severity-weighted RLE;
- modeled emissions indicator;
- social-performance proxy;
- routing latency and protocol telemetry; and
- reconstructability, provenance, and sign-consistency diagnostics.

These quantities are simulation-derived. They are not measured field waste,
lifecycle emissions, social outcomes, or demographic equity.

## Service layers

- `agribrain/backend/src/`: API, case state, models, routing, agent runtime,
  and optional chain integration;
- the backend retrieval package: corpus ingestion, retrieval, context
  construction, guards, MCP, policy traces, and provenance;
- `agribrain/frontend/`: operations, quality, decisions, analytics,
  institutional retrieval inspection, and administration;
- `mvp/simulation/`: stochastic benchmark, aggregation, statistical tests,
  validators, and figures;
- `hpc/`: commit-stamped Slurm execution and deterministic publication
  packaging.

Legacy internal identifiers are preserved where renaming would break archived
evidence or imports. The public mechanism name is institutional retrieval and
the public single-channel arm name is Retrieval-only.
