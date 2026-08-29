# Methods and Reproducibility Appendix

**Project:** AGRI-BRAIN  
**Repository:** <https://github.com/kprodigi/AGRI-BRAIN>
**Run identity:** report the repository commit and the SHA-256 manifest from
`mvp/simulation/results/artifact_manifest.json`.

## Canonical publication run

The confirmatory run uses 20 prespecified seeds:

`42, 1337, 2024, 7, 99, 101, 202, 303, 404, 505, 606, 707, 808, 909,
1010, 1111, 1212, 1313, 1414, 1515`.

On a Slurm cluster, submit the benchmark and stress arrays from the repository
root:

```bash
AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh
```

`hpc_run.sh` and every dependent Slurm job source
`hpc/publication_env.sh` and run the environment and clean-checkout validators
before doing work. The orchestrator creates a new
`.publication_venvs/<RUN_TAG>` environment for each submission and refuses to
reuse or overwrite an existing path. Every stage verifies normalized package
uniqueness, exact applicable lock versions, the backend version, and the absence
of unexpected runtime distributions. The canonical treatment also sets
`MCP_RATE_LIMITS=disabled`, so in-process tool calls cannot change with elapsed
wall time or global invocation order. Numerical-library thread counts are fixed
at one to avoid node-dependent BLAS/OpenMP reduction order and oversubscription.
Each context-enabled episode records dispatcher failures, JSON-RPC errors, real
tool `isError` responses, and recorder truncation. Under
`STRICT_VALIDATION=1`, any such genuine failure or incomplete record aborts the
run. The H3 fault dose replaces tool results only after successful protocol
calls, so it remains a declared stress treatment rather than a tool error.
`hpc_seed.sh` runs one seed over all five scenarios and the eleven locked
publication modes (eight primary plus three secondary one-factor ablations).
`hpc_stress.sh` runs one scenario over the same seed list and five declared
stressors. After both arrays succeed, `hpc_publish.sh` aggregates the seed-level
files, computes H1-H3 statistics, generates figures and evidence tables, runs
the strict validators, records the actual interpreter, platform, packages, and
canonical settings in `publication_environment.json`, builds and verifies the
manifest, and creates the publication archive. Publication artifacts must
not be used if any dependent job or validator fails.

The figure stage emits ten exact 800-DPI PNG/vector-PDF pairs. Promotion is
transactional and fails unless every PNG meets the resolution and color-mode
contract and every one-page PDF contains vector primitives and embedded
TrueType-compatible fonts without Type 3 fonts or raster image objects. Figure
provenance binds the accessible semantic palette and redundant marker,
line-style, hatch, and position encodings, the resolved font-file hash, renderer
versions, literal inputs, and literal output hashes.

The same episode-indexed exogenous streams are used across modes within each
scenario. Learned modes adapt on episodes 0--2 and retain episode 3 as a frozen,
no-update evaluation. Learner state persists only across the three adaptation
episodes and is reset before the next scenario-mode-seed block.
Operational random draws are source- and counter-keyed within each environmental
stream, so a conditional branch cannot shift later temperature, humidity,
demand, inventory, transport, or telemetry-lag draws in another arm. The 25%
demand perturbation is applied to the exogenous policy-observed demand series
before the rolling forecast, Bollinger regime flag, and price signal are
computed; it is not added post hoc to the forecast. Adaptive-pricing Gaussian
scenario noise is drawn from the explicit scenario/seed generator rather than a
fixed internal seed.
Decision history is not learner state: each episode receives a fresh in-memory
`DecisionLedger`, and `chain_query` sees only earlier decisions in that episode.
The evidence pipeline now retains every executed episode as a lossless,
content-addressed episode archive, retains a separate adaptation ledger for
each learned-arm adaptation episode, and retains the final-evaluation JSONL
ledger used for endpoint recomputation. Across the core, H3, and secondary-
ablation treatments this is exactly 6,100 episode archives, 4,500 adaptation
ledgers, and 1,600 final ledgers. Across the separate structural treatment it
is exactly 24,500 episode archives, 18,000 adaptation ledgers, and 6,500 final
ledgers.
Stochastic policy-temperature noise, injected context faults, dynamic feedback,
and blockchain submission are disabled in the confirmatory benchmark unless a
stress cell explicitly declares the relevant fault.

## Models and runtime choices

- The scored spoilage-risk outcome is the noise-free trajectory from the
  declared independent synthetic DGP and is identical across paired modes.
  Policy observations use a mechanistic Arrhenius first-order ODE with a
  declared rational lag factor, plus a frozen residual where the mode enables
  it. The frozen checkpoint is fit on 36
  synthetic trajectories with a locked 24/6/6 train/validation/test split;
  dataset and checkpoint hashes are recorded. This is neither empirical nor
  external shelf-life validation. The `no_pinn` arm is a paired
  mechanistic-only policy-estimator ablation: all other policy, context, peer,
  social-proxy, learning-budget, random-stream capabilities, and the scored
  DGP outcome match AGRI-BRAIN. The base
  `k_ref=0.0021 h^-1` and `Ea/R=8000 K` receive one counter-keyed
  episode draw each: multipliers `1+Normal(0,0.20)` and
  `1+Normal(0,0.14)`, with floors `1e-6 h^-1` and `100 K`. Paired modes share
  those draws.
- One-step rolling-origin validation on the internal 288-row synthetic series
  selects non-seasonal Holt-linear for demand and persistence for the
  inventory-based supply proxy by validation RMSE. The test segment is report
  only. LSTM and Holt-linear supply remain diagnostics; the exercise is not
  external predictive validation, and the supply ranking reverses on test.
- Green-AI energy and water values are activity-based estimates, not hardware
  telemetry. Only `coordinator.step` action-selection wall time is measured;
  scenario construction, forecast preparation, outcome scoring, learner
  post-step updates, artifact I/O, and idle allocation are outside the timer.
  The measured seconds are multiplied by the declared 10 W nominal power and
  1.8e-6 L/server-second water rate. These values are therefore not a whole-job
  footprint and are not comparable to the modeled transport-emissions outcome.
  Historical per-decision constants are retained only under explicit
  `per_step_proxy` labels and are not reported as elapsed-time estimates.
- Exploratory Carbon Efficiency is computed within each seed as episode mean
  ARI divided by the episode-summed modeled transport-emissions indicator, in
  ARI per modeled kg CO2-eq. No factor of 1,000 is applied. Its BCa interval is
  bootstrapped from the paired within-seed ratios, rather than approximated
  from the marginal ARI and carbon intervals. Canonical figures report the two
  outcomes directly; the ratio is not a confirmatory endpoint.
- Institutional retrieval requests the top four items from BM25 plus TF-IDF
  over a constructed 20-document synthetic corpus, with physics-aware query
  expansion and reranking. Its feasibility guard accepts the inclusive range
  `[-1e9,+1e9]`; that deliberately permissive bound is an integrity/schema
  screen rather than substantive physical validation. The deterministic
  template answer engine is used; the publication run makes no external LLM
  call.
- During `12 <= hour < 30`, a context-enabled arm composes the primary and
  cooperative external-context modifiers with weights 0.70 and 0.30. If the
  cooperative MCP operating-envelope result is critical and the primary MCP
  result is not, the primary modifier is replaced by the cooperative modifier
  plus `[-0.20,+0.20,0.00]` for `[cold-chain, local-redistribute,recovery]`.
  The composed vector is clipped to `[-1,+1]`. This author-declared benchmark
  adjustment is distinct from the probability-gap action rule and is not a
  legal or calibrated safety rule.
- Within each episode, every decision is included in the active ledger and its
  Merkle root. The retained JSONL is the final episode for each arm. On-chain
  submission is optional and disabled in the confirmatory run, so transaction
  claims require separate deployment evidence.

## Primary modes and diagnostics

The eight primary modes are `static`, `hybrid_rl`, `no_pinn`, `no_slca`,
`no_context`, `mcp_only`, `pirag_only`, and `agribrain`. The three secondary one-factor modes
are `agribrain_standard_rag`, `agribrain_no_peer`, and
`agribrain_sign_unconstrained`. The residual is supported only by the declared
independent synthetic DGP, not observed target labels. Structural sensitivity is a separately
labelled 100-point Latin-hypercube design over active declared parameters; it
is not a probability distribution over real-world uncertainty. Neither the
core treatment nor the structural treatment is an “800 stochastic simulation
episodes” design. Changing the declared seeds, scenarios, modes, stressors,
factor points, episode schedule, or simulation semantics requires rerunning the
affected scientific treatment. Deterministic re-export of the structural
CSV/PNG/PDF from its hash-valid analysis JSON does not alter its statistics.

Both Slurm publishers capture post-job scheduler accounting for every declared
simulation worker and bind that accounting into final evidence. Failed worker
attempts and their artifacts are retained separately for diagnosis and audit;
they are excluded from the canonical successful episode and ledger counts. The
structural publisher additionally emits a deterministic machine-readable CSV,
PNG, and PDF with a self-hashed receipt bound to the analysis JSON, source
commit, design hash, manifest hash, literal artifact hashes, accessible style
record, and measured PNG/PDF quality checks. Its scenario-grouped panels are a
display transformation only: all 5 H1, 25 H2, and 25 H3 summary cells are
exported without recomputation.

## Hypotheses and inference

The seed is the inferential unit; hourly observations are not treated as
independent replicates. Full definitions and formulas are in
`docs/STATISTICAL_METHODS.md`.

- **H1 (external-context value):** paired per-seed AGRI-BRAIN minus No-external-context
  ARI differences, one test per scenario. Directional Wilcoxon signed-rank tests
  use Holm adjustment across the five primary tests. Paired effect sizes and
  95% BCa bootstrap confidence intervals are reported; any percentile fallback
  is labelled per interval and cannot support the practical-margin claim. A
  practical claim of at least 0.005 ARI additionally requires the paired 95%
  **BCa** interval lower bound to exceed 0.005. Other method/metric contrasts are
  secondary and use Benjamini–Yekutieli adjustment within scenario.
- **Frozen-residual ablation:** paired per-seed AGRI-BRAIN minus `no_pinn` ARI
  differences are tested directionally by Wilcoxon signed-rank in a separate
  five-scenario Holm family. This prespecified ablation does not change or join
  the H1 or H2 families.
- **H2 (joint channel contribution):** each scenario contains four directional
  paired contrasts: MCP-only minus No-external-context, Retrieval-only minus
  No-external-context, full AGRI-BRAIN minus MCP-only, and full AGRI-BRAIN minus
  Retrieval-only. All 20 ARI tests form one Holm family; a universal statement
  requires all 20 cells to pass. The generated
  `h2_directional_evidence.csv` contains all 20 rows and the raw/adjusted tests,
  BCa intervals, paired effects, and support flags. Superadditivity is reported separately as the
  exploratory interaction `Full - MCP - Retrieval + No-external-context`. A separate
  conditional observed-state analysis masks MCP-derived and retrieval-derived
  context feature groups in recorded full-system decisions, summarizes results
  within seed, and then aggregates across seeds. Because it reuses observed
  retrieval and guard outputs, this diagnostic does not represent disabled
  communication channels.
- **H3 (sensing and tool-channel robustness):** AGRI-BRAIN is evaluated in 25 paired cells
  (five scenarios by five stressors). The primary endpoint is the seed-level ARI
  difference from the unstressed run. Equivalence is tested with two one-sided
  tests at a prespecified margin of ±0.01 ARI. A cell passes only when both
  one-sided tests reject at α=0.05. Ninety- and 95-percent confidence intervals,
  seed count, and absolute metric deltas are retained. The margin is a
  prespecified simulation tolerance, not an externally validated deployment
  threshold. The nominal endpoint is the identical frozen episode-3 cell from
  the primary benchmark and is not rerun. Each stressed arm adapts on episodes
  0--2 and is evaluated without updates on episode 3. The MCP reliability
  setting remains the canonical disabled posture in both nominal and stressed
  cells; only the declared fault injection is toggled. The fixed rule
  `int(hour) % 11 == 0` creates 28 scheduled fault
  opportunities on a complete 288-step trace. Generated artifacts separately
  retain scheduled opportunities, observed trigger steps, and the number of
  invoked tool results replaced; actual exposure can be lower when MCP is
  unavailable, including during the declared cyber outage. Global H3 support
  requires all 25 TOST cells and verified nonzero treatment exposure.

No validator requires AGRI-BRAIN to win, requires a minimum effect, or encodes a
preferred ordering. Validators check completeness, balanced panels, numeric
finiteness, construct bounds, inferential schema, source identity, and artifact
hashes.

## Artifacts

The canonical result directory contains at least:

- `table1_summary.csv`: eight primary modes by five scenarios, with seed-level
  confidence intervals;
- `table2_ablation.csv`: compact six-mode architectural ablation;
- `benchmark_summary.json` and `benchmark_significance.json`;
- `stress_summary.json`, `stress_passfail.csv`, and `stress_h3_test.json`;
- `forecast_validation_summary.json` and
  `forecast_validation_predictions.csv`, the commit-bound internal synthetic
  rolling-origin receipt;
- channel-attribution and H2 test artifacts;
- figure PNG/PDF pairs and paper-evidence exports;
- 55 retained primary/secondary final-episode decision ledgers per seed and
  policy traces, plus separately partitioned H3 stressed ledgers;
- `publication_environment.json` and `artifact_manifest.json`.

The manifest stores the source commit and SHA-256 hash for each generated
artifact. The tracked `agribrain/backend/requirements-lock.txt` is consumed by
the publication job; it is not generated by that job. Report the actual Python
version, platform, installed packages, canonical environment values, and
lockfile hash from `publication_environment.json`, not from development-machine
examples or the lockfile header alone.

## Scope limitations

The benchmark is simulation-based, uses one 288-row spinach sensor trace and
constructed scenario perturbations, and does not constitute field validation,
external crop/region validation, food-safety certification, causal inference,
or validation of the social-performance proxy as an empirical life-cycle
assessment. Wall-
clock latency is hardware-dependent and descriptive. Claims in the paper must
be limited to the validated artifact set and must carry these scope conditions.
