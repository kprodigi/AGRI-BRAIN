# AGRI-BRAIN

[![CI](https://github.com/kprodigi/AGRI-BRAIN/actions/workflows/ci.yml/badge.svg)](https://github.com/kprodigi/AGRI-BRAIN/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AGRI-BRAIN is a coordinator-mediated heterogeneous multi-agent framework for
simulated perishable supply-chain routing. It combines typed in-process peer
messages, structured tool outputs, institutional retrieval, mechanistic
spoilage-risk estimates, an independent synthetic-DGP outcome reference, and
a sign-constrained contextual policy interface.
Each instrumented decision is linked to a reconstructable context-to-policy
trace and Merkle-rooted provenance record.

> **Repository status:** version `1.3.0` is a methodology-aligned,
> source-only candidate. No confirmatory benchmark-effect, H1-H3, or
> structural-sensitivity result is currently claimed. The public tag
> `simulation-source-d3286ae` is byte-identical to the source submitted to
> HPC; see [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md).

The publication benchmark is a synthetic 72-hour spinach cold-chain case. It
is designed to test the coordination mechanism under controlled conditions;
it is not a field trial and does not report measured waste, emissions, social,
or demographic outcomes.

## What is implemented

- four sequential decision-owner roles with an overlapping cooperative
  advisory layer;
- typed peer messages connected to the policy through a separate clipped
  logit term;
- structured MCP tools and institutional retrieval connected through
  channel-separated context features;
- a mechanistic spoilage estimate with an optional frozen neural residual,
  evaluated against a separate synthetic data-generating process;
- action-specific operating-regime effects, exact forward-policy gradients,
  paired stochastic streams, and prespecified H1/H2/H3 inference;
- lossless decision ledgers, calculation traces, local Merkle evidence, and
  commit-bound HPC publication validation; and
- a FastAPI service, React dashboard, and optional local smart-contract
  prototype in addition to the research benchmark.

### Policy-connected information channels

| Channel | Mathematical connection to policy | Confirmatory behavior |
|---|---|---|
| Peer communication | Separate additive `b_peer` logit term, clipped to +/-0.30 per action | Active except in the explicit no-peer secondary arm |
| MCP tools | `Theta_MCP psi_MCP` contribution to the external modifier | Active in modes whose capability contract enables MCP |
| Institutional retrieval | `g_r g_p tau_c Theta_RAG psi_RAG` contribution | Active in retrieval-enabled modes; gates scale retrieval only |

The binary regime flag adds the action-specific bias
`b_tau = [0.25, 0.05, -0.25]`, so it changes relative softmax probabilities.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete decision flow.

## Methodology-alignment status

This source tree is aligned to the locked protocol in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Its changes affect simulation
semantics, so results from the historical `2fd7bff` run must not be attributed
to this code. No confirmatory benchmark-effect, H1-H3, or structural-
sensitivity result is currently claimed from this tree. A new result set is
accepted only when it is generated from a clean commit of this source and
passes the raw-input, inference, ledger, environment, and artifact validators.

The locked core design uses five scenarios and 20 paired seeds. Learned arms
run three adaptation episodes followed by a frozen evaluation episode; Static
runs the matched fixed evaluation episode only. The complete core, H3, and
secondary-ablation treatment has 1,600 retained evaluation cells, 6,100
executed episodes, and 1,756,800 simulated decision steps. It must not be
described as “800 stochastic simulation episodes.” A separate 100-point
structural-sensitivity design adds 6,500 retained cells and 24,500 executed
episodes. Valid core, H3, and secondary-ablation evidence is required to retain
6,100 lossless episode archives, 4,500 adaptation ledgers, and 1,600 final-
evaluation ledgers. Valid structural evidence is required to retain 24,500
episode archives, 18,000 adaptation ledgers, and 6,500 final-evaluation
ledgers. Changing the scientific design, including changing these episode or
cell counts, requires a new simulation run; it cannot be achieved by editing
documentation or regenerating tables and figures.

The old `2fd7bff` evidence receipt remains under `provenance/` solely as
historical lineage. It is not validating evidence for this methodology-aligned
source.

The concise public claim boundary is documented in
[docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md).

## Dataset

The bundled case is a constructed 288-row, 15-minute spinach telemetry series.
It is synthetic benchmark input, not a field dataset. Its columns, in file
order, are:

| Column | Meaning |
|---|---|
| `timestamp` | ISO 8601 observation time |
| `tempC` | Product temperature in degrees Celsius |
| `RH` | Relative humidity in percent |
| `shockG` | Synthetic handling-shock acceleration |
| `ambientC` | Ambient temperature in degrees Celsius |
| `inventory_units` | Synthetic inventory-unit count |
| `demand_units` | Synthetic demand-unit count |
| `quality_preference` | Declared unitless quality preference |
| `regulatory_temp_max` | Scenario temperature limit in degrees Celsius |

## Install and verify

Create a Python 3.11 environment and install the locked dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r agribrain/backend/requirements-lock.txt
python -m pip install --no-deps -e agribrain/backend
```

Run the full core treatment from a clean commit with the Slurm workflow:

```bash
AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh
```

The workflow creates a commit-stamped run tag and runs the validators before
packaging any evidence. Structural sensitivity has its own fail-closed runner
under `mvp/simulation/sensitivity/`. See [HOW_TO_RUN.md](HOW_TO_RUN.md) for the
complete operating and validation sequence.

Run that separate treatment only on Slurm, with an absolute shared-scratch
root outside this repository:

```bash
AGRIBRAIN_PARTITION=<partition> \
AGRIBRAIN_SENSITIVITY_ROOT=/shared/scratch/$USER/agribrain-structural \
bash hpc/hpc_sensitivity_run.sh
```

It uses 100 seed-balanced Latin-hypercube points and 29 active factors
(including `slca_carbon_cap`), yielding exactly 6,500 retained cells, 24,500
executed episodes, and 7,056,000 simulated steps. A valid completed archive is
required to contain the 3,000 hash-bound task results, all 24,500 lossless
episode archives, all 18,000 adaptation ledgers, all 6,500 final-evaluation
ledgers used to recompute endpoints, worker runtime receipts, post-job
scheduler accounting, and the deterministic structural CSV/PNG/PDF with its
self-hashed publication receipt.
Failed attempts, if any, are retained and inventoried separately and are not
included in the canonical episode or ledger counts. These artifacts remain in
the external run directory; they are never canonical core files under
`mvp/simulation/results`.

## Quick start for the interactive system

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "agribrain/backend[dev]"
python -m uvicorn src.app:API --port 8100
```

In a second terminal:

```bash
cd agribrain/frontend
npm ci
npm run dev
```

The API runs at `http://127.0.0.1:8100`; the dashboard runs at
`http://127.0.0.1:5173`.

## Repository layout

| Path | Purpose |
|---|---|
| `agribrain/backend/` | API, agent coordination, policy, retrieval, and provenance |
| `agribrain/frontend/` | React/Vite interactive dashboard |
| `agribrain/contracts/` | Optional local smart-contract prototype |
| `mvp/simulation/` | Benchmark, aggregation, inference, validation, and figures |
| `hpc/` | Commit-stamped Slurm orchestration and evidence packaging |
| `docs/` | Architecture, methods, statistics, claims, and release guidance |
| `provenance/` | Superseded lineage retained as explicitly non-current evidence |

Legacy internal directory, class, route, tool, and mode identifiers remain in
the code where changing them would break compatibility. Public-facing
terminology is "institutional retrieval" for the mechanism and
"Retrieval-only" for the corresponding experimental arm.

## Documentation

| Document | Use |
|---|---|
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | Local setup, tests, application, HPC, and evidence validation |
| [Publication recovery](docs/PUBLICATION_RECOVERY.md) | Fail-closed, no-simulation-rerun recovery after a completed run's publisher fails |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Locked scientific design and accounting |
| [Architecture](docs/ARCHITECTURE.md) | Decision flow, channel separation, learning, and evidence layers |
| [Methods appendix](docs/METHODS_REPRO_APPENDIX.md) | Exact reproducibility specification |
| [Statistical methods](docs/STATISTICAL_METHODS.md) | H1, H2, H3, effect sizes, and multiplicity |
| [Claims and limitations](docs/CLAIMS_AND_LIMITATIONS.md) | What the synthetic benchmark does and does not establish |
| [Source provenance](SOURCE_PROVENANCE.md) | Public tag, HPC commit/tree mapping, and result status |
| [Contributing](CONTRIBUTING.md) | Development workflow and scientific-change rules |
| [Release procedure](docs/RELEASE.md) | Versioning and validated evidence release |
| [GitHub publishing](GITHUB_UPDATE_INSTRUCTIONS.md) | Commands for pushing this prepared repository |

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). After validated
evidence exists, cite the software version together with its exact simulation
source commit, publication source commit when distinct, run tag, and
evidence-archive checksum. The code is released under the
[MIT License](LICENSE).

Security reports should follow [SECURITY.md](SECURITY.md); general
contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).
