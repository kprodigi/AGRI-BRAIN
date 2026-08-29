# AGRI-BRAIN

AGRI-BRAIN is a coordinator-mediated heterogeneous multi-agent framework for
simulated perishable supply-chain routing. It combines typed in-process peer
messages, structured tool outputs, institutional retrieval, mechanistic
spoilage-risk estimates, an independent synthetic-DGP outcome reference, and
a sign-constrained contextual policy interface.
Each instrumented decision is linked to a reconstructable context-to-policy
trace and Merkle-rooted provenance record.

The publication benchmark is a synthetic 72-hour spinach cold-chain case. It
is designed to test the coordination mechanism under controlled conditions;
it is not a field trial and does not report measured waste, emissions, social,
or demographic outcomes.

## Methodology-alignment status

This source tree is aligned to the locked protocol in
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Its changes affect simulation
semantics, so results from the historical `2fd7bff` run must not be attributed
to this code. No numerical publication result is currently claimed from this
tree. A new result set is accepted only when it is generated from a clean
commit of this source and passes the raw-input, inference, ledger, environment,
and artifact validators.

The locked core design uses five scenarios and 20 paired seeds. Learned arms
run three adaptation episodes followed by a frozen evaluation episode; Static
runs the matched fixed evaluation episode only. The complete core-plus-H3 design has
1,600 retained evaluation cells, 6,100 executed episodes, and 1,756,800
simulated decision steps. It must not be described as “800 stochastic
simulation episodes.” A separate 100-point structural-sensitivity design adds
6,500 retained cells and 24,500 executed episodes. The complete core-plus-H3
evidence retains 6,100 lossless episode archives, 4,500 adaptation ledgers,
and 1,600 final-evaluation ledgers. The separate structural evidence retains
24,500 episode archives, 18,000 adaptation ledgers, and 6,500 final-evaluation
ledgers. Changing the scientific design, including changing these episode or
cell counts, requires a new simulation run; it cannot be achieved by editing
documentation or regenerating tables and figures.

The old `2fd7bff` evidence receipt remains under `provenance/` solely as
historical lineage. It is not validating evidence for this methodology-aligned
source.

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

## Run and verify the aligned benchmark

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
executed episodes, and 7,056,000 simulated steps. Its verified archive contains
the 3,000 hash-bound task results, all 24,500 lossless episode archives, all
18,000 adaptation ledgers, all 6,500 final-evaluation ledgers used to recompute
endpoints, worker runtime receipts, post-job scheduler accounting, and the
deterministic structural CSV/PNG/PDF with its self-hashed publication receipt.
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
npm install
npm run dev
```

The API runs at `http://127.0.0.1:8100`; the dashboard runs at
`http://127.0.0.1:5173`.

## Repository layout

```text
agribrain/backend/       API, policy, institutional retrieval, provenance
agribrain/frontend/      interactive dashboard
agribrain/contracts/     optional local smart-contract implementation
mvp/simulation/          benchmark, aggregation, analysis, and figures
hpc/                     commit-stamped Slurm orchestration and validation
docs/                    architecture and statistical documentation
provenance/              superseded repair lineage, explicitly non-evidentiary
```

Legacy internal directory, class, route, tool, and mode identifiers remain in
the code where changing them would break compatibility. Public-facing
terminology is "institutional retrieval" for the mechanism and
"Retrieval-only" for the corresponding experimental arm.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). After a fresh
validated treatment exists, cite the software version together with its exact
simulation source commit, run tag, and evidence-archive checksum. The code is released under the
[MIT License](LICENSE).
