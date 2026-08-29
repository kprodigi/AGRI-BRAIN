# Contributing to AGRI-BRAIN

Thank you for helping improve AGRI-BRAIN. This repository combines an
interactive software system with a prespecified synthetic research benchmark.
Changes to those two surfaces have different evidence requirements, so every
pull request should identify its scope clearly.

## Development setup

Use Python 3.11 for the backend and simulation code:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "agribrain/backend[dev]"
```

For the dashboard, use Node.js 22.12 or later:

```bash
cd agribrain/frontend
npm ci
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for the API, dashboard, local validation,
and Slurm procedures.

## Choose the change category

- **Documentation or repository metadata:** no executable scientific behavior
  changes. Confirm that equations, names, counts, and claim boundaries still
  agree with the locked protocol.
- **Interactive application:** API, dashboard, or optional contract behavior
  that does not alter the publication benchmark. State why the benchmark is
  unaffected and add focused tests.
- **Simulation or statistical semantics:** policy logits, peer messaging,
  context channels, PINN behavior, outcomes, stochastic streams, scenarios,
  learning, hypotheses, estimands, or inferential rules. These changes
  invalidate affected numerical results and require a fresh commit-bound
  simulation treatment.
- **Deterministic publication or validation:** aggregation, figure rendering,
  validation, provenance, or evidence packaging that leaves simulation
  semantics and preserved raw outputs unchanged. Existing completed workers
  may be republished only through the narrowly authorized, byte-preserved
  dual-provenance recovery in `docs/PUBLICATION_RECOVERY.md`; otherwise rerun
  the complete affected pipeline.
- **Publication evidence:** generated tables, figures, or summaries. Accept
  these only from a canonical HPC publisher after every validator passes.

Do not relabel a scientific change as documentation-only. A change that can
alter a simulated decision, raw endpoint, hypothesis, estimand, or inferential
result is a simulation/statistical-semantic change. A deterministic publisher
change that alters derived evidence bytes must be separately provenance-bound;
it neither authorizes editing raw outputs nor by itself implies a simulation
rerun.

## Checks before opening a pull request

Run the focused source and metadata guards:

```bash
python -m pytest \
  mvp/simulation/tests/test_publication_repair.py \
  mvp/simulation/tests/test_publication_evidence_scope.py \
  agribrain/backend/tests/test_metadata_consistency.py -q
```

Run the default Python suite:

```bash
python -m pytest \
  agribrain/backend/tests \
  agribrain/backend/pirag/tests \
  mvp/simulation/tests -q
```

When frontend files change:

```bash
cd agribrain/frontend
npm ci
npm run lint
npm test
npm run build
```

When contract files change:

```bash
cd agribrain/contracts/hardhat
npm ci
npm test
```

CI also runs structural Python lint checks and the slow simulator tests.

## Pull-request requirements

Describe:

1. the problem and the exact files changed;
2. whether scientific behavior or current evidence is affected;
3. the commands run and their outcomes;
4. any new assumptions, parameters, or limitations; and
5. whether a fresh HPC treatment is required.

Keep generated dependencies, local environments, credentials, cluster logs,
private data, draft manuscripts, and unvalidated result artifacts out of Git.
Never include a private key, even a disposable local-chain key, in source,
tests, examples, issue reports, or commit history.

## Scientific claim discipline

AGRI-BRAIN currently evaluates a constructed spinach benchmark. Contributions
must not turn simulation outputs into claims of field effectiveness, measured
waste, lifecycle emissions, demographic equity, or regulatory compliance.
See [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md) before
changing public language.
