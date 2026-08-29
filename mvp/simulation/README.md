# AGRI-BRAIN simulation and analysis package

This package implements the synthetic 72-hour spinach cold-chain benchmark,
its stochastic treatments, paired-seed statistics, publication validation,
and figure generation.

## Primary experimental modes

| Public label | Internal key | Description |
|---|---|---|
| Static | `static` | Fixed routing policy without contextual adaptation |
| No-social-performance | `no_slca` | Context-enabled policy with its social-performance routing term removed |
| Hybrid-RL | `hybrid_rl` | Learned comparator without the full protocol-mediated context interface |
| Mechanistic-only (No-PINN) | `no_pinn` | Full system with the frozen residual removed only from the policy-observed spoilage estimate; the scored DGP outcome remains paired |
| No-external-context | `no_context` | Same learned policy family with MCP and retrieval disabled; peer messages remain |
| MCP-only | `mcp_only` | Structured tool-output channel only |
| Retrieval-only | `pirag_only` | Institutional retrieval channel only |
| AGRI-BRAIN | `agribrain` | Both context channels with online sign-constrained adaptation |

The internal key for Retrieval-only is retained for backward compatibility
with the preserved evidence. It is not the public name of the mechanism.

## Declared endpoints

| Endpoint | Definition and scope |
|---|---|
| ARI | `(1 - waste) * social_performance * (1 - modeled_spoilage_risk)`; the declared Adaptive Resilience Index |
| RLE | Severity-weighted mean of the declared risk-conditional route-utility table on at-risk steps |
| Waste | Waste fraction per routing opportunity |
| Route emissions | Emissions indicator summed over standardized dispatch opportunities; no payload or tonne-kilometre model |
| Social performance | Author-declared social-performance proxy composed of an inverse modeled-emissions term and labour-practice, community-network, and price-information priors |

All endpoints are simulation-derived and are not field measurements.

## Locked publication design

- five scenarios: heatwave, overproduction, cyber outage, adaptive pricing,
  and baseline;
- 20 paired stochastic seeds;
- three adaptation episodes followed by one frozen evaluation episode for
  every learned arm; Static uses the matched fixed evaluation episode;
- H1: AGRI-BRAIN versus No-external-context on ARI in each scenario, with a
  five-test Holm family and a separate 0.005 practical threshold;
- H2: four directional contrasts per scenario—both single channels versus
  No-external-context and AGRI-BRAIN versus each single channel—in one 20-test
  Holm family;
- H3: AGRI-BRAIN stress cells versus paired AGRI-BRAIN reference runs, tested
  by TOST against a ±0.01 ARI equivalence margin in all 25 scenario-stressor
  cells, with verified nonzero exposure;
- BCa intervals, paired Wilcoxon tests, Holm correction for H1 and H2, and
  paired-seed TOST for H3.

The eight primary modes produce 800 retained cells but execute 2,900 episodes.
Three secondary one-factor ablations add 300 retained cells and 1,200 executed
episodes. H3 reuses the primary nominal cells and adds 500 retained stressed
cells and 2,000 executed episodes. Thus the core-plus-H3 total is 1,600 unique
retained cells, 6,100 executed episodes, and 1,756,800 steps—not 800 episodes.

## Main entry points

| Script | Purpose |
|---|---|
| `generate_results.py` | Local scenario × mode development run, isolated under `development_results/` |
| `benchmarks/run_single_seed.py` | One paired benchmark seed |
| `benchmarks/aggregate_seeds.py` | Multi-seed aggregation and inference |
| `benchmarks/run_stress_suite.py` | H3 stress treatment |
| `generate_figures.py` | Publication figure generator |
| `regenerate_figures_from_cache.py` | Figure regeneration from retained evidence |
| `validation/validate_publication_artifacts.py` | Strict publication schema and inference validation |
| `validation/validate_forecasts.py` | Leakage-free internal rolling-origin forecast receipt |
| `analysis/verify_manifest.py` | Literal-byte manifest verification |
| `analysis/build_publication_archive.py` | Deterministic archive construction |
| `sensitivity/run_structural_sensitivity.py` | Hash-bound 100-point structural design and task runner |

## Validate newly generated evidence

The historical `2fd7bff` evidence is incompatible with this aligned source.
After a fresh clean-commit run, checksum and inspect the newly named release
asset, extract a validation copy into `mvp/simulation/results/`, and run:

```bash
python mvp/simulation/analysis/verify_manifest.py --strict-commit
python mvp/simulation/validation/validate_publication_artifacts.py
```

## Regenerate figures from preserved values

```bash
export STRICT_VALIDATION=1
export AGRIBRAIN_GIT_COMMIT=<full-source-commit>
export RUN_TAG=<run-tag>
export BENCHMARK_SEEDS=42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515
export FIGURE_SEED_ROOT=mvp/simulation/results/benchmark_seeds
export FIGURE_OUTPUT_DIR=/absolute/path/to/derived_figures
python mvp/simulation/regenerate_figures_from_cache.py
```

This route does not rerun the expensive seed simulations. The fresh validated
archive remains the evidence source. Identity validation and the complete
20-seed panel are mandatory; relabeled or reformatted figures belong in the
explicit separate derived-output directory.

## New HPC run

```bash
AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh
```

The Slurm scripts enforce a clean source checkout, a run-scoped environment,
paired arrays, dependent publication validation, and checksummed archive
construction. See the root [HOW_TO_RUN.md](../../HOW_TO_RUN.md) for the full
sequence and interpretation limits.

The publication job also regenerates
`forecast_validation_summary.json` and
`forecast_validation_predictions.csv` from the bundled synthetic series. The
validator checks the no-lookahead panel, recomputes every metric, confirms that
validation—not test—RMSE selected Holt-linear demand and persistence supply,
and binds both literal files to the same run tag and source commit. This is
internal synthetic validation, not external predictive validation.

## Separate structural-sensitivity run

```bash
AGRIBRAIN_PARTITION=<partition> \
AGRIBRAIN_SENSITIVITY_ROOT=/shared/scratch/$USER/agribrain-structural \
bash hpc/hpc_sensitivity_run.sh
```

The external root is mandatory and must be visible to every Slurm node. The
runner audits 29 active factors (including `slca_carbon_cap`), creates 100
seed-balanced Latin-hypercube points, and dispatches 3,000 hash-bound tasks.
The exact accounting is 6,500 retained cells, 24,500 executed episodes, and
7,056,000 simulated steps. The publisher preserves all 24,500 lossless episode
archives, all 18,000 adaptation ledgers, and the 6,500 final-evaluation ledgers
used to recompute the retained endpoints. It also binds the 3,000 task results,
worker resource receipts, post-job scheduler accounting, and deterministic
structural CSV/PNG/PDF plus their self-hashed publication receipt. Failed
attempt artifacts are inventoried separately and excluded from canonical
episode and ledger counts. Structural artifacts never populate the core
`results/` directory; the dependent publisher keeps them in the external run
directory described in the root operating guide. The 800-episode label is not
valid for either the core or structural design, and scientific-design changes
still require a new simulation run.
