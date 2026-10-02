# Reproduction

## Verify the distributed evidence

From the repository root run `python scripts/verify_package.py`. It checks file hashes, the public mode registry, primary and sensitivity coverage, primary paired gains, nominal sensitivity agreement and the relative links in the documentation. This does not require raw archives or repeat the experiment.

## Recompute the primary statistics and figures

`analysis/primary_statistics.py` recomputes the primary comparison from `data/primary/three_mode_endpoints.csv`: per-scenario and overall estimates with 95% BCa intervals, paired gains, one-sided exact Wilcoxon tests with Holm adjustment across the five scenarios, and the H1 decision rule. It then compares every recomputed value with `plotted_estimates.csv`, `paired_seed_ARI.csv` and `overall_outcomes.json`, and exits with status 1 if any differs by more than 1e-9. Reported latency, routing and audit rows have no per-seed data in `data/primary` and are not recomputed.

`analysis/export_dashboard_data.py --check` confirms that the dashboard's results file matches `data/`.

`analysis/make_figures.py --output DIR` regenerates the ARI performance figure (`ari_performance_by_scenario`) and the weight sensitivity figure (`weight_sensitivity`). It first recomputes the plotted values from `three_mode_endpoints.csv` and `data/sensitivity/seed_endpoints.csv` and compares them with the supplied tables, and it stops without drawing if they differ. The result is a re-rendering of the data, not a byte copy of `figures/`. The architecture diagram and the other figures are supplied as final images with their data and are not regenerated.

Both scripts need numpy, scipy and matplotlib at the versions in `analysis/requirements.txt`, which match the locked environment, and together run in under a minute:

```bash
python -m pip install -r analysis/requirements.txt
python analysis/primary_statistics.py
python analysis/make_figures.py --output /absolute/path/to/new-figures
```

## Prepare the scientific environment

The canonical simulation environment uses Python 3.11 and the supplied exact dependency lock. On Linux/HPC:

```bash
cd reproduction
# Load your cluster's Python 3.11 module if needed.
bash setup.sh
source setup.env
source "$AGRI_WEIGHT_VENV/bin/activate"
python test_study.py
```

Use a writable project or scratch directory with sufficient quota. A full study can require over 100 GB before compression. The setup script installs packages into a dedicated environment; it does not modify the source snapshot. Cluster account and partition options are site-specific.

## Repeat the completed sensitivity design

```bash
bash submit.sh --account=YOUR_ACCOUNT --partition=YOUR_PARTITION
```

Replace the site-specific options or omit them if your cluster supplies defaults. Inspect `submit.sh` for output directory and concurrency settings. It submits a pilot, independent repeat and dependency-controlled arrays before collection. It does not contact GitHub. The study has 2,300 tasks; each executes all three modes. Its collector refuses incomplete studies. See `PROTOCOL.md` for the perturbations, normalization and inference.

For one nominal seed–scenario cell, outside Slurm and with the environment activated:

```bash
python study.py prepare --output /absolute/path/to/new-study
python study.py task --output /absolute/path/to/new-study --index 0
```

Indices 0–99 are the 100 nominal seed–scenario cells, each with three modes. The remaining indices are sensitivity settings. These commands create a new run; they do not replace supplied evidence. `analyze.py` expects the complete 2,300-task study, not only nominal cells.

## Rerun the primary comparison

The scripts that executed the primary comparison (No-Context and AGRI-BRAIN, 20 seeds, five scenarios) are at the repository root, in `hpc/` and `mvp/simulation/benchmarks/`:

- `hpc/submit_context_pair.sh` prepares the environment, checks the source and submits a 20-task Slurm array.
- `hpc/no_context_rerun.sh` is the array task. Each task runs one seed across both modes and all five scenarios.
- `hpc/collect_context_pair.sh` validates all outputs and writes the archive.
- `mvp/simulation/benchmarks/rerun_no_context.py` and `verify_context_pair.py` are the per-seed runner and the 20-seed completeness check. Their comparison identifier, `agribrain_standard_rag_vs_no_injection_v1`, is the label stored in the recorded evidence and is kept unchanged.

Run them from a clean Git checkout of this repository, not from `reproduction/source`: the submission preflight also needs `mvp/simulation/experiment_protocol.json` and the complete mode registry, which the trimmed engine copy in `reproduction/source` does not contain. On a Linux/Slurm system with Python 3.11:

```bash
git clone https://github.com/kprodigi/AGRI-BRAIN.git
cd AGRI-BRAIN
module load python/3.11
export AGRIBRAIN_PYTHON_BIN="$(command -v python)"
bash hpc/submit_context_pair.sh --partition=YOUR_PARTITION
```

The submit script creates its virtual environment inside the checkout (`.publication_venvs/`, which `.gitignore` excludes), and the collector checks again that the tree is clean. Pass your actual `--account` and `--partition` options. The script prints the results directory and the exact collection command, `bash hpc/collect_context_pair.sh /absolute/path/to/results`. Static is not rerun by these scripts; see `data/primary/static_compatibility.json`.

A new checkout has a different commit identifier from the evaluated source, so a rerun records its own identity instead of reproducing `83c91ce8592c1d40234e5a6bb2a8ce342412f383`. The original history is in `source.bundle` inside the primary raw archive. The source-identity and spoilage-model artifact checks of the submission preflight pass on this tree, but the full job was not rerun from this repository. The completed sensitivity study uses the separate harness described above.

## Interface and frozen-policy controls

These need the original primary archive, extracted without altering its files. Activate the same environment and set:

```bash
export AGRIBRAIN_PRIMARY_EVIDENCE=/absolute/path/to/context_pair_20260914_193548_83c91ce_results
export AGRIBRAIN_VALIDATION_OUTPUT=/absolute/path/to/new-validation-output
python validation/protocol_validation.py
python validation/run_all_controls.py
python validation/verify_controls.py
python validation/analyze_controls.py
python validation/analyze_roles.py
```

Run these from the repository root. Four frozen-control rollouts are evaluated per seed–scenario pair. The scripts use the distributed engine and explicit output paths. Their path handling has been adapted for portability; original result hashes and source provenance remain in the evidence. The frozen-control experiment is not the separately adapted No-Context comparator.

## Source and runtime identity

`reproduction/SOURCE_MANIFEST.json` checks this distributed source tree. The recorded base commit identifies the source from which it was derived. `provenance/source_distribution.json` lists the public-registry patch. The retained three-mode capability definitions and numerical implementation are unchanged. The original evidence was produced before the registry restriction was applied. The checks in this repository do not establish an independently repeated HPC study.
