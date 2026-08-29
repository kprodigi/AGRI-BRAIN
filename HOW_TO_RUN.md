# AGRI-BRAIN operating and reproduction guide

## 1. Requirements

- Python 3.11 (the publication workflow rejects other Python minors)
- Node.js 22.12 or later for the dashboard only (required by the locked Vite
  toolchain)
- Git
- A Slurm cluster only when running a new full 20-seed treatment

The publication workflow fixes BLAS-related thread counts to one and records
the interpreter, installed-package versions, environment contract, platform,
and source hashes. This is a version-resolved runtime inventory, not a claim
of byte-identical wheels, BLAS binaries, or a container image.

## 2. Install the backend

For ordinary development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "agribrain/backend[dev]"
```

On Windows PowerShell, use:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e "agribrain/backend[dev]"
```

For the closest local match to the publication environment:

```bash
python3.11 -m venv .venv-publication
source .venv-publication/bin/activate
python -m pip install --upgrade pip
python -m pip install -r agribrain/backend/requirements-lock.txt
python -m pip install --no-deps -e agribrain/backend
```

## 3. Run the API and dashboard

Start the API from the repository root:

```bash
python -m uvicorn src.app:API --host 127.0.0.1 --port 8100
```

Start the dashboard in another terminal:

```bash
cd agribrain/frontend
npm ci
npm run dev
```

Load the bundled synthetic telemetry trace and verify health:

```bash
curl -X POST http://127.0.0.1:8100/case/load
curl http://127.0.0.1:8100/health
```

## 4. Run tests

The focused publication-integrity and metadata guards are:

```bash
python -m pytest \
  mvp/simulation/tests/test_publication_repair.py \
  mvp/simulation/tests/test_publication_evidence_scope.py \
  agribrain/backend/tests/test_metadata_consistency.py -q
```

The default backend and simulation test selection excludes tests marked
`slow`:

```bash
python -m pytest agribrain/backend/tests agribrain/backend/pirag/tests \
  mvp/simulation/tests -q
```

Internal paths and identifiers retained for compatibility may use historical
names. User-facing output refers to institutional retrieval and to the
Retrieval-only arm.

## 5. Evidence compatibility boundary

The historical `2fd7bff` archive is not compatible with this
methodology-aligned source. Do not use it to validate or populate results in
this tree. A new treatment must be run from a clean commit first.

For a newly generated archive, inspect its member names and verify the
run-issued SHA-256 before extraction. Then validate literal bytes, membership,
schemas, H1/H2 inference, H3 equivalence, and retained evaluation ledgers:

```bash
python mvp/simulation/analysis/verify_manifest.py --strict-commit
python mvp/simulation/validation/validate_publication_artifacts.py
python hpc/validate_decision_ledgers.py
```

The manifest's simulation source commit must exactly equal the clean commit
used by the new run. Any mismatch is a validation failure.

## 6. Regenerate figures and tables from fresh evidence

After extracting and validating a fresh evidence archive, render only into a
separate derived-output directory. Substitute the source commit and run tag
recorded in that archive's manifest:

```bash
export STRICT_VALIDATION=1
export AGRIBRAIN_GIT_COMMIT=<full-source-commit>
export RUN_TAG=<run-tag>
export BENCHMARK_SEEDS=42,1337,2024,7,99,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515
export FIGURE_SEED_ROOT=mvp/simulation/results/benchmark_seeds
export FIGURE_OUTPUT_DIR=/absolute/path/to/derived_figures
python mvp/simulation/regenerate_figures_from_cache.py
```

The renderer refuses missing identity fields, a partial seed panel, and any
attempt outside the canonical HPC publisher to overwrite
`mvp/simulation/results`. It also requires the executing checkout to be clean
outside the run-output tree and its HEAD to equal the simulation commit. It is
a same-code deterministic replay, not a way to apply changed figure code to
old results. Do not edit the preserved archive or its extracted verification
copy.

Each canonical figure is written as an 800-DPI PNG and a one-page vector PDF.
The publisher rejects a PNG below the declared resolution or physical-size
gate and rejects PDFs with Type 3/unembedded fonts, raster image objects, or no
vector drawing primitives. The figure provenance records the exact accessible
color/marker/line/hatch contract, renderer package versions, resolved font-file
path and SHA-256, input hashes, and final PNG/PDF hashes. Method colors therefore
never carry identity alone: line series also use markers and line patterns, and
bar series use hatches and grouping position.

The reported quantities must be described as simulation-derived:

- waste fraction per routing opportunity;
- modeled emissions indicator; and
- social-performance proxy.

The last quantity is not demographic equity, and none of the three is a field
measurement.

## 7. Run the aligned full treatment on Slurm

From a clean checkout on the cluster login node:

```bash
git status --short
AGRIBRAIN_PARTITION=<partition> bash hpc/hpc_run.sh
```

The orchestrator creates a commit- and time-scoped run tag, creates a fresh
environment and detached read-only source snapshot, submits a 20-task seed
array and five-task stress array, then runs the dependent publication job only
after both arrays succeed. Its immutable receipt is deliberately labelled
submission-only: it records the submitted DAG, source-snapshot mode, and
literal source-tree SHA-256 but does not claim scheduler completion. Every
seed and stress payload records its actual Slurm job, parent array, task index,
and the same source-tree digest. The publisher refuses to run unless its
`SLURM_JOB_ID` equals the publisher declared in that receipt. The final semantic
receipt, validated worker payloads, and `afterok` execution are the completion
evidence. Do not resubmit simply because a job is pending or running.

The publisher also reruns the locked internal rolling-origin forecast check
and manifest-binds `forecast_validation_summary.json` and
`forecast_validation_predictions.csv`. These files document model selection
on the constructed benchmark series; they are not external or field
validation.

The complete core, H3, and secondary-ablation evidence retains exactly 6,100
lossless episode archives, 4,500 adaptation ledgers, and 1,600 final-evaluation
ledgers. The
publisher captures post-job scheduler accounting for every seed and stress
worker before finalization. Failed worker resource receipts are retained and
inventoried separately; resumable episode archives remain hash-validated at
their canonical paths. Failed attempts do not enter scheduler-success counts.
The locked design is not an 800-episode design. Any scientific change to
scenarios, seeds, modes, stressors, adaptation episodes, evaluation episodes,
or stochastic semantics requires a fresh run.

Monitor using the job identifiers printed by the orchestrator:

```bash
squeue -j <seed_job>,<stress_job>,<publish_job> \
  -o "%.18i %.9P %.24j %.2t %.10M %.6D %R"
sacct -j <job_id> --format=JobID,State,ExitCode,NodeList
```

On successful completion, verify the remote archive checksum, transfer the
archive, and verify the local checksum before extraction. Preserve the remote
original and the first local copy unchanged.

## 8. Run the separate structural-sensitivity treatment on Slurm

Structural sensitivity is not part of the core results directory or core
archive. From the same clean committed checkout, choose an absolute shared
scratch directory that is visible to login, compute, and publisher nodes:

```bash
export AGRIBRAIN_PARTITION=<partition>
export AGRIBRAIN_SENSITIVITY_ROOT=/shared/scratch/$USER/agribrain-structural
bash hpc/hpc_sensitivity_run.sh
```

The orchestrator refuses a root inside the repository. It creates an external
`sensitivity_<commit>_<timestamp>` run directory and a matching run-scoped
virtual environment, generates and dynamically audits the immutable run plan,
then submits exactly 3,000 manifest tasks. By default those tasks are split
into three fail-closed arrays of at most 1,000 indices, chained with `afterok`,
with at most 50 simultaneous tasks per array. Cluster-specific caps can be set
before submission, without changing the treatment:

```bash
export AGRIBRAIN_SENSITIVITY_ARRAY_CHUNK_SIZE=500
export AGRIBRAIN_SENSITIVITY_MAX_CONCURRENT=25
```

The locked design is 100 seed-balanced Latin-hypercube points over 29 active
factors, including `slca_carbon_cap`. It retains 6,500 cells, executes 24,500
complete episodes, and simulates 7,056,000 decision steps. The bounded factor
ranges are a space-filling structural stress design, not probability
distributions, confidence intervals, or an additional inferential sample.

Each worker verifies the exact source commit, clean checkout, read-only source
tree digest, run tag, plan hashes, locked environment, task index, parent
Slurm array, and external output boundary before execution. Every one of the
3,000 task results records its actual job/array/local/logical indices. The
dependent publisher runs only after every array succeeds and must have the
exact `SLURM_JOB_ID` declared in the submission-only receipt. It
hash-checks all 3,000 outputs, regenerates the analysis from those literal
bytes, checks its clean fixed validator checkout both before and after archive
creation, records the environment and Slurm submission chain, and creates
these files inside the external run directory:

- `structural_sensitivity_analysis.json`
- `structural_sensitivity_summary.csv`
- `structural_sensitivity_summary.png`
- `structural_sensitivity_summary.pdf`
- `structural_sensitivity_publication_receipt.json`
- `slurm_simulation_accounting.json`
- `structural_sensitivity_artifact_manifest.json`
- `structural_sensitivity_evidence_<RUN_TAG>.tar.gz`
- `structural_sensitivity_archive_receipt.json`

Use the SHA-256 in the archive receipt for transfer verification. The manifest
and archive contain the 3,000 hash-bound task JSON files, all 24,500 lossless
episode archives, all 18,000 adaptation ledgers, all 6,500 final-evaluation
decision ledgers, 3,000 per-task episode manifests, successful worker runtime
receipts, completed task logs, post-job scheduler accounting, and the
deterministic structural table/figure artifacts and provenance receipt. Each
task endpoint binds its final ledger's run-relative path, literal SHA-256,
Merkle root, and 288-record count; final validation recomputes endpoint metrics
from those ledger records. Failed-attempt artifacts are retained in separate
`__attempts` paths and reported separately; they do not change the canonical
24,500/18,000/6,500 counts. Temporary files, interpreter caches, and the
in-progress publisher log are excluded. Never copy structural files into
`mvp/simulation/results`, and never use `--allow-dirty` or
`--skip-dynamic-audit` for publication evidence.

The structural publication receipt additionally records the 800-DPI PNG pixel
dimensions and DPI metadata plus the PDF page size, embedded TrueType font
programs, vector-operator check, and zero raster-image-object check. H1/H2/H3
values are unchanged; scenario grouping and distinct markers only improve the
legibility of the same 55 summary cells.

The structural design is exactly the prespecified 100-point, 3,000-task design,
not an 800-episode design. Regenerating the deterministic CSV/PNG/PDF from the
hash-valid analysis JSON does not rerun or alter statistics. Conversely, any
change to the factor box, points, seeds, scenarios, stressors, modes, episode
schedule, or simulation logic is a scientific-design change and requires a new
structural simulation run.

## 9. Canonical publication controls

The canonical treatment requires the values declared in
`hpc/publication_env.sh`. Important controls include:

- `STRICT_VALIDATION=1`
- `DETERMINISTIC_MODE=false`
- `MCP_RATE_LIMITS=disabled`
- `DYNAMIC_KB_FEEDBACK=false`
- `PYTHONHASHSEED=0`
- one thread for OpenMP, MKL, OpenBLAS, NumExpr, and vecLib

`MCP_RATE_LIMITS=disabled` removes wall-clock token buckets from the scientific
treatment; deployment rate limits are an operational control, not an
experimental factor.

## 10. Interpretation limits

The benchmark uses a constructed spinach telemetry trace and synthetic
disruptions. It evaluates the declared coordination mechanism under that
design. It does not establish field effectiveness, lifecycle emissions,
empirical waste, demographic equity, or causal transparency of the upstream
forecaster, spoilage estimator, or retrieval process. The available
context-to-policy trace covers the mapping from retained context inputs to the
routing calculation.
