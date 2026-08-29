# Publication-only recovery after a failed Slurm publisher

This workflow is only for a run whose simulation workers completed but whose
declared publisher failed. It recomputes deterministic tables, figures,
validation receipts, and archives from byte-preserved raw outputs. It does
**not** submit or rerun any simulation episode.

The recovered evidence has intentionally separate provenance:

- simulation source: the immutable commit recorded by the original workers;
- publication source: a later clean commit containing deterministic
  publication/validation repairs;
- `simulation_rerun: false` in both run-scoped recovery authorizations.

Do not relabel recovery as a fresh single-commit run. Do not edit or re-sign an
original submission receipt. Do not run `hpc/hpc_publish.sh` or
`hpc/hpc_sensitivity_publish.sh` manually with a replacement job ID: their
normal paths correctly reject that identity mismatch.

## Current preserved run

The completed simulation identity is:

- simulation commit: `d3286aef28803c715045176008fae6b9c7e3367b`
- scratch root:
  `/scratch/tmp/agribrain_101198337_d3286ae_20260829_105500`
- core run tag: `d3286ae_20260829_105800`
- core seed array: `14473469`
- core stress array: `14473470`
- original failed core publisher: `14473471`
- structural run tag: `sensitivity_d3286ae_20260829_105855`
- structural arrays: `14473491`, `14473492`, `14473493`
- original failed structural publisher: `14473494`

The original publisher failures, their terminal Slurm accounting, and their
literal stdout/stderr hashes are bound by separate core and structural
recovery receipts. Rejected manual attempts `14476501`–`14476503` are not
treated as simulation or publication provenance and are not required to
authorize recovery.

## Required preserved inputs

Core inputs, under the original detached simulation snapshot:

```text
/scratch/tmp/agribrain_101198337_d3286ae_20260829_105500/
  source_d3286ae/.publication_sources/d3286ae_20260829_105800/
    mvp/simulation/results/
      benchmark_seeds/d3286ae_20260829_105800/
      stress_runs/d3286ae_20260829_105800/
      decision_ledger_h3/d3286ae_20260829_105800/
      core_submission_receipts/d3286ae_20260829_105800.json
    logs/publish_14473471.out
    logs/publish_14473471.err
```

Structural inputs:

```text
/scratch/tmp/agribrain_101198337_d3286ae_20260829_105500/
  structural_results_d3286ae/sensitivity_d3286ae_20260829_105855/
    tasks/
    runtime_receipts/
    logs/
    run_plan.json
    parameter_registry.json
    lhs_design.json
    lhs_design.csv
    task_manifest.json
    task_manifest.jsonl
    episode_accounting.json
    experiment_protocol.json
    slurm_submission.json
```

The structural failed-publisher logs are expected at
`logs/publish_14473494.out` and `logs/publish_14473494.err` in that structural
run directory.

## Exact launch command

First transfer/checkout the reviewed code, commit it, and run all local tests.
On the Slurm login node, enter that **clean publication-repair commit**. Choose
a new absolute control directory that does not already exist, then run:

```bash
ROOT=/scratch/tmp/agribrain_101198337_d3286ae_20260829_105500
CORE_SOURCE="$ROOT/source_d3286ae/.publication_sources/d3286ae_20260829_105800"
STRUCTURAL_RUN="$ROOT/structural_results_d3286ae/sensitivity_d3286ae_20260829_105855"

export AGRIBRAIN_PARTITION=compute
export AGRIBRAIN_RECOVERY_CONTROL_ROOT="$ROOT/publication_recovery_20260829_v1"
export AGRIBRAIN_CORE_RAW_SOURCE_SNAPSHOT="$CORE_SOURCE"
export AGRIBRAIN_STRUCTURAL_RUN_DIR="$STRUCTURAL_RUN"
export AGRIBRAIN_SIMULATION_COMMIT=d3286aef28803c715045176008fae6b9c7e3367b
export AGRIBRAIN_CORE_RUN_TAG=d3286ae_20260829_105800
export AGRIBRAIN_STRUCTURAL_RUN_TAG=sensitivity_d3286ae_20260829_105855
export AGRIBRAIN_CORE_FAILED_PUBLISHER_JOB_ID=14473471
export AGRIBRAIN_STRUCTURAL_FAILED_PUBLISHER_JOB_ID=14473494
export AGRIBRAIN_CORE_FAILED_STDOUT="$CORE_SOURCE/logs/publish_14473471.out"
export AGRIBRAIN_CORE_FAILED_STDERR="$CORE_SOURCE/logs/publish_14473471.err"
export AGRIBRAIN_STRUCTURAL_FAILED_STDOUT="$STRUCTURAL_RUN/logs/publish_14473494.out"
export AGRIBRAIN_STRUCTURAL_FAILED_STDERR="$STRUCTURAL_RUN/logs/publish_14473494.err"

bash hpc/publication_recovery_run.sh
```

If the cluster's Python executable is not `python3.11`, also export
`AGRIBRAIN_PYTHON_BIN` to the correct Python 3.11 executable.

The orchestrator fails closed and cancels any still-held recovery job if setup,
evidence, or receipt validation fails. Its order is:

1. capture self-hashed terminal accounting for jobs `14473471` and `14473494`;
2. hash the complete, exact core and structural raw-input inventories;
3. create three independent detached read-only worktrees of the identical clean
   publication commit (core, structural, and combined finalizer), eliminating
   cross-job dirty-tree/output races;
4. submit one core publisher, one structural publisher, and one combined-
   evidence finalizer with `sbatch --hold`; the finalizer has an exact
   `afterok:<core>:<structural>` dependency;
5. create separate receipts binding each new held job ID to its original
   receipt, failed job, logs, raw manifest, simulation commit, publication
   commit/tree, and `simulation_rerun: false`, plus a self-hashed finalizer
   authorization binding its exact job ID, both publisher IDs, held state, and
   exact `afterok` dependency;
6. validate both authorizations and revalidate that all three jobs remain
   user-held (including the finalizer's exact two-publisher dependency);
7. release all three held jobs together. Slurm runs the finalizer only after
   both publishers complete successfully.

If release fails or its state transition cannot be proved, the orchestrator
preserves the canonical authorization evidence, reports the observed states,
and requests cancellation of all three publication-only jobs. It never reports
an unverified cancellation or successful release.

The command prints all three Slurm job IDs and the final combined-evidence
destination. The finalizer defaults to 8 hours, 32 GiB, and 4 CPUs; those can
be overridden before launch with `AGRIBRAIN_FINALIZER_WALLTIME`,
`AGRIBRAIN_FINALIZER_MEMORY`, and `AGRIBRAIN_FINALIZER_CPUS`.

No seed, stress, or structural task array is submitted by this script.

## Live-input integrity

The core publisher validates the original bindings, copies them into its clean
worktree, rebinds the manifest to the staged bytes actually consumed, and
revalidates those staged bytes before aggregation and immediately before final
archive/receipt creation. Scheduler accounting is written outside the bound
raw seed directory.

The structural publisher writes live logs outside the preserved structural
run, validates the exact bound directories/files before analysis, and repeats
that validation immediately before and inside final archive creation.

Any changed, missing, extra, symlinked, or copy-corrupted raw file aborts
recovery.

## Outputs and certification boundary

After both replacement publishers and their dependent finalizer finish
successfully:

- core publication bundle:
  `$AGRIBRAIN_RECOVERY_CONTROL_ROOT/publication_source_core/publication_bundle_d3286ae_20260829_105800/`
- core lossless evidence:
  `$AGRIBRAIN_RECOVERY_CONTROL_ROOT/publication_source_core/mvp/simulation/results/complete_run_evidence/d3286ae_20260829_105800/`
- structural archive and receipt: in the structural run directory;
- combined, atomic submission evidence:
  `$AGRIBRAIN_RECOVERY_CONTROL_ROOT/full_submission_evidence/`, containing
  exactly `FULL_SUBMISSION_EVIDENCE_RECEIPT.json`, `READY.json`,
  `FINALIZER_SUBMISSION_AUTHORIZATION.json`, and
  `FINALIZER_PUBLICATION_ENVIRONMENT.json`;
- recovery authorizations, raw manifests, failed accounting, and replacement
  logs: under `$AGRIBRAIN_RECOVERY_CONTROL_ROOT` (with canonical copies in the
  corresponding evidence trees).

Do not describe results as fully recovered, validated, certified, or ready for
submission merely because the recovery code exists or the jobs were submitted.
That statement becomes supportable only after all three printed job IDs have
terminal `COMPLETED` status, the core publication and complete-run bundles
have their READY/receipt markers, the structural archive receipt exists, and
`full_submission_evidence/READY.json` validates. No separate manual combined
build is required.

After the three jobs finish, validate that exact promoted four-file bundle with
the clean publication checkout and the three job IDs printed by the launcher:

```bash
CORE_RECOVERY_JOB=<printed-core-publisher-job-id>
STRUCTURAL_RECOVERY_JOB=<printed-structural-publisher-job-id>
FINALIZER_RECOVERY_JOB=<printed-finalizer-job-id>
python hpc/validate_full_submission_ready.py validate \
  --directory "$AGRIBRAIN_RECOVERY_CONTROL_ROOT/full_submission_evidence" \
  --simulation-commit "$AGRIBRAIN_SIMULATION_COMMIT" \
  --publication-commit "$(git rev-parse HEAD)" \
  --finalizer-job-id "$FINALIZER_RECOVERY_JOB" \
  --core-job-id "$CORE_RECOVERY_JOB" \
  --structural-job-id "$STRUCTURAL_RECOVERY_JOB" \
  --run-tag "finalizer_${AGRIBRAIN_CORE_RUN_TAG}"
```

This command rejects extra or missing files, any lexical symlink component,
literal-byte or canonical self-hash changes, job substitution, provenance
substitution, and publication-environment identity changes.

The dependent finalizer activates its own finalizer-scoped venv in its clean,
independent finalizer worktree and
revalidates its runtime settings and exact locked Python environment before it
invokes `build_full_submission_evidence.py` with all eight required inputs:
the core archive/receipt/READY; the lossless
`complete_run_evidence_<core-tag>.tar.gz`/`RECEIPT.json`/`READY.json`; and the
structural archive/receipt. It revalidates both separate recovery
authorizations, exact preserved-raw mappings, shared simulation and
publication identities, scheduler accounting, semantic evidence, and archive
hashes before atomically promoting its receipt and self-hashed READY marker.
The promoted directory also contains the exact held-finalizer scheduler
authorization and the finalizer's version-resolved publication-environment
receipt; `READY.json` binds both files by literal SHA-256 and binds the
scheduler authorization by its canonical self-hash.

The old failed publisher jobs and the rejected manual attempts must never be
cancelled, modified, or deleted from the audit record.
