# Release procedure

## Preflight

1. Work in a separate clone or worktree.
2. Confirm `git status --short` is empty.
3. Run the focused publication and metadata tests.
4. Scan tracked files for credentials, local absolute paths, draft
   manuscripts, private logs, and unintended attribution text.
5. Confirm the backend, frontend, and `CITATION.cff` versions agree.

## Publication evidence

This methodology-aligned source requires a fresh simulation treatment; the
historical `2fd7bff` archive cannot be reused. Upload only the new validated
evidence archive and publish its exact SHA-256 checksum, member count, byte
size, source commit, protocol hash, environment receipt, and run tag.

The shippable archive builder does not accept publication-only or
dual-provenance repairs. Any post-run change that affects generated tables,
figures, validation, or packaging requires the complete evidence pipeline to
be rerun from one clean commit. Historical repair receipts remain audit-trail
records only and cannot produce a READY release bundle.

Both Slurm workflows execute from a detached read-only snapshot and bind its
literal source-tree SHA-256 into submission receipts and worker payloads. Those
receipts are explicitly submission-only and never stand in for scheduler
completion. Core seed/scenario envelopes and all 3,000 structural task results
must map to the exact array parent and task index in their receipt; each
publisher must run under the exact declared `SLURM_JOB_ID`. Structural
finalization additionally requires one clean, commit-exact fixed validator
checkout before and after archive creation. Post-job scheduler accounting must
cover every declared simulation worker before either evidence set is accepted.

The complete core, H3, and secondary-ablation evidence must retain exactly
6,100 lossless episode archives, 4,500 adaptation ledgers, and 1,600 final-
evaluation ledgers. The separate structural evidence must retain exactly
24,500 episode archives,
18,000 adaptation ledgers, and 6,500 final-evaluation ledgers. Failed-attempt
artifacts are retained and inventoried separately for diagnosis and audit; they
must not inflate those canonical counts or successful scheduler accounting.
The structural release also includes the deterministic
`structural_sensitivity_summary.csv`, `.png`, and `.pdf` plus the self-hashed
`structural_sensitivity_publication_receipt.json` that binds them to the saved
analysis.

Neither treatment is an 800-episode design. A documentation edit or downstream
table/figure export cannot authorize a changed scientific design. Any change to
seeds, scenarios, modes, stressors, structural points or bounds, episode
schedule, stochastic semantics, or simulation logic requires a fresh affected
simulation run and a newly validated evidence archive.

## Version and tag

The committed Python and npm locks are evidence inputs. CI installs and checks
those locks; it does not compare them with newly resolved upstream versions.
Upgrade dependencies only in a dedicated reviewed change, regenerate the
affected lock, run the complete tests, and rerun any affected simulation
treatment before attaching publication evidence.

Update these together only when cutting a new software version:

- `agribrain/backend/pyproject.toml`
- `agribrain/frontend/package.json`
- `CITATION.cff`

Then run:

```bash
python -m pytest agribrain/backend/tests/test_metadata_consistency.py -q
TAG=v$(python -c "import tomllib; print(tomllib.load(open('agribrain/backend/pyproject.toml','rb'))['project']['version'])")
git tag -a "$TAG" -m "AGRI-BRAIN $TAG"
```

Push only after reviewing the exact commit, tag, release-asset checksum, and
public file inventory. Never commit environment files, keys, credentials,
cluster logs, private provenance, or manuscript comparison files.
