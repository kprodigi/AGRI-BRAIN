# Release procedure

This file is the single source of truth for cutting a tagged release of
AGRI-BRAIN. Follow the steps in order; each step has an automated guard
that catches the common failure mode.

## 0. Pre-flight

Ensure the working tree is clean and CI is green on `main`:

```bash
git status                      # must be empty
git fetch origin && git rev-parse HEAD == git rev-parse origin/main
gh run list --branch main --limit 1   # must show success on the latest run
```

## 1. Bump the version

The version lives in two files. Both must be updated together; the
`tests/test_metadata_consistency.py` guard fails CI if they drift.

* `agribrain/backend/pyproject.toml` -> `[project] version`
* `CITATION.cff` -> `version` and `date-released`

After editing, verify locally:

```bash
pytest agribrain/backend/tests/test_metadata_consistency.py -v
```

## 2. Regenerate the dependency lockfile (production reproducibility)

The lockfile pins every transitive dependency at a known-good
combination so external reviewers reproduce the exact numerical
results the paper reports. Regenerate from a clean Python 3.11 venv
into the canonical path:

```bash
python3.11 -m venv .venv-lock
source .venv-lock/bin/activate
python -m pip install --upgrade pip
pip install -e "agribrain/backend[dev]"
pip freeze --exclude-editable > agribrain/backend/requirements-lock.txt
deactivate
rm -rf .venv-lock
```

Commit the regenerated lockfile in the same commit as the version bump.

## 3. Tag and push

```bash
TAG=v$(grep -E '^\s*version\s*=' agribrain/backend/pyproject.toml | head -1 | sed -E "s/.*['\"]([^'\"]+)['\"].*/\\1/")
git tag -a "$TAG" -m "AGRI-BRAIN $TAG"
git push origin main
git push origin "$TAG"
```

## 4. Create the GitHub release

On GitHub:

1. Open <https://github.com/kprodigi/AGRI-BRAIN/releases/new>
2. Pick the tag created in step 3.
3. Title: `AGRI-BRAIN <TAG>` (e.g. `AGRI-BRAIN v1.2.0`)
4. Paste the relevant section of the changelog into the body.
5. Publish.

## 5. Update the artifact manifest

After the next HPC run, the artifact manifest under
`mvp/simulation/results/artifact_manifest.json` will pick up the new
git commit SHA and the lockfile contents (Stage `build_artifact_manifest.py`
hashes the lockfile when present). Confirm:

```bash
python -c "import json; m=json.load(open('mvp/simulation/results/artifact_manifest.json')); \
           assert m.get('git_commit'), 'manifest missing git_commit'; \
           print('manifest commit:', m['git_commit'])"
```

## Citation policy

This repository deliberately does **not** carry a DOI in
`CITATION.cff` or in the README BibTeX block. Cite via the
``version`` field in `CITATION.cff` plus the ``git_commit`` recorded
in `mvp/simulation/results/artifact_manifest.json` plus the
repository URL. The
``tests/test_metadata_consistency.py::test_citation_omits_doi_field``
guard rejects re-introduction of a top-level ``doi:`` key, and
``test_readme_omits_doi_in_bibtex`` rejects a ``doi`` field in the
README citation block. If a future release does need a DOI, prefer
the CFF 1.2 ``identifiers:`` block over the legacy top-level
``doi:`` key.

## Branch protection (one-time setup)

Install the GitHub branch-protection rules that gate `main` on the
required CI checks:

```bash
GH_TOKEN="<PAT with repo admin>" bash hpc/set_branch_protection.sh
```

The PAT needs `Administration: Read and write` on this repository
(fine-grained) or `repo` scope (classic). After install, every push
to `main` requires the following status checks to pass before merge:
`artifact-validation`, `slow-tests (ubuntu / py3.11)`,
`backend-tests` on Ubuntu / Windows / macOS Python 3.11,
`python-lint (ruff)`, `contract-tests`, `contract-analysis`,
`frontend-build`. The script also disallows force-pushes and
deletions, and requires linear history. This is a one-time action
unless the CI workflow's status-check names change.

## Post-HPC commit (refresh regression baseline)

The `mvp/simulation/baseline_snapshot.json` regression baseline
captures the deterministic-mode digest of `table1_summary.csv` and
`table2_ablation.csv`. The `artifact-validation` CI job hard-fails
on `main` when the snapshot disagrees with the committed tables.
After every HPC run that lands new tables, refresh the snapshot:

```bash
git pull origin main      # pull HPC's table1/table2 update
DETERMINISTIC_MODE=true REGRESSION_GUARD_INIT=true \
    python mvp/simulation/validation/run_regression_guard.py
git add mvp/simulation/baseline_snapshot.json
git commit -m "Refresh regression baseline after HPC run <RUN_TAG>"
git push origin main
```

The same applies to `mvp/simulation/nightly_baseline.json` (the
nightly-smoke-pipeline baseline) -- copy the new
`benchmark_summary.json` to `nightly_baseline.json` after every HPC
run that lands new stochastic-mode summaries:

```bash
cp mvp/simulation/results/benchmark_summary.json \
   mvp/simulation/nightly_baseline.json
git add mvp/simulation/nightly_baseline.json
git commit -m "Refresh nightly-smoke baseline after HPC run <RUN_TAG>"
git push origin main
```

## Rollback

If the tag is wrong, delete it locally and remotely:

```bash
git tag -d v1.2.0
git push origin :refs/tags/v1.2.0
```

Then re-cut the tag at the corrected commit and push.
