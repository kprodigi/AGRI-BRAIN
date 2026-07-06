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

The canonical version lives in `agribrain/backend/pyproject.toml`
(`[project] version`); the frontend `package.json` must track it. The
`tests/test_metadata_consistency.py` guard fails CI if they drift.

* `agribrain/backend/pyproject.toml` -> `[project] version` (canonical)
* `agribrain/frontend/package.json` -> `version`

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

2026-06: author identity is withheld from the repository — there is no
`CITATION.cff` and no README BibTeX block. Reference a specific state of
the work by the ``[project] version`` in `pyproject.toml` plus the
``git_commit`` recorded in
`mvp/simulation/results/artifact_manifest.json` plus the repository URL.
The ``test_readme_omits_doi_in_bibtex`` guard still rejects a ``doi``
field if a BibTeX block is ever re-added.

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

The nightly-validation drift gate is **optional**: its baseline
`mvp/simulation/nightly_baseline.json` is not committed by default, so
the gate stays dormant (it logs a skip notice) until you seed it. A
single-seed nightly run diverges from the 20-seed mean by more than the
workflow's 5% band, so if you commit a baseline, widen the tolerance in
`.github/workflows/nightly-validation.yml` first to avoid false
positives. To seed it, copy the new `benchmark_summary.json` after an
HPC run that lands new stochastic-mode summaries:

```bash
cp mvp/simulation/results/benchmark_summary.json \
   mvp/simulation/nightly_baseline.json
git add mvp/simulation/nightly_baseline.json
git commit -m "Refresh nightly-validation baseline after HPC run <RUN_TAG>"
git push origin main
```

## Rollback

If the tag is wrong, delete it locally and remotely:

```bash
git tag -d v1.2.0
git push origin :refs/tags/v1.2.0
```

Then re-cut the tag at the corrected commit and push.
