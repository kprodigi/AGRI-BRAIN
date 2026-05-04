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

## 4. Create the GitHub release (Zenodo trigger)

If this is the first release on this account, do the one-time Zenodo
setup at <https://zenodo.org/account/settings/github/> first; flip the
AGRI-BRAIN repo's switch to "On".

Then on GitHub:

1. Open <https://github.com/kprodigi/AGRI-BRAIN/releases/new>
2. Pick the tag created in step 3.
3. Title: `AGRI-BRAIN <TAG>` (e.g. `AGRI-BRAIN v1.2.0`)
4. Paste the relevant section of the changelog into the body.
5. Publish.

Zenodo creates a DOI of the form `10.5281/zenodo.<NNNNNNN>`.

## 5. Record the DOI

Replace the empty placeholder in `CITATION.cff`:

```yaml
doi: "10.5281/zenodo.<NNNNNNN>"
```

Commit on `main` with message `Record Zenodo DOI for v1.2.0` and push.
The `tests/test_metadata_consistency.py::test_citation_doi_format_when_set`
guard now activates and validates the format.

## 6. Update the artifact manifest

After the next HPC run, the artifact manifest under
`mvp/simulation/results/artifact_manifest.json` will pick up the new
git commit SHA and the lockfile contents (Stage `build_artifact_manifest.py`
hashes the lockfile when present). Confirm:

```bash
python -c "import json; m=json.load(open('mvp/simulation/results/artifact_manifest.json')); \
           assert m.get('git_commit'), 'manifest missing git_commit'; \
           print('manifest commit:', m['git_commit'])"
```

## Rollback

If the tag is wrong, delete it locally and remotely (within minutes
of pushing; Zenodo has not minted yet):

```bash
git tag -d v1.2.0
git push origin :refs/tags/v1.2.0
```

If Zenodo has already minted a DOI, do **not** rewrite history -- mint a
new patch tag (`v1.2.1`) instead and let the broken DOI become the
"superseded" version on Zenodo.
