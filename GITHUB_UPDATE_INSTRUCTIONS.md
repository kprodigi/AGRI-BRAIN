# Publish this prepared repository to GitHub

The prepared GitHub-ready folder is a history-free Git repository on branch
`main`. It has no configured remote and contains two intended layers:

1. `simulation-source-d3286ae` tags the byte-identical scientific source sent
   to HPC; and
2. `main` adds the audited deterministic publication-recovery, validation,
   provenance, documentation, and packaging layers. These additions do not
   alter or rerun the preserved simulation algorithms or raw outputs.

The companion ZIP is a source-only journal/sharing archive and intentionally
does not contain `.git` metadata. It is not the history- and tag-preserving
vehicle for a GitHub push. For GitHub, either use the prepared folder directly
or reconstruct it from the companion Git bundle:

```bash
git clone AGRI-BRAIN_GitHub_ready_<commit>_<date>.bundle AGRI-BRAIN
cd AGRI-BRAIN
git remote remove origin
```

The clone preserves `main` and `simulation-source-d3286ae`; removing the local
bundle remote restores the intended no-remote state. Do not initialize the
prepared folder again, initialize an extracted source-only ZIP as a substitute,
or copy in an older `.git` directory.

## 1. Verify the prepared repository

From this folder, run:

```bash
git status --short --branch
git log --oneline --decorate --max-count=5
git remote -v
git rev-parse 'simulation-source-d3286ae^{tree}'
git diff --name-status simulation-source-d3286ae..main
```

Expected conditions:

- branch `main` is clean;
- no remote is configured;
- the tag tree is `cef1e66f0b3cadeaf54f7189b080f26810d8212c`;
- the diff is the reviewed recovery/publication inventory described in
  `SOURCE_PROVENANCE.md`, rather than an assertion that `main` is identical to
  the simulation tag; and
- `mvp/simulation/results/` contains only `README.md`.

## 2. Create an empty GitHub repository

Create the repository without adding a GitHub-generated README, license, or
`.gitignore`; those files already exist here. Then connect and push:

```bash
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git remote -v
git push -u origin main
git push origin simulation-source-d3286ae
```

Replace `<OWNER>` and `<REPOSITORY>` with the actual destination. Review the
rendered README, Actions run, dependency alerts, file inventory, commit list,
and tag after the push.

The prepared badges, `CITATION.cff`, and backend package metadata currently
point to `https://github.com/kprodigi/AGRI-BRAIN`. If the destination differs,
update those public URLs together before committing and pushing.

## 3. If the destination already has work

Do not force-push immediately. Fetch it and publish this repository to a
review branch first:

```bash
git fetch origin
git push -u origin main:methodology-aligned-source-review-20260829
git push origin simulation-source-d3286ae
```

Archive or merge the existing work deliberately. Keep any migration procedure
outside the public repository. If replacement is ultimately necessary, use a
reviewed `--force-with-lease`, never an unguarded force push.

## 4. Publication-evidence gate

Pushing this source does not authorize numerical claims. Do not create a
result-bearing release or upload the historical `2fd7bff` evidence. Wait for
both authorized publication-only recovery publishers and the dependent
combined `READY` validation boundary to complete. This recovery must preserve
`simulation_rerun: false`. Then record the exact simulation and publication
identities, run identifiers, archive checksums, and validation status in the
release notes and follow [docs/RELEASE.md](docs/RELEASE.md).
