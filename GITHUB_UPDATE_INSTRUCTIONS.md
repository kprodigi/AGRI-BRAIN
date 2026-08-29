# Publish this prepared repository to GitHub

This folder is already a history-free Git repository on branch `main`. It has
no configured remote and contains two intended layers:

1. `simulation-source-d3286ae` tags the byte-identical scientific source sent
   to HPC; and
2. `main` adds documentation, GitHub community metadata, and non-scientific
   packaging hygiene without changing simulation behavior.

Do not initialize it again and do not copy in an older `.git` directory.

## 1. Verify the prepared repository

From this folder, run:

```bash
git status --short --branch
git log --oneline --decorate --max-count=5
git remote -v
git rev-parse 'simulation-source-d3286ae^{tree}'
git diff --name-only simulation-source-d3286ae..main -- agribrain mvp hpc \
  | grep -Ev '\.md$|package(-lock)?\.json$|pyproject\.toml$|pnpm-(lock|workspace)\.yaml$'
```

Expected conditions:

- branch `main` is clean;
- no remote is configured;
- the tag tree is `cef1e66f0b3cadeaf54f7189b080f26810d8212c`;
- the filtered diff command prints nothing; and
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
the fresh methodology-aligned HPC publishers and all validators to complete.
Then record the exact run identifiers and archive checksum in the release
notes and follow [docs/RELEASE.md](docs/RELEASE.md).
