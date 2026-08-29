# Safe GitHub publication procedure

**Release gate:** this methodology-aligned tree is a source-only candidate.
It may be published on a clearly labelled review branch so the exact source
can be executed on HPC. Do not tag it as validated publication evidence or
attach the historical `2fd7bff` results. Result-bearing release notes and
manuscripts require a fresh, validator-approved treatment from this source.

## 1. Start from the history-free source archive

Use the SHA-256-verified source ZIP issued with this candidate. Do **not** push
the packaging repository, an old Git bundle, or an earlier release directory:
their history is not part of the public source deliverable and can retain
superseded files or scanner-triggering localhost test material.

Extract the ZIP into a new empty directory, enter its single project root, and
inspect the exact inventory before creating Git history:

```bash
sha256sum <methodology-aligned-source.zip>
unzip -q <methodology-aligned-source.zip> -d <new-empty-directory>
cd <new-empty-directory>/AGRI-BRAIN_public_release
find . -type f -print | sort
```

Confirm that `mvp/simulation/results` contains only `README.md` and that no
credentials, environment files, local paths, generated dependencies, or
historical result artifacts are present.

## 2. Create one clean public-source commit

Use the human maintainer's own Git identity:

```bash
git init -b main
git config user.name "<maintainer name>"
git config user.email "<maintainer email>"
git add -A
git status --short
git commit -m "Publish AGRI-BRAIN methodology-aligned source candidate"
```

Run the focused source checks before any push:

```bash
python -m pytest \
  mvp/simulation/tests/test_publication_repair.py \
  mvp/simulation/tests/test_publication_evidence_scope.py \
  agribrain/backend/tests/test_metadata_consistency.py -q
git status --short
```

## 3. Add the remote and preserve its current main branch

```bash
git remote add origin https://github.com/kprodigi/AGRI-BRAIN.git
git fetch origin main
REMOTE_MAIN=$(git rev-parse origin/main)
printf '%s\n' "$REMOTE_MAIN"
git push origin "$REMOTE_MAIN":refs/heads/archive/pre-methodology-alignment-<date>
```

Verify the archive branch exists before continuing. Keep `REMOTE_MAIN` in the
same shell for the guarded lease below.

## 4. Push and inspect a review branch

```bash
git push origin HEAD:refs/heads/methodology-aligned-source-review-<date>
```

Inspect that branch on GitHub, including its one-commit history, file
inventory, rendered README, security scan, and continuous-integration result.
Do not replace `main` until the review branch is approved.

## 5. Replace main only after explicit approval

The history-free source commit is intentionally unrelated to the old remote
history. `--force-with-lease` stops if remote `main` changed after step 3:

```bash
git push \
  --force-with-lease=refs/heads/main:"$REMOTE_MAIN" \
  origin HEAD:refs/heads/main
```

If the lease fails, stop, fetch again, and review the intervening commits. Do
not retry with an unguarded force push.

## 6. Tag and attach only fresh validated evidence

After the locked HPC treatment and every validator have passed, replace the
placeholders below with the new run identifiers:

```bash
git tag -a submission-evidence-<fresh-run-tag> \
  -m "AGRI-BRAIN validated submission evidence"
git push origin submission-evidence-<fresh-run-tag>
sha256sum <fresh-evidence-archive>
gh release create submission-evidence-<fresh-run-tag> \
  <fresh-evidence-archive> \
  --title "AGRI-BRAIN validated submission evidence" \
  --notes-file RELEASE_NOTES.md
```

Confirm the uploaded asset name, size, and checksum after downloading it from
the release page. Never attach the superseded `2fd7bff` archive as evidence
for this source.
