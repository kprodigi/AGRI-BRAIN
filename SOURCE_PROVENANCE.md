# Source provenance and evidence status

This public repository was created as a history-free snapshot so superseded
files and a well-known local-development chain key that existed in earlier Git
history cannot be pushed or trigger secret scanners. The scientific source is
still byte-identifiable.

## Source identity

| Item | Value |
|---|---|
| Validated HPC source commit | `d3286aef28803c715045176008fae6b9c7e3367b` |
| Validated HPC source tree | `cef1e66f0b3cadeaf54f7189b080f26810d8212c` |
| Public history-free root commit | `7bcd883ac88b7c3fabf1dee9cdd79db1f69420f2` |
| Public source tag | `simulation-source-d3286ae` |
| Tree at the public source tag | `cef1e66f0b3cadeaf54f7189b080f26810d8212c` |

The identical tree hashes prove that the tagged public snapshot contains the
same tracked bytes as the source submitted to HPC. The current `main` branch
adds documentation, GitHub-community metadata, and non-scientific package
hygiene; the scientific simulation implementation is unchanged.

Verify locally:

```bash
git rev-parse 'simulation-source-d3286ae^{tree}'
git diff --name-only simulation-source-d3286ae..main -- agribrain mvp hpc \
  | grep -Ev '\.md$|package(-lock)?\.json$|pyproject\.toml$|pnpm-(lock|workspace)\.yaml$'
```

The first command must print
`cef1e66f0b3cadeaf54f7189b080f26810d8212c`; the second must print nothing.
The excluded package files contain metadata and package-manager cleanup only;
the committed dependency versions used by HPC are unchanged.

## Issued source-package checksums

The methodology-aligned source was also issued as:

| Package | SHA-256 |
|---|---|
| Git bundle `AGRI-BRAIN_HPC_ready_d3286ae_20260829_final.bundle` | `8a0e90ac446cace291ec6addf54cbf4bfcf5d3ded28909787562855541acf224` |
| Source ZIP `AGRI-BRAIN_HPC_ready_d3286ae_20260829.zip` | `aaab0daf5c2e2c4ae51bd90048ee47e4f94252f9c2e3d2c6967d7f85a3f85665` |

These checksums identify the issued inputs; the package files are not tracked
inside this repository.

## Current result status

No methodology-aligned confirmatory benchmark-effect, H1-H3, or structural-
sensitivity result is committed here. The historical `2fd7bff` evidence
predates the aligned simulation semantics and is retained under `provenance/`
only as a superseded audit record. It must not be used to populate tables,
figures, claims, or paper results for this source.

A result-bearing release is valid only after the fresh core, H3, secondary,
and structural treatments complete and the raw-input, inference, environment,
ledger, figure, manifest, and archive validators all pass. Its release record
must state the exact simulation source commit and tree, run tag, archive size,
member count, and SHA-256.
