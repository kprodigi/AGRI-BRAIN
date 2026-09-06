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
same tracked bytes as the source submitted to HPC. That tag—not the later
`main` tree—is the authoritative simulation identity. The current `main`
branch also contains deterministic publication recovery, Slurm-accounting,
manifest, validation, and archive code added after the simulation workers
finished. Those changes do not alter or rerun the preserved simulation
algorithms or raw outputs, but they are substantive publication-code changes
and are recorded as a separate publication commit in recovered evidence.

Verify locally:

```bash
git rev-parse 'simulation-source-d3286ae^{tree}'
git diff --name-only simulation-source-d3286ae..main
```

The first command must print
`cef1e66f0b3cadeaf54f7189b080f26810d8212c`. The second command is now expected
to list the reviewed deterministic recovery/publication changes; it is an audit
inventory, not a no-difference assertion. Recovery receipts preserve both the
simulation commit/tree and the later clean publication commit/tree.

## Issued source-package checksums

The methodology-aligned source was also issued as:

| Package | SHA-256 |
|---|---|
| Git bundle `AGRI-BRAIN_HPC_ready_d3286ae_20260829_final.bundle` | `8a0e90ac446cace291ec6addf54cbf4bfcf5d3ded28909787562855541acf224` |
| Source ZIP `AGRI-BRAIN_HPC_ready_d3286ae_20260829.zip` | `aaab0daf5c2e2c4ae51bd90048ee47e4f94252f9c2e3d2c6967d7f85a3f85665` |

These checksums identify the issued inputs; the package files are not tracked
inside this repository.

## Current result status

The methodology-aligned core and structural simulation workers completed and
their original publisher jobs failed. The authorized publication-only
replacement publishers have since completed: jobs 14482387, 14482388 and
14482389 finished on 2026-09-04, and the combined full-submission validation
reports `PASS`. The publication validation receipt for run tag
`d3286ae_20260829_105800`, at publication code commit
`675bdb2d43efd2ef46b6db78df337dbb5892d059`, binds 1,683 semantic artifacts
under the Merkle root
`e9ad0fc3c536de873de4bd8d419170423ba0652c614e2b50803ebfd9bc254293`. The
simulation was not rerun; the recovery re-aggregated the preserved raw payloads
of the same run.

Those certified artifacts are not committed to this repository, which tracks
source rather than results. The historical `2fd7bff` evidence predates the
aligned simulation semantics and is retained under `provenance/` only as a
superseded audit record. It must not be used to populate tables, figures,
claims, or paper results for this source.

A result-bearing release is valid only after both authorized publication-only
recovery publishers complete and the raw-input, inference, environment,
ledger, figure, manifest, archive, and combined full-submission validators all
pass. Its release record must state the exact simulation source commit/tree,
publication code commit/tree, `simulation_rerun: false`, run tags, archive
sizes, member counts, and SHA-256 values. See
[docs/PUBLICATION_RECOVERY.md](docs/PUBLICATION_RECOVERY.md).
