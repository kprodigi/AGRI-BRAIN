# Archived Documentation

Historical working notes kept under version control for traceability
of past code changes, but **not part of the public repository
surface** — the current canonical documentation lives at the
repository root and under `docs/` (`CLAIMS_TO_EVIDENCE.md`,
`METHODS_REPRO_APPENDIX.md`, `STATISTICAL_METHODS.md`,
`PRE_2025_04_FAIRNESS_FIX_DELTA.md`).

## Contents

### `path_b/`
Internal coordination notes for the supply/demand forecasting work
stream that landed in commit 330ff67 (yield_query + psi_5). The
"path B" label was internal jargon for the work parallel to the
core reverse-logistics simulator; the production code uses the
neutral name **"yield/demand forecasting"** throughout. See
`backend/pirag/mcp/tools/yield_query.py` and `demand_query.py` for
the canonical entry points.

## Status

These files are **frozen as historical records**. Do not update
them — the live truth is the working code, the live tests, and
the canonical docs at the repository root.
