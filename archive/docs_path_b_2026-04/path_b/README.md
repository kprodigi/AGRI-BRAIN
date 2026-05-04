# Path B (HISTORICAL — REVERTED)

> ⚠️ **This entire directory describes a feature branch that was
> implemented (commit `330ff67`) and then reverted in subsequent
> commits.** Do **not** apply any of the tracked-change instructions in
> `manuscript_updates.md`, `implementation_log.md`,
> `final_pre_hpc_check_2026-04-22.md`, `pre_hpc_blockers.md`,
> `pre_hpc_check_report.md`, `deep_review_2026-04-22.md`, or
> `sanity_report.md` to the manuscript. They reference a 6D ψ vector,
> a 3×6 `THETA_CONTEXT`, a `no_yield` ablation mode, and a
> `yield_query` MCP tool that no longer exist in the head of `main`.

## What Path B was

A prototype that added a sixth context-feature dimension (ψ₅ = supply
uncertainty) and a corresponding sixth column in `THETA_CONTEXT`,
along with a `no_yield` ablation mode and a `yield_query` MCP tool.

## Why it was reverted

The supply-uncertainty signal turned out to overlap with what the
state vector φ already exposes at indices 6–8 (supply point, supply
uncertainty, demand uncertainty). The reverted design folds the
supply forecast back into φ and keeps ψ at its original 5D
institutional-context shape, which is what every file in `head` now
documents (see `agri-brain-mvp-1.0.0/backend/pirag/context_to_logits.py`
for the canonical 5D definition).

## What to read instead

- `docs/STATISTICAL_METHODS.md` — current statistical methodology.
- `docs/METHODS_REPRO_APPENDIX.md` — current reproduction recipe.
- `docs/CLAIMS_TO_EVIDENCE.md` — current claim-artefact map.
- `docs/CALIBRATION_NOTES.md` — current calibration notes for hand-
  picked parameters.
- `docs/KNOWN_LIMITATIONS.md` — current open-issues list.
- The `# REVIEWER NOTE` blocks throughout `agri-brain-mvp-1.0.0/backend/`
  and `mvp/simulation/` — current per-parameter provenance.

## Why we keep this directory

The Path B docs are retained as a Git-trail of the prototype work and
the reasons for reverting it. They are not an authoritative source on
the system as of `main`.
