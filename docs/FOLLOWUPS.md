# Tracked follow-ups

Items deliberately deferred to a future cleanup pass, with the
trigger that should reactivate them. Inline `# TODO` comments in
code link here; CI does not gate on these (they are real future
work, not known bugs).

## After the next 20-seed HPC publication run

* **Tighten `test_metric_variants.py::test_save_factor_average_matches_mode_eff_agribrain`
  tolerance from ±0.20 to ±0.10.** Reference:
  `agribrain/backend/tests/test_metric_variants.py:731` and
  `docs/MODE_EFF_EMPIRICAL.md`. The current 0.20 window
  accommodates calibration drift between the post-2026-04 simulator
  (new physics + Levers 1+2 + MODE_CARBON_EFF) and the on-disk
  `benchmark_summary.json` produced by the pre-fix simulator. Once
  `hpc/hpc_run.sh` regenerates the bench file with the new physics,
  the empirical save mean is expected to rise from ~0.68 to ~0.80
  (close to `MODE_EFF["agribrain"] = 0.83`).

* **Regenerate `mvp/simulation/baseline_snapshot.json`** from the
  new-physics deterministic run (`docs/RELEASE.md` step 6 + 7
  document the procedure under "Regenerate the regression-guard
  snapshot"). Until the snapshot is regenerated, CI runs with
  `ALLOW_MISSING_BASELINE=1` so the regression guard is advisory.

## Deferred until 1.3.0

* **Rename `src/` → `agribrain_backend/`** (Reviewer-2 audit M-12).
  Generic top-level package name `src` collides with other research
  projects in the same venv. Mechanical rename + import update +
  CI/test matrix verification. Coordinated with a major-minor bump
  so users have a single migration window.

* **Tighten ruff to `--select` including `I` (import order) and
  `B` (bug-prone patterns)**. Currently CI gates only on the
  structural rule subset (`E9,F63,F7,F82,F401,F821`). Enabling `I`
  needs a single auto-fix pass + a test snapshot bump for files
  that have shipped imports out of canonical order.

* **Frontend page-level Playwright smoke test** for the
  `OpsPage` -> `take decision` -> `expand explainability` flow. The
  current `ExplainabilityPanel` test pins the rendering contract
  in isolation; an end-to-end test would also catch routing and
  data-flow regressions that page-internal mocks miss.

## Reactivation rule

When closing a follow-up: delete its bullet here, remove the
referencing inline TODO from code, and adjust any tests/docs that
called the item out as known-deferred. The CI guard
`tests/test_followups_referenced.py` (when present) enforces that
inline `# FOLLOWUP:` comments link to a still-listed bullet.
