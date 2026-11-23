# Provenance

## The study run

The study ran from commit `78b631b8a07e750027a3611ec9222a389df71554` of this source on a clean checkout. [`provenance/revised_study.json`](../provenance/revised_study.json) records the identity of the run:

- run `study_20261114_164524_results`, plan sha256 `5c29cf6e627133f9d64ee7702ff18f754ae0a5898d4062ea45e8d24b463c6d90`;
- 6,300 tasks over 63 settings, 200 frozen controls and 59,700 decision ledgers, all verified, with no development results; the endpoints reconciled with the ledgers with a largest difference of 0;
- analysis archive `study_20261114_164524_results_analysis.tar.gz`, sha256 `451ffddf0a8cb8eba23e6d281ffaf44f97610d54886cf12b916baf3d950c7f4d`.

Each table in `data/study` is a subset of one file of that archive, limited to the rows and columns the paper shows; [`data/study/MANIFEST.json`](../data/study/MANIFEST.json) gives the source file's hash, which matches the hash recorded for it in the run's completion record. No value was recomputed or edited.

## Files changed for publication

The code here is the source the study ran from. Publication removed the submitted study's results and figures, added `data/study`, `data/figure_inputs`, the final figures and the figure scripts in `analysis/figures` with `analysis/make_figures.py`, and rewrote the documentation, the dashboard data export and `scripts/verify_package.py`. `provenance/revised_study.json` lists every file added, changed or removed against the run commit. No file under `agribrain/backend`, `mvp`, `hpc`, `study` or `reproduction` was changed.

The study protocol names the tags of the development history; the published history is a single commit of this source.

## Earlier source records

`provenance/original_source_manifest.json` and `provenance/source_distribution.json` record how the source of the submitted configuration (base commit `83c91ce8592c1d40234e5a6bb2a8ce342412f383`) was distributed and which files the revised study changed or added; `reproduction/SOURCE_MANIFEST.json` checks the copy in `reproduction/source`. The study's own changes are switched off in the submitted setting, which must route identically to the stored results of the submitted configuration in every seed–scenario cell.
