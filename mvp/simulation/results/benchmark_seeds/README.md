# Per-seed benchmark envelopes

Each `seed_*.json` is the scalar metric envelope for one of the 20
benchmark seeds (~2.7 MB each). They are committed so a fresh clone can
re-run the BCa bootstrap behind `benchmark_summary.json` without
re-running the 50-hour HPC benchmark.

## Layout

- **Top-level `seed_*.json`** — the canonical set consumed by
  `benchmarks/aggregate_seeds.py`. Byte-identical to the
  `5ad0256_20260605_2203/` snapshot below.
- **`<RUN_TAG>/` subdirectories** — frozen snapshots of earlier HPC
  runs, named `<git-commit>_<YYYYMMDD_HHMM>`, retained for provenance:
  - `5ad0256_20260605_2203/` — **canonical run** (source of the
    published numbers; identical to the top-level set).
  - `dab51b1_20260512_1148/`, `d33b8de_20260507_1024/`,
    `485c769_20260505_0349/` — superseded runs kept for cross-run
    comparison.

All files are pinned by SHA-256 in
`mvp/simulation/results/artifact_manifest.json`.
