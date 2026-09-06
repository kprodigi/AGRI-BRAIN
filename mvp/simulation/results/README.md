# Publication evidence

The canonical tables, statistics and receipts of the certified run are
committed here, so the values cited in the paper can be checked against a
clone without downloading anything.

| | |
|---|---|
| run tag | `d3286ae_20260829_105800` |
| simulation source commit | `d3286aef28803c715045176008fae6b9c7e3367b` |
| publication code commit | `675bdb2d43efd2ef46b6db78df337dbb5892d059` |
| manifest | `artifact_manifest.json`, schema 2, 1,684 artifacts |
| validation | `publication_validation_receipt.json` |

The reported results were re-aggregated from the preserved per-seed payloads of
the original run rather than regenerated, which is why the manifest is
dual-provenance: `recovery_authorization` binds the preserved payloads to the
original submission receipt, and the three files it names are committed
alongside it.

## What is here, and what is not

Committed: the benchmark summary and significance tables, the paper benchmark
table, the ablation and channel analyses, the stress and H3 tables including
the 25-cell grid, the forecast validation, the explainability metrics, and the
manifest and receipts that identify all of it.

Not committed, by design:

- **The per-seed envelopes and per-decision ledgers.** 1,600 ledger files, far
  past what belongs in a source repository. They are in the evidence deposit.
- **The rendered figures.** The manuscript prints a presentation-only re-render
  of these same certified numbers, and the renderer in this tree reproduces
  that set. Committing the publication run's own images would ship figures this
  code no longer produces. They remain manifest artifacts and are in the
  deposit.

Both are recorded in `artifact_manifest.json` with their hashes, so verifying
with `--allow-missing` reports them as warnings rather than failures.

## Verifying what is committed

```bash
python mvp/simulation/analysis/verify_manifest.py --strict-commit \
  --allow-missing --require-tracked \
  --recovery-receipt mvp/simulation/results/publication_recovery_receipts/d3286ae_20260829_105800.json
```

```bash
STRICT_VALIDATION=1 python mvp/simulation/validation/validate_results.py
```

The first hashes every committed artifact against the manifest and checks the
recovery authorization; the second validates the tables themselves. Both run in
CI on every push.

## Reproducing the figures

```bash
python mvp/simulation/regenerate_figures_from_cache.py
```

This needs the per-seed envelopes from the deposit. Point it at an extracted
copy; keep the downloaded archive unchanged and work from a separate copy.
