# docs/figures/

Source of truth for hand-authored figures that are not produced by
the simulation pipeline.

## fig1 — architecture overview

* `fig1_architecture.md` — Mermaid description of the AGRI-BRAIN
  architecture (sensors → spoilage/forecast models → MCP/piRAG context
  layer → policy → chain anchoring → frontend). This is the source.
* `fig1_architecture.svg` — rendered SVG (regenerate after edits).
* `fig1_architecture.pdf` — rendered PDF for paper inclusion.

## Regenerating fig1

The Mermaid description renders to SVG/PDF either:

* **Online**: open <https://mermaid.live/>, paste the Mermaid block from
  `fig1_architecture.md`, click "Actions" → "Save as SVG" / "Save as PDF".
* **CLI**: `npx -p @mermaid-js/mermaid-cli mmdc -i docs/figures/fig1_architecture.md -o docs/figures/fig1_architecture.svg`
  (extract just the fenced ```mermaid block first, or pass `--input` directly to a `.mmd` file).

After regenerating, commit both the source `.md` and the `.svg`/`.pdf`
together so the artifact-manifest hash changes for both in lockstep.

## Other simulator-produced figures

`fig2_*` through `fig10_*` are produced by
`mvp/simulation/generate_figures.py` and live under
`mvp/simulation/results/`. Do not place them here.
