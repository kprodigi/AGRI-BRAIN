# Institutional retrieval knowledge base

This directory contains the constructed 20-document corpus used by the
publication benchmark. The documents encode synthetic operating procedures,
contingency guidance, waste-hierarchy assumptions, carbon-accounting
assumptions, and social-performance assumptions for the controlled case.

The corpus is untrusted benchmark input. It is not a collection of verified
current regulations, legal or operational advice, food/feed-safety evidence,
certification, measured social performance, or field validation. Text inside a
retrieved document never overrides executable policy, outcome, evidence, or
protocol contracts. Retrieval can alter a context-enabled policy only through
the guarded context-to-logit path; the retained calculation trace records that
path but does not prove that a retrieved statement is true.

Every `.txt` document must begin with a self-contained source-scope disclaimer
because the complete document can be returned as one retrieval passage. The
stable retrieval identifier is the filename stem. `README.md` is not ingested.
There must be exactly 20 `.txt` files; renaming a file changes its retrieval ID.

Current source scope is deliberately limited to the executable synthetic case:
three canonical actions, modelled transport emissions, author-declared social
priors, mechanistic spoilage risk, scenario perturbations, coordinator-mediated
peer context, and optional external ledger anchoring. The corpus must not invent
audits, logs, geographic overlays, measurements, legal duties, safety decisions,
deployment capabilities, or unimplemented workflows.

When extending the corpus:

1. record a stable document identifier and source note;
2. distinguish constructed assumptions from externally verified guidance;
3. rerun ingestion and retrieval-quality tests;
4. run the benchmark as a new treatment rather than mixing the new corpus with
   the preserved publication evidence; and
5. record the new commit and tracked-source-tree digest in the environment and
   artifact manifests; retained decisions record SHA-256 hashes of the exact
   retrieved passages.

Any result generated from a previous source-tree digest is historical and
cannot be presented as evidence for this revised corpus.
