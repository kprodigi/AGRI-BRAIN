# piRAG Knowledge Base

This directory holds the 20-document corpus that the piRAG retriever
indexes. **The contents are synthesised domain notes**, not verbatim
extracts of regulatory or research text. Each document is a short
paraphrase summarising the relevant standard, SOP, or guideline that a
field-deployed system would index.

## Honest framing

The synthesis was done so that the corpus is internally consistent
with the simulator's Arrhenius-Baranyi parameters and the SLCA / cold-
chain constants used by the rest of AGRI-BRAIN. As a side effect, some
factual statements are tuned to make the simulation's metrics scan —
notably the line in `regulatory_fda_leafy_greens.txt` claiming the
effective decay rate "approximately doubles" between 4 °C and 10 °C
(this is a paraphrase of the simulator's Arrhenius output at those
temperatures, not a direct citation).

A reviewer evaluating the retrieval and reranking quality should treat
this corpus as a **labelled retrieval benchmark constructed for the
simulator**, not as an external knowledge source. Any test of
generalisation to a new domain should use a fresh corpus.

## Citations referenced (paraphrased, not verbatim)

- FSMA Produce Safety Rule, Section 204 (Food Traceability Final Rule).
- EU Directive 2008/98/EC (Waste Framework Directive).
- ISO 14040 (LCA framework) and ISO 14044 (LCA requirements).
- USDA Cold Chain Best Practices (paraphrase of public guidance).
- EIP-2535 (Diamond Standard for upgradable smart contracts).

## How to extend

To extend the corpus with verbatim regulatory text:

1. Confirm the source is public domain (FSMA 204 text is; commercial
   guidance often is not).
2. Place the verbatim file under
   `agribrain/backend/pirag/knowledge_base/`.
3. Re-run the simulator's piRAG ingestion pipeline so the new
   document enters the BM25/TF-IDF index.

The retrieval pipeline does not currently distinguish synthesised
notes from verbatim text in its scoring — both go through the same
`HybridRetriever` and `lexical_arrhenius_rerank`. If you mix the two,
add a `metadata["source_type"]` field and adjust scoring accordingly.
