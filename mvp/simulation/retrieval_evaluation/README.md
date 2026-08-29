# Independent retrieval-ranking evaluation

This interface supports one narrow comparison: the rankings returned by
`agribrain_standard_rag` (Standard RAG) versus the piRAG retrieval variant in
`agribrain`. It does **not** evaluate the Adaptive Resilience Index (ARI),
routing outcomes, or operational performance. Retrieval-quality evidence and
downstream simulation evidence must remain separate.

No relevance judgments or numerical results are included in this repository.
The files under `templates/` are deliberately incomplete and fail validation.
A retrieval-superiority statement is therefore not supported until a completed
bundle of independently supplied, blinded human judgments passes the validator
and the predeclared primary interval rule.

## Required study workflow

1. Copy all six files in `templates/` to a new study directory. Do not edit the
   templates in place.
2. Freeze the query set, query identifiers, document identifiers, system code
   revisions, configurations, and run depth before any relevance assessment.
   `query_text_sha256` is the SHA-256 of the exact UTF-8 query text.
3. Run both systems over every fixed query. Each system must return consecutive
   ranks `1..run_depth`. Pool and content-deduplicate the retrieved documents.
4. Remove every system label from the assessment material and randomize item
   presentation. Assessors must not know which system returned a document.
5. Obtain judgments from people who did not design, develop, tune, or select
   either evaluated retrieval system. Record one row per assessor in
   `assessors.csv`. `assessor_id` is deliberately pseudonymous (for example,
   `assessor_A01`); personal names, email addresses, and other personal data are
   neither requested nor needed. A provenance record identifier can point to a
   separately governed study log or attestation.
6. Each assessor judges every unique `(query_id, doc_id)` pair in the pooled
   top-depth results. Use integer `0/1` for `binary_0_1`, or integer `0/1/2` for
   `ordinal_0_2`. The validator never silently converts a missing judgment to
   nonrelevant.
7. Fill the metadata attestations and provenance fields. Print the byte-level
   hashes of the five CSV inputs, copy them into `input_sha256`, then validate:

   ```text
   python -m mvp.simulation.retrieval_evaluation.validate_retrieval_evaluation STUDY_DIR --print-observed-hashes
   python -m mvp.simulation.retrieval_evaluation.validate_retrieval_evaluation STUDY_DIR
   ```

8. Only after validation succeeds, run the analysis:

   ```text
   python -m mvp.simulation.retrieval_evaluation.analyze_retrieval_quality STUDY_DIR --output retrieval_quality_results.json
   ```

## File schema

The normative metadata schema is
`retrieval_evaluation.schema.json`. The validator also enforces constraints
that JSON Schema cannot express across CSV files.

| File | Required content |
|---|---|
| `evaluation_metadata.json` | Exact systems/configurations, frozen/blinded design attestations, judgment scale, predeclared primary nDCG cutoff, bootstrap settings, and all input hashes |
| `query_set.csv` | Stable query ID and text, text hash, stratum, and a provenance record ID |
| `document_catalog.csv` | Stable document ID, content hash, source record ID, and non-sensitive descriptor |
| `retrieval_runs.csv` | Query, exact system ID, consecutive rank, document ID, optional finite score, and run record ID |
| `assessors.csv` | Pseudonymous assessor ID, expertise category, independence/blinding attestations, provenance record ID, and UTC completion time |
| `relevance_judgments.csv` | Query, document, pseudonymous assessor, allowed integer relevance, and judgment record ID |

The five CSV headers must match the templates exactly. IDs may contain letters,
digits, `.`, `_`, `:`, and `-`; assessor IDs must begin with `assessor_`.
Document content hashes must be unique after pooling. All returned documents
must be judged by all declared assessors.

## Analysis and interpretation

For each assessor and query, the analysis computes nDCG, precision, reciprocal
rank, judged-pool recall, and judged-pool average precision at each predeclared
cutoff. It averages assessor-specific metrics within each query, then compares
systems using paired query-level differences. Uncertainty is a seeded
percentile bootstrap that resamples queries; its confidence level and number of
resamples are locked in metadata.

Only nDCG at `primary_cutoff` is inferential. The recorded directional rule is
satisfied only if the paired piRAG-minus-Standard-RAG interval has a lower
bound above zero. Other metrics and cutoffs are descriptive. Any statement is
limited to the fixed query set, judged pool, run depth, and assessor cohort.
Judged-pool recall is not full-corpus recall.

The validator can verify completeness, file integrity, and recorded
attestations. It cannot independently prove that an assessor was independent
or remained blinded; the external provenance records are therefore essential.
