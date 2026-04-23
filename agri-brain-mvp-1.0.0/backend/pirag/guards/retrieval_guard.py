"""Retrieval-quality guard for the piRAG routing context pipeline.

Paper Section 3.7 declares three guards on the routing context pipeline:
dimensional analysis (see ``unit_guard.py``), feasibility (see
``feasibility_guard.py``), and retrieval quality (this module). When any
guard returns False the downstream context_to_logits integrator zeroes
the logit modifier, so a bad retrieval cannot degrade decision quality
below the no-context baseline.

The retrieval-quality guard is deliberately simple: a retrieval is usable
when the hybrid (BM25 + TF-IDF) retriever returned at least one citation
and the top passage's combined score exceeds a small non-trivial floor.
The floor 0.15 was selected empirically so purely keyword-proxy hits
(scores near the BM25 + TF-IDF idle floor) do not trigger context
injection.
"""
from __future__ import annotations

from typing import Iterable


# Minimum top citation score for retrieval to be considered usable.
MIN_TOP_CITATION_SCORE: float = 0.15


def retrieval_quality_ok(
    citations: Iterable,
    top_citation_score: float,
    *,
    min_score: float = MIN_TOP_CITATION_SCORE,
) -> bool:
    """Return True when the retrieval result is usable as context.

    Parameters
    ----------
    citations : iterable of citation records from the hybrid retriever.
    top_citation_score : combined BM25 + TF-IDF score of the top passage.
    min_score : threshold below which retrieval is considered low-quality.
        Defaults to ``MIN_TOP_CITATION_SCORE``.

    Returns
    -------
    bool
        ``True`` if at least one citation is present and the top score
        exceeds ``min_score``; ``False`` otherwise.
    """
    try:
        has_citations = len(list(citations)) > 0
    except TypeError:
        has_citations = False
    return bool(has_citations) and float(top_citation_score) > float(min_score)
