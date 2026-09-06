"""Author-declared RRF-floor guard for the piRAG routing context pipeline.

Paper Section 3.7 declares three guards on the routing context pipeline:
dimensional analysis (see ``unit_guard.py``), feasibility (see
``feasibility_guard.py``), and an RRF-score floor (this module). When any
guard returns False the downstream context-to-logit integrator zeroes the
piRAG-derived term only. Separately computed MCP signals remain active.
This does not establish guard completeness or guarantee non-degradation of
outcomes.

This guard is deliberately simple: a retrieval is
usable when the hybrid (BM25 + TF-IDF) retriever returned at least
one citation and the top passage's combined score exceeds a small
non-trivial floor.

**Threshold rescale, 2026-04.** The hybrid retriever was changed from
min-max-normalised score fusion to Reciprocal Rank Fusion (Cormack
2009; see ``pyrag/hybrid_retriever.py``). With two ranked lists and
``K = 60``, the maximum fused score is ``2/(K + 1)`` (~0.0328).
The guard therefore operates on the raw RRF strength, before any lexical or
Arrhenius reranking bonus. To verify the score scale, compare against
``HybridRetriever.RRF_K``.

Passing this floor is an implementation gate, not a calibrated probability,
an independent retrieval-quality judgment, or evidence of answer faithfulness.
"""
from __future__ import annotations

from typing import Iterable


# Minimum top citation score for retrieval to be considered usable.
# RRF-scaled. With K=60 the maximum RRF score for a doc top-ranked by
# both retrievers is 2/(K+1) = ~0.0328; for a doc top of ONE retriever
# only it is 1/(K+1) = ~0.0164; rank-3 single-list is 1/(K+3) = ~0.0159.
# The previous floor of 0.012 admitted essentially every non-empty
# result. The new floor is 1.5/(K+1) ≈ 0.0246 — i.e. "either both
# retrievers placed the doc in the top 3, or one retriever placed it
# at rank 1 *plus* the other contributed any score". This gates idle
# retrievals while keeping signal-bearing hits through.
MIN_TOP_CITATION_SCORE: float = 1.5 / 61.0  # ≈ 0.0246


def retrieval_quality_ok(
    citations: Iterable,
    top_citation_score: float,
    *,
    min_score: float = MIN_TOP_CITATION_SCORE,
) -> bool:
    """Return True when the result clears the declared RRF floor.

    The function name is retained as a compatibility API; the returned boolean
    is not a calibrated or independently validated measure of retrieval quality.

    Parameters
    ----------
    citations : iterable of citation records from the hybrid retriever.
    top_citation_score : raw two-list RRF strength of the top passage selected
        after reranking; excludes the reranking bonus.
    min_score : author-declared RRF-score floor.
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
