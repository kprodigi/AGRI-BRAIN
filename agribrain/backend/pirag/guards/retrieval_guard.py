"""Retrieval-quality guard for the piRAG routing context pipeline.

Paper Section 3.7 declares three guards on the routing context pipeline:
dimensional analysis (see ``unit_guard.py``), feasibility (see
``feasibility_guard.py``), and retrieval quality (this module). When any
guard returns False the downstream context_to_logits integrator zeroes
the logit modifier, so a bad retrieval cannot degrade decision quality
below the no-context baseline.

The retrieval-quality guard is deliberately simple: a retrieval is
usable when the hybrid (BM25 + TF-IDF) retriever returned at least
one citation and the top passage's combined score exceeds a small
non-trivial floor.

**Threshold rescale, 2026-04.** The hybrid retriever was changed from
min-max-normalised score fusion to Reciprocal Rank Fusion (Cormack
2009; see ``pyrag/hybrid_retriever.py``). RRF scores are bounded by
``1/(K + 1)`` with ``K = 60``, so the top score is at most ~0.0164
and the previous floor of ``0.15`` would never pass. The floor is
rescaled to ``0.012`` (about 75% of the maximum RRF score, i.e.
"either retriever ranks the top doc better than rank 4 of 4") so the
guard still gates idle-floor retrievals while staying compatible
with the RRF score distribution. To verify the guard calibration,
compare against ``HybridRetriever.RRF_K``.
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
