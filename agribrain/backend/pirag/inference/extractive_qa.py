"""Extractive question-answering engine for PiRAG.

Selects the most relevant sentence window from retrieved passages
using keyword overlap scoring. Returns the top-scoring span as the
answer along with source attribution and a confidence score.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "and but or nor not so yet both either neither each every all any "
    "few more most other some such no only own same than too very it "
    "its this that these those what which who whom how when where why".split()
)


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def _sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class ExtractiveQA:
    """Extracts the most relevant span from passages.

    Parameters
    ----------
    window_size : number of sentences in the extraction window.
    """

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = window_size

    def answer(
        self,
        question: str,
        passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract the best-matching sentence window.

        Parameters
        ----------
        question : the user query.
        passages : retrieved passages with ``id``, ``text``, ``score``,
            ``metadata`` keys.

        Returns
        -------
        Dict with ``answer``, ``source_id``, ``confidence``, ``span``.
        """
        q_tokens = set(_tokenize(question))
        max_possible = max(len(q_tokens), 1)

        best_score = -1.0
        best_span = ""
        best_source = ""

        for p in passages:
            sentences = _sentence_split(p.get("text", ""))
            if not sentences:
                continue

            for start in range(len(sentences)):
                end = min(start + self.window_size, len(sentences))
                window_text = " ".join(sentences[start:end])
                w_tokens = set(_tokenize(window_text))
                overlap = len(q_tokens & w_tokens)

                if overlap > best_score:
                    best_score = overlap
                    best_span = window_text
                    best_source = p.get("id", "unknown")

        confidence = best_score / max_possible if max_possible > 0 else 0.0

        return {
            "answer": best_span if best_span else "No relevant span found.",
            "source_id": best_source,
            "confidence": min(confidence, 1.0),
            "span": best_span,
        }
