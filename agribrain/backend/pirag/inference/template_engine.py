"""Template-based answer synthesis engine for PiRAG.

Composes retrieved passages into structured, source-cited answers
without requiring an external LLM API. Suitable for reproducible
research demonstrations of the RAG architecture.
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


class TemplateAnswerEngine:
    """Synthesises structured answers from retrieved passages.

    Parameters
    ----------
    max_passages : maximum number of passages to include.
    max_passage_length : truncate passage summaries beyond this.
    """

    def __init__(
        self,
        max_passages: int = 3,
        max_passage_length: int = 500,
    ) -> None:
        self.max_passages = max_passages
        self.max_passage_length = max_passage_length

    def synthesize(
        self,
        question: str,
        passages: List[Dict[str, Any]],
    ) -> str:
        """Build a structured answer with citations.

        Parameters
        ----------
        question : the user query.
        passages : retrieved passages, each with ``id``, ``text``,
            ``score``, and ``metadata`` keys.

        Returns
        -------
        Formatted answer string with source citations and key findings.
        """
        if not passages:
            return "No evidence retrieved."

        top = passages[: self.max_passages]
        n = len(top)

        lines = [f"Based on {n} relevant source{'s' if n != 1 else ''}:\n"]

        for i, p in enumerate(top, 1):
            summary = self._summarize_passage(
                p.get("text", ""), self.max_passage_length
            )
            doc_id = p.get("id", "unknown")
            score = p.get("score", 0.0)
            lines.append(
                f"[{i}] {summary} (Source: {doc_id}, Relevance: {score:.2f})"
            )

        key_sentences = self._extract_key_sentences(question, top)
        if key_sentences:
            lines.append(f'\nKey findings related to "{question}":')
            for sent in key_sentences:
                lines.append(f"- {sent}")

        return "\n".join(lines)

    def _extract_key_sentences(
        self,
        question: str,
        passages: List[Dict[str, Any]],
        max_sentences: int = 3,
    ) -> List[str]:
        """Extract top sentences by keyword overlap with the question."""
        q_tokens = set(_tokenize(question))
        if not q_tokens:
            return []

        scored: List[tuple] = []
        seen: set = set()
        for p in passages:
            for sent in _sentence_split(p.get("text", "")):
                if sent in seen:
                    continue
                s_tokens = set(_tokenize(sent))
                overlap = len(q_tokens & s_tokens)
                if overlap > 0:
                    scored.append((overlap, sent))
                    seen.add(sent)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_sentences]]

    def _summarize_passage(self, text: str, max_length: int) -> str:
        """Truncate text at a sentence boundary before *max_length*."""
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        # Find last sentence-ending punctuation
        last_period = max(
            truncated.rfind("."),
            truncated.rfind("!"),
            truncated.rfind("?"),
        )
        if last_period > 0:
            truncated = truncated[: last_period + 1]
        return truncated + "..."
