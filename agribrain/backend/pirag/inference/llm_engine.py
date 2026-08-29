"""LLM abstraction layer for piRAG explanation synthesis.

Provides a pluggable engine interface that supports:
  - Template-based synthesis (default, no external dependencies)
  - API-based LLM calls via any standard chat-completions-compatible endpoint

Supports any standard chat-completions-compatible endpoint including locally hosted
LLaMA, Falcon, or Mistral via Ollama/vLLM.

Configuration via environment variables:
  LLM_API_URL  - endpoint URL (e.g. http://localhost:11434/v1)
  LLM_MODEL    - model identifier (e.g. llama3, falcon-7b)
  LLM_API_KEY  - API key (if required by the endpoint)
"""
from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from src.settings import SETTINGS


class ExplanationEngine(ABC):
    """Abstract base for explanation synthesis engines."""

    @abstractmethod
    def synthesize(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Generate an explanation from a question and retrieved evidence.

        Parameters
        ----------
        question : the user's query.
        evidence : list of retrieved passages with id, text, score, metadata.

        Returns
        -------
        Synthesized explanation string.
        """


class TemplateEngine(ExplanationEngine):
    """Template-based engine wrapping the existing TemplateAnswerEngine."""

    def __init__(self) -> None:
        from .template_engine import TemplateAnswerEngine
        self._engine = TemplateAnswerEngine()

    def synthesize(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        return self._engine.synthesize(question, evidence)


class APIEngine(ExplanationEngine):
    """API-based engine calling an external LLM endpoint.

    Uses the standard chat-completions-compatible chat completions format. Falls back to
    the template engine on any failure (timeout, connection error, etc.).
    """

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 30,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._fallback = TemplateEngine()

    def synthesize(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[{i+1}] {p.get('text', '')}" for i, p in enumerate(evidence[:5])
        )
        prompt = (
            f"Based on the following evidence, answer the question concisely.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.3,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return self._fallback.synthesize(question, evidence)
        except Exception:
            return self._fallback.synthesize(question, evidence)


def get_engine() -> ExplanationEngine:
    """Factory: returns APIEngine if env vars set, else TemplateEngine.

    Environment variables:
        LLM_API_URL - API endpoint URL
        LLM_MODEL   - model name/identifier
        LLM_API_KEY - API key (optional for local endpoints)
    """
    if SETTINGS.llm_provider == "template":
        return TemplateEngine()

    api_url = os.environ.get("LLM_API_URL", "")
    model = os.environ.get("LLM_MODEL", "")

    if api_url and model:
        api_key = os.environ.get("LLM_API_KEY", "")
        return APIEngine(api_url=api_url, model=model, api_key=api_key)

    return TemplateEngine()
