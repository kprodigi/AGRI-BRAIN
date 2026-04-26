"""TF-IDF vectorizer for piRAG dense retrieval (numpy-only).

Provides a lightweight embedding layer that converts document text into
dense vectors without requiring external ML libraries. The TF-IDF approach
captures term importance for domain-specific retrieval in cold chain and
food safety contexts.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np


class TFIDFEmbedder:
    """Numpy-only TF-IDF vectorizer for document embedding.

    Parameters
    ----------
    max_features : maximum vocabulary size.
    """

    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        """Build vocabulary and compute IDF weights from a corpus.

        Parameters
        ----------
        texts : list of document strings.

        Returns
        -------
        self (for chaining).
        """
        N = len(texts)
        if N == 0:
            self._fitted = True
            return self

        # Build document frequency
        df: Counter = Counter()
        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                df[t] += 1

        # Select top features by document frequency
        top = df.most_common(self.max_features)
        self.vocab = {term: idx for idx, (term, _) in enumerate(top)}

        # Compute IDF: log(N / (1 + df))
        self.idf = np.zeros(len(self.vocab))
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log(N / (1.0 + df.get(term, 0)))

        self._fitted = True
        return self

    def transform(self, text: str) -> np.ndarray:
        """Convert a single text to a TF-IDF vector.

        Parameters
        ----------
        text : input text string.

        Returns
        -------
        Numpy array of shape (vocab_size,).
        """
        if not self._fitted or not self.vocab:
            return np.zeros(1)

        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = np.zeros(len(self.vocab))

        for term, count in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                # Sub-linear TF: 1 + log(tf)
                vec[idx] = (1.0 + math.log(count)) * self.idf[idx]

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def transform_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to TF-IDF vectors.

        Returns
        -------
        Numpy array of shape (len(texts), vocab_size).
        """
        return np.array([self.transform(t) for t in texts])

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return len(self.vocab) if self.vocab else 0
