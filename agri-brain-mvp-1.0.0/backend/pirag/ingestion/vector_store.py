"""In-memory vector store for piRAG dense retrieval.

Provides add, search, persist, and load operations for document vectors
with metadata. Uses cosine similarity for nearest-neighbor search.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class VectorStore:
    """In-memory vector store with cosine similarity search.

    Stores document vectors alongside their text and metadata for
    retrieval by the piRAG pipeline.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []

    def add(
        self,
        doc_id: str,
        text: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the store.

        Parameters
        ----------
        doc_id : unique document identifier.
        text : document text content.
        vector : embedding vector.
        metadata : optional metadata dict.
        """
        self._ids.append(doc_id)
        self._texts.append(text)
        self._vectors.append(vector.copy())
        self._metadata.append(metadata or {})

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Find the k most similar documents to the query vector.

        Parameters
        ----------
        query_vector : query embedding.
        k : number of results to return.

        Returns
        -------
        List of dicts with keys: id, text, score, metadata.
        """
        if not self._vectors:
            return []

        mat = np.array(self._vectors)
        q = query_vector.reshape(1, -1)

        # Cosine similarity
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_normalized = q / q_norm

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        mat_normalized = mat / norms

        scores = (mat_normalized @ q_normalized.T).flatten()
        top_k = min(k, len(scores))
        indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in indices:
            results.append({
                "id": self._ids[idx],
                "text": self._texts[idx],
                "score": float(scores[idx]),
                "metadata": self._metadata[idx],
            })
        return results

    def persist(self, path: str) -> None:
        """Save the vector store to disk.

        Parameters
        ----------
        path : directory path for persistence.
        """
        os.makedirs(path, exist_ok=True)
        meta = {
            "ids": self._ids,
            "texts": self._texts,
            "metadata": self._metadata,
        }
        with open(os.path.join(path, "index_meta.json"), "w") as f:
            json.dump(meta, f)
        if self._vectors:
            np.save(os.path.join(path, "vectors.npy"), np.array(self._vectors))

    def load(self, path: str) -> None:
        """Load a vector store from disk.

        Parameters
        ----------
        path : directory path containing persisted data.
        """
        meta_path = os.path.join(path, "index_meta.json")
        vec_path = os.path.join(path, "vectors.npy")

        if not os.path.exists(meta_path):
            return

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self._ids = meta.get("ids", [])
        self._texts = meta.get("texts", [])
        self._metadata = meta.get("metadata", [])

        if os.path.exists(vec_path):
            vecs = np.load(vec_path)
            self._vectors = [vecs[i] for i in range(len(vecs))]
        else:
            self._vectors = []

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self._ids)
