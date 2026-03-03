
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import math, re, hashlib
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class BM25:
    def __init__(self, k1: float = 1.6, b: float = 0.72):
        self.k1, self.b = k1, b
        self.docs: List[Document] = []
        self.df = Counter()
        self.avg_len = 0.0
        self.N = 0
        self.doc_tfs: List[Counter] = []
        self.doc_lens: List[int] = []

    def add(self, docs: List[Document]):
        for d in docs:
            tokens = self._tok(d.text)
            tf = Counter(tokens)
            self.docs.append(d)
            self.doc_tfs.append(tf)
            L = len(tokens)
            self.doc_lens.append(L)
            for t in tf.keys():
                self.df[t] += 1
        self.N = len(self.docs)
        self.avg_len = (sum(self.doc_lens) / max(1, self.N)) if self.N else 0.0

    def _tok(self, s: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", s.lower())

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0: 
            return 0.0
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, q: str, doc_idx: int) -> float:
        q_terms = self._tok(q)
        tf = self.doc_tfs[doc_idx]
        L = self.doc_lens[doc_idx] or 1
        score = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f == 0: 
                continue
            idf = self._idf(t)
            denom = f + self.k1*(1 - self.b + self.b*(L/self.avg_len if self.avg_len else 1.0))
            score += idf * (f*(self.k1+1))/denom
        return score

    def search(self, q: str, k: int = 5) -> List[Tuple[Document, float]]:
        scores = []
        for i, _ in enumerate(self.docs):
            s = self.score(q, i)
            if s > 0: 
                scores.append((self.docs[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class HybridRetriever:
    def __init__(self, dense_model_name: Optional[str] = None):
        self.bm25 = BM25()
        self.docs: List[Document] = []
        self.dense_model = None
        self.doc_vecs = None
        if dense_model_name and SentenceTransformer is not None:
            self.dense_model = SentenceTransformer(dense_model_name)

    def add_documents(self, docs: List[Document]):
        self.docs.extend(docs)
        self.bm25.add(docs)
        if self.dense_model is not None:
            texts = [d.text for d in self.docs]
            self.doc_vecs = self.dense_model.encode(texts, normalize_embeddings=True)

    def _dense_search(self, q: str, k: int = 5):
        if self.dense_model is None or self.doc_vecs is None or np is None:
            return []
        qv = self.dense_model.encode([q], normalize_embeddings=True)[0]
        sims = (self.doc_vecs @ qv)
        idx = sims.argsort()[-k:][::-1]
        return [(self.docs[i], float(sims[i])) for i in idx]

    def search(self, q: str, k: int = 6) -> List[Dict[str, Any]]:
        sparse = self.bm25.search(q, k)
        dense = self._dense_search(q, k)

        def norm(xs):
            if not xs: return []
            vals = [s for _, s in xs]
            mn, mx = min(vals), max(vals)
            rng = (mx - mn) or 1e-9
            return [(d, (s - mn)/rng) for d, s in xs]

        merged: Dict[str, Dict[str, Any]] = {}
        for d, s in norm(sparse):
            merged[d.id] = {"doc": d, "sparse": s, "dense": 0.0}
        for d, s in norm(dense):
            if d.id in merged:
                merged[d.id]["dense"] = s
            else:
                merged[d.id] = {"doc": d, "sparse": 0.0, "dense": s}

        W_sparse, W_dense = (0.6, 0.4) if dense else (1.0, 0.0)
        items = []
        for rec in merged.values():
            score = W_sparse*rec["sparse"] + W_dense*rec["dense"]
            items.append({"id": rec["doc"].id, "score": score, "text": rec["doc"].text, "metadata": rec["doc"].metadata})
        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:k]

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
