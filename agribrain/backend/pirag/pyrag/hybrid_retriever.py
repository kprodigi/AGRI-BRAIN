
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import math
import re
import hashlib
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


from ..ingestion.embedder import TFIDFEmbedder
from ..ingestion.vector_store import VectorStore

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
    def __init__(
        self,
        dense_model_name: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[TFIDFEmbedder] = None,
    ):
        self.bm25 = BM25()
        self.docs: List[Document] = []
        self.dense_model = None
        self.doc_vecs = None
        self.vector_store = vector_store
        self.embedder = embedder
        if dense_model_name and SentenceTransformer is not None:
            self.dense_model = SentenceTransformer(dense_model_name)

    def add_documents(self, docs: List[Document]):
        self.docs.extend(docs)
        self.bm25.add(docs)
        if self.dense_model is not None:
            texts = [d.text for d in self.docs]
            self.doc_vecs = self.dense_model.encode(texts, normalize_embeddings=True)
        elif self.vector_store is not None and self.embedder is not None:
            all_texts = [d.text for d in self.docs]
            self.embedder.fit(all_texts)
            # Re-embed all documents so vectors share the same dimension
            self.vector_store._ids.clear()
            self.vector_store._texts.clear()
            self.vector_store._vectors.clear()
            self.vector_store._metadata.clear()
            for d in self.docs:
                vec = self.embedder.transform(d.text)
                self.vector_store.add(d.id, d.text, vec, d.metadata)

    def _dense_search(self, q: str, k: int = 5):
        # Priority 1: SentenceTransformer
        if self.dense_model is not None and self.doc_vecs is not None:
            qv = self.dense_model.encode([q], normalize_embeddings=True)[0]
            sims = (self.doc_vecs @ qv)
            idx = sims.argsort()[-k:][::-1]
            return [(self.docs[i], float(sims[i])) for i in idx]
        # Priority 2: TF-IDF VectorStore
        if (self.vector_store is not None and self.embedder is not None
                and self.vector_store.size > 0):
            qv = self.embedder.transform(q)
            results = self.vector_store.search(qv, k=k)
            out = []
            for r in results:
                doc = Document(id=r["id"], text=r["text"], metadata=r.get("metadata", {}))
                out.append((doc, r["score"]))
            return out
        # Priority 3: no dense search available
        return []

    # Reciprocal Rank Fusion constant. RRF score for a document at
    # rank r in a list is 1/(RRF_K + r). The standard k=60 (Cormack
    # et al. 2009, "Reciprocal Rank Fusion outperforms Condorcet and
    # individual Rank Learning Methods") is robust across n=20..100
    # corpora; tunable via the constructor if needed.
    RRF_K = 60

    def search(self, q: str, k: int = 6) -> List[Dict[str, Any]]:
        """Hybrid search via Reciprocal Rank Fusion (Cormack 2009).

        Replaces the previous min-max-normalisation merge (which was
        unstable on small corpora because the per-query min and max
        are noisy with n=20 documents and the 0.6/0.4 weights had no
        principled justification). RRF combines two ranked lists by
        summing 1/(K + rank) across lists, which is parameter-light
        (only K) and rank-invariant under monotone score
        transformations of either list.

        Each result carries its sparse rank, dense rank, and the
        component RRF contributions so the per-retriever contribution
        is auditable.
        """
        sparse = self.bm25.search(q, k)
        dense = self._dense_search(q, k)

        merged: Dict[str, Dict[str, Any]] = {}

        for rank, (d, s) in enumerate(sparse, start=1):
            rrf = 1.0 / (self.RRF_K + rank)
            merged.setdefault(d.id, {
                "doc": d,
                "sparse_rank": None, "sparse_score": 0.0, "sparse_rrf": 0.0,
                "dense_rank": None, "dense_score": 0.0, "dense_rrf": 0.0,
            })
            merged[d.id]["sparse_rank"] = rank
            merged[d.id]["sparse_score"] = s
            merged[d.id]["sparse_rrf"] = rrf

        for rank, (d, s) in enumerate(dense, start=1):
            rrf = 1.0 / (self.RRF_K + rank)
            merged.setdefault(d.id, {
                "doc": d,
                "sparse_rank": None, "sparse_score": 0.0, "sparse_rrf": 0.0,
                "dense_rank": None, "dense_score": 0.0, "dense_rrf": 0.0,
            })
            merged[d.id]["dense_rank"] = rank
            merged[d.id]["dense_score"] = s
            merged[d.id]["dense_rrf"] = rrf

        items: List[Dict[str, Any]] = []
        for rec in merged.values():
            score = rec["sparse_rrf"] + rec["dense_rrf"]
            items.append({
                "id": rec["doc"].id,
                "score": score,
                "text": rec["doc"].text,
                "metadata": rec["doc"].metadata,
                "sparse_rank": rec["sparse_rank"],
                "sparse_score": rec["sparse_score"],
                "dense_rank": rec["dense_rank"],
                "dense_score": rec["dense_score"],
                "fusion": "rrf",
            })
        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:k]

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
