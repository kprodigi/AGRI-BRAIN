
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .pyrag.hybrid_retriever import HybridRetriever, Document, sha256_hex
from .ingestion.embedder import TFIDFEmbedder
from .ingestion.vector_store import VectorStore
from .guards.unit_guard import units_consistent
from .guards.feasibility_guard import within_ranges, verify_with_sim
from .provenance.merkle import merkle_root
from .chain.client import anchor_root as anchor_onchain

_log = logging.getLogger(__name__)


@dataclass
class Citation:
    doc_id: str
    passage: str
    sha256: str
    meta: Dict[str, Any]
    # Implementation note: 2025-04 retrieval-score propagation fix.
    # Earlier versions discarded the BM25/dense hybrid score returned by
    # HybridRetriever.search and downstream code substituted a hardcoded
    # 0.5. That made `top_citation_score` constant and rendered psi_2
    # (retrieval confidence) and psi_3 (regulatory pressure gating)
    # uninformative. The score field below carries the real W_sparse *
    # BM25 + W_dense * cosine score from the hybrid retriever.
    score: float = 0.0

@dataclass
class PiRAGResponse:
    answer: str
    citations: List[Citation]
    guards_passed: bool
    evidence_hashes: List[str]
    merkle_root: str
    chain_tx: Optional[str]

class PiRAGPipeline:
    def __init__(self, dense_model_name: Optional[str] = None):
        self._embedder = TFIDFEmbedder()
        self._vector_store = VectorStore()
        self.retriever = HybridRetriever(
            dense_model_name=dense_model_name,
            vector_store=self._vector_store,
            embedder=self._embedder,
        )
        from .inference.llm_engine import get_engine
        self.answer_engine = get_engine()

        # Auto-ingest knowledge base documents on init
        self._ingest_knowledge_base()

    def _ingest_knowledge_base(self):
        """Auto-ingest documents from the knowledge_base directory."""
        from pathlib import Path
        kb_dir = Path(__file__).parent / "knowledge_base"
        if not kb_dir.exists():
            return
        docs = []
        for f in sorted(kb_dir.iterdir()):
            if f.suffix in (".txt", ".json", ".csv"):
                try:
                    text = f.read_text(encoding="utf-8").strip()
                    if text:
                        docs.append({"id": f.stem, "text": text, "metadata": {"source": f.name}})
                except Exception as _exc:
                    _log.debug("corpus doc %s skipped: %s", f.name, _exc)
        if docs:
            self.ingest(docs)

    def ingest(self, docs: List[Dict[str, Any]]):
        self.retriever.add_documents([Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {})) for d in docs])

    def _plan(self, question: str) -> Dict[str, Any]:
        return {"tools": ["retriever","units","sim"], "k": 6, "constraints": {"min": -1e12, "max": 1e12}}

    def _answer_inference(self, question: str, topk: List[Dict[str, Any]]) -> str:
        return self.answer_engine.synthesize(question, topk)

    def ask(self, question: str, k: int = 6, anchor_on_chain: bool = False) -> PiRAGResponse:
        plan = self._plan(question)
        k = plan.get("k", k)
        hits = self.retriever.search(question, k=k)
        citations: List[Citation] = []
        for h in hits:
            sha = sha256_hex(h["text"])
            citations.append(Citation(
                doc_id=h["id"],
                passage=h["text"],
                sha256=sha,
                meta=h["metadata"],
                score=float(h.get("score", 0.0)),
            ))

        if hits:
            answer = self._answer_inference(question, hits)
        else:
            answer = "No evidence retrieved."

        u_ok = units_consistent(answer)
        f_ok = within_ranges(answer, plan["constraints"])
        s_ok = verify_with_sim(answer, {"question": question, "hits": hits})
        guards_ok = all([u_ok, f_ok, s_ok])

        if not guards_ok:
            answer = "Cannot return a confident answer: guard checks failed. (Units/Feasibility/Simulator)"

        evidence_hashes = [c.sha256 for c in citations]
        root = merkle_root(evidence_hashes) if evidence_hashes else ""

        tx = None
        if anchor_on_chain and root:
            tx = anchor_onchain(root, policy_uri=os.getenv("POLICY_URI",""))

        return PiRAGResponse(
            answer=answer,
            citations=citations,
            guards_passed=guards_ok,
            evidence_hashes=evidence_hashes,
            merkle_root=root,
            chain_tx=tx
        )
