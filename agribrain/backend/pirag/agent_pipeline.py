
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .chain.client import anchor_root as anchor_onchain
from .guards.feasibility_guard import verify_with_sim, within_ranges
from .guards.unit_guard import units_consistent
from .ingestion.embedder import TFIDFEmbedder
from .ingestion.vector_store import VectorStore
from .provenance.merkle import merkle_root
from .pyrag.hybrid_retriever import Document, HybridRetriever, sha256_hex
from .strict_validation import handle_unexpected_failure

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
    # (retrieval-score signal) and psi_3 (retrieved-policy-signal gating)
    # uninformative. The score field below carries the raw reciprocal-rank-
    # fusion strength from the hybrid retriever.
    score: float = 0.0
    retrieval_rank: int = 0
    fused_score: float = 0.0
    sparse_rank: Optional[int] = None
    sparse_score: float = 0.0
    sparse_rrf: float = 0.0
    dense_rank: Optional[int] = None
    dense_score: float = 0.0
    dense_rrf: float = 0.0
    fusion: str = ""

@dataclass
class PiRAGResponse:
    answer: str
    citations: List[Citation]
    guards_passed: bool
    evidence_hashes: List[str]
    merkle_root: str
    chain_tx: Optional[str]
    guard_breakdown: Dict[str, Optional[bool]] = field(default_factory=dict)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

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
            handle_unexpected_failure(
                "knowledge-base discovery",
                FileNotFoundError(f"knowledge-base directory not found: {kb_dir}"),
                _log,
            )
            return
        docs = []
        for f in sorted(kb_dir.iterdir()):
            if f.suffix in (".txt", ".json", ".csv"):
                try:
                    text = f.read_text(encoding="utf-8").strip()
                    if text:
                        docs.append({"id": f.stem, "text": text, "metadata": {"source": f.name}})
                except Exception as _exc:
                    handle_unexpected_failure(
                        f"knowledge-base document read ({f.name})", _exc, _log,
                    )
        if docs:
            self.ingest(docs)
        else:
            handle_unexpected_failure(
                "knowledge-base ingestion",
                RuntimeError(f"no readable corpus documents found in {kb_dir}"),
                _log,
            )

    def ingest(self, docs: List[Dict[str, Any]]):
        self.retriever.add_documents([Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {})) for d in docs])

    def _plan(self, question: str) -> Dict[str, Any]:
        return {"tools": ["retriever","units","sim"], "k": 6, "constraints": {"min": -1e12, "max": 1e12}}

    def _answer_inference(self, question: str, topk: List[Dict[str, Any]]) -> str:
        return self.answer_engine.synthesize(question, topk)

    def ask(self, question: str, k: Optional[int] = None, anchor_on_chain: bool = False) -> PiRAGResponse:
        plan = self._plan(question)
        # An explicit caller value is authoritative.  The planner supplies a
        # default only when the caller omits k; otherwise the simulator's
        # declared k=4 was silently replaced by the planner's k=6.
        effective_k = int(plan.get("k", 6) if k is None else k)
        hits = self.retriever.search(question, k=effective_k)
        citations: List[Citation] = []
        for retrieval_rank, h in enumerate(hits, start=1):
            sha = sha256_hex(h["text"])
            fused_score = float(h.get("fused_score", h.get("score", 0.0)))
            citations.append(Citation(
                doc_id=h["id"],
                passage=h["text"],
                sha256=sha,
                meta=h["metadata"],
                score=fused_score,
                retrieval_rank=retrieval_rank,
                fused_score=fused_score,
                sparse_rank=h.get("sparse_rank"),
                sparse_score=float(h.get("sparse_score", 0.0)),
                sparse_rrf=float(h.get("sparse_rrf", 0.0)),
                dense_rank=h.get("dense_rank"),
                dense_score=float(h.get("dense_score", 0.0)),
                dense_rrf=float(h.get("dense_rrf", 0.0)),
                fusion=str(h.get("fusion", "")),
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
            chain_tx=tx,
            guard_breakdown={
                "unit": bool(u_ok),
                "feasibility": bool(f_ok),
                "simulator": (
                    bool(s_ok) if s_ok is not None else None
                ),
                "aggregate": bool(guards_ok),
            },
            retrieval_metadata={
                "query": question,
                "requested_k": k,
                "planner_default_k": int(plan.get("k", 6)),
                "effective_k": effective_k,
                "returned_count": len(citations),
                "anchor_on_chain": bool(anchor_on_chain),
                "fusion_methods": sorted({
                    citation.fusion
                    for citation in citations if citation.fusion
                }),
            },
        )
