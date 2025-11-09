
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from .pyrag.hybrid_retriever import HybridRetriever, Document, sha256_hex
from .guards.unit_guard import units_consistent
from .guards.feasibility_guard import within_ranges, verify_with_sim
from .provenance.hasher import hash_text
from .provenance.merkle import merkle_root
from .chain.client import anchor_root as anchor_onchain

@dataclass
class Citation:
    doc_id: str
    passage: str
    sha256: str
    meta: Dict[str, Any]

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
        self.retriever = HybridRetriever(dense_model_name=dense_model_name)

    def ingest(self, docs: List[Dict[str, Any]]):
        self.retriever.add_documents([Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {})) for d in docs])

    def _plan(self, question: str) -> Dict[str, Any]:
        return {"tools": ["retriever","units","sim"], "k": 6, "constraints": {"min": -1e12, "max": 1e12}}

    def _answer_llm(self, question: str, topk: List[Dict[str, Any]]) -> str:
        raise NotImplementedError("Connect your LLM here (OpenAI function tools recommended).")

    def ask(self, question: str, k: int = 6, anchor_on_chain: bool = False) -> PiRAGResponse:
        plan = self._plan(question)
        k = plan.get("k", k)
        hits = self.retriever.search(question, k=k)
        citations: List[Citation] = []
        for h in hits:
            sha = sha256_hex(h["text"])
            citations.append(Citation(doc_id=h["id"], passage=h["text"], sha256=sha, meta=h["metadata"]))

        try:
            answer = self._answer_llm(question, hits)
        except NotImplementedError:
            answer = f"Provisional answer for: {question}\n\nEvidence 1: {hits[0]['text'][:300]}..." if hits else "No evidence retrieved."

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
