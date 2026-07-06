
from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from ...agent_pipeline import PiRAGPipeline
from src.security import enforce_api_key

router = APIRouter()
_pipe = PiRAGPipeline()

class DocIn(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}

class AskReq(BaseModel):
    question: str
    k: int = 4
    anchor_on_chain: bool = False

@router.post("/ingest")
def ingest(
    docs: List[DocIn],
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    enforce_api_key(request, x_api_key)
    if len(docs) > 200:
        raise HTTPException(400, f"Too many documents ({len(docs)}); max 200 per request")
    _pipe.ingest([d.model_dump() for d in docs])
    return {"ok": True, "n": len(docs)}


@router.get("/kb")
def list_kb(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    """Return the live piRAG knowledge base (static + dynamically synthesised entries).

    Each document is tagged ``static`` (loaded from
    ``pirag/knowledge_base/`` at startup) or ``synthesised`` (added at
    runtime by the dynamic-knowledge feedback loop, see
    ``pirag/dynamic_knowledge.py``). Section 3.7 / Section 4.13 promise
    that synthesised entries are surfaced alongside the static documents
    in the KB browser; this endpoint is what the panel reads.
    """
    enforce_api_key(request, x_api_key)
    docs = []
    for d in getattr(_pipe.retriever, "docs", []):
        meta = dict(getattr(d, "metadata", {}) or {})
        source = str(meta.get("source", "")).lower()
        kind = "synthesised" if source in ("decision_feedback", "dynamic", "synthesised", "synthesized") else "static"
        docs.append({
            "id": d.id,
            "kind": kind,
            "char_count": len(d.text or ""),
            "preview": (d.text or "")[:240],
            "metadata": meta,
        })
    static = [d for d in docs if d["kind"] == "static"]
    synth = [d for d in docs if d["kind"] == "synthesised"]
    return {
        "total": len(docs),
        "static_count": len(static),
        "synthesised_count": len(synth),
        "documents": docs,
    }

@router.post("/ask")
def ask(
    req: AskReq,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    enforce_api_key(request, x_api_key)
    try:
        out = _pipe.ask(req.question, k=req.k, anchor_on_chain=req.anchor_on_chain)
        return {
            "answer": out.answer,
            "citations": [{
                "doc_id": c.doc_id,
                "sha256": c.sha256,
                "metadata": c.meta,
                "excerpt": c.passage[:240]
            } for c in out.citations],
            "guards_passed": out.guards_passed,
            "merkle_root": out.merkle_root,
            "chain_tx": out.chain_tx
        }
    except Exception:
        raise HTTPException(400, "PiRAG query failed")
