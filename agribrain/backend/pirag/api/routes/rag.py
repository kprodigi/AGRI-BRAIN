
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
