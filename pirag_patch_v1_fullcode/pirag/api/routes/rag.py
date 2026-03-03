
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from ...agent_pipeline import PiRAGPipeline

router = APIRouter()
_pipe = PiRAGPipeline()

class DocIn(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}

class AskReq(BaseModel):
    question: str
    k: int = 6
    anchor_on_chain: bool = False

@router.post("/ingest")
def ingest(docs: List[DocIn]):
    _pipe.ingest([d.dict() for d in docs])
    return {"ok": True, "n": len(docs)}

@router.post("/ask")
def ask(req: AskReq):
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
    except Exception as e:
        raise HTTPException(400, f"PiRAG error: {e}")
