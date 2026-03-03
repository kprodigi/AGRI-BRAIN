
# Physics-informed RAG (PiRAG) + MCP Patch (v1 – full code)

This scaffold adds **Physics-Informed RAG**, an **MCP-style tool interface**, and **cryptographic provenance** with optional **on-chain anchoring** to your existing MVP.
Flow: **Constrain → Retrieve → Verify → Prove → Consensus** (fail-closed).

See `pirag/agent_pipeline.py` (orchestrator) and `pirag/api/routes/rag.py` (FastAPI routes).
Override `_answer_inference` in a subclass to connect your inference backend, as described in the main instructions.
