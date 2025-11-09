
from pirag.agent_pipeline import PiRAGPipeline
def test_pipeline_smoke():
    p = PiRAGPipeline()
    p.ingest([{"id":"d1","text":"Spinach shelf life at 4 C is 10 days.","metadata":{"source":"SOP"}}])
    out = p.ask("What is spinach shelf life at 4 C?", k=1, anchor_on_chain=False)
    assert isinstance(out.answer, str)
    assert out.merkle_root is not None
    print("Pipeline smoke OK")
