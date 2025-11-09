
from typing import List, Dict, Any
def faithfulness_at_k(answer: str, citations: List[Dict[str,Any]], k: int = 3) -> float:
    if not citations:
        return 0.0
    top = citations[:k]
    ok = 0
    for c in top:
        if any(tok in c.get("excerpt","").lower() for tok in answer.lower().split()[:10]):
            ok += 1
    return ok / k
def attribution_precision_recall(pred_cites: List[str], true_cites: List[str]):
    ps = set(pred_cites); ts = set(true_cites)
    tp = len(ps & ts)
    prec = tp / max(1, len(ps))
    rec = tp / max(1, len(ts))
    return prec, rec
def evidence_coverage(answer: str, citations: List[Dict[str,Any]]) -> float:
    sents = [s.strip() for s in answer.split(".") if s.strip()]
    if not sents: return 1.0
    covered = 0
    for _ in sents:
        if citations: covered += 1
    return covered/len(sents)
def expected_calibration_error(bins: List[float], confs: List[float], correct: List[bool]) -> float:
    assert len(confs) == len(correct)
    m = len(confs); ece = 0.0
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        idx = [j for j,c in enumerate(confs) if lo <= c < hi]
        if not idx: continue
        acc = sum(correct[j] for j in idx)/len(idx)
        conf = sum(confs[j] for j in idx)/len(idx)
        ece += (len(idx)/m) * abs(acc - conf)
    return ece
def grounded_answer_change_rate(ans_base: str, ans_perturbed: str) -> float:
    return 0.0 if ans_base.strip() == ans_perturbed.strip() else 1.0
