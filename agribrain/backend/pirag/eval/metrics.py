"""Evaluation metrics for retrieval/evidence quality."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split(" ") if t]


def precision_at_k(pred_ids: Sequence[str], true_ids: Sequence[str], k: int = 3) -> float:
    if k <= 0:
        return 0.0
    p = list(pred_ids[:k])
    if not p:
        return 0.0
    t = set(true_ids)
    return sum(1 for x in p if x in t) / float(k)


def recall_at_k(pred_ids: Sequence[str], true_ids: Sequence[str], k: int = 3) -> float:
    t = set(true_ids)
    if not t:
        return 1.0
    p = set(pred_ids[:k])
    return len(p & t) / float(len(t))


def faithfulness_at_k(answer: str, citations: List[Dict[str, Any]], k: int = 3) -> float:
    if not citations:
        return 0.0
    top = citations[:k]
    answer_tokens = set(_tokenize(answer)[:20])
    if not answer_tokens:
        return 0.0
    ok = 0
    for c in top:
        excerpt = c.get("excerpt") or c.get("passage") or ""
        if set(_tokenize(excerpt)) & answer_tokens:
            ok += 1
    return ok / max(1, min(k, len(top)))


def attribution_precision_recall(pred_cites: List[str], true_cites: List[str]) -> Tuple[float, float]:
    ps = set(pred_cites)
    ts = set(true_cites)
    tp = len(ps & ts)
    prec = tp / max(1, len(ps))
    rec = tp / max(1, len(ts))
    return prec, rec


def evidence_coverage(answer: str, citations: List[Dict[str, Any]]) -> float:
    sents = [s.strip() for s in answer.split(".") if s.strip()]
    if not sents:
        return 1.0
    if not citations:
        return 0.0
    cited_text = " ".join((c.get("excerpt") or c.get("passage") or "") for c in citations).lower()
    covered = 0
    for s in sents:
        if any(tok in cited_text for tok in _tokenize(s)[:5]):
            covered += 1
    return covered / len(sents)


def expected_calibration_error(
    bins: List[float], confs: List[float], correct: List[bool]
) -> float:
    assert len(confs) == len(correct)
    if not confs:
        return 0.0
    m = len(confs)
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        idx = [j for j, c in enumerate(confs) if lo <= c < hi]
        if not idx:
            continue
        acc = sum(correct[j] for j in idx) / len(idx)
        conf = sum(confs[j] for j in idx) / len(idx)
        ece += (len(idx) / m) * abs(acc - conf)
    return ece


def brier_score(confs: List[float], correct: List[bool]) -> float:
    if not confs:
        return 0.0
    ys = [1.0 if c else 0.0 for c in correct]
    return sum((p - y) ** 2 for p, y in zip(confs, ys)) / len(confs)


def grounded_answer_change_rate(ans_base: str, ans_perturbed: str) -> float:
    return 0.0 if ans_base.strip() == ans_perturbed.strip() else 1.0


def summarize_retrieval_quality(
    pred_ids: Sequence[str],
    true_ids: Sequence[str],
    answer: str,
    citations: List[Dict[str, Any]],
    confs: Iterable[float] = (),
    correct: Iterable[bool] = (),
) -> Dict[str, float]:
    conf_list = list(confs)
    corr_list = list(correct)
    return {
        "p_at_3": precision_at_k(pred_ids, true_ids, k=3),
        "r_at_3": recall_at_k(pred_ids, true_ids, k=3),
        "faithfulness_at_3": faithfulness_at_k(answer, citations, k=3),
        "evidence_coverage": evidence_coverage(answer, citations),
        "ece_10bin": expected_calibration_error(
            [i / 10 for i in range(11)],
            conf_list,
            corr_list,
        ) if conf_list and len(conf_list) == len(corr_list) else 0.0,
        "brier": brier_score(conf_list, corr_list) if conf_list and len(conf_list) == len(corr_list) else 0.0,
    }
