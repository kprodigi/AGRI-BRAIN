"""Compare channel-attribution metrics across PYTHONHASHSEED values to bound
sensitivity of the headline rates to the hash-seed pin. Restricted to the same
4 base seeds in every group for an apples-to-apples comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "benchmarks"))
sys.path.insert(0, str(HERE.parents[1] / "agribrain" / "backend"))
from aggregate_channel_attribution import _decision, _load_episode  # noqa: E402

PERT = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
BASE = [7, 99, 707, 2024]
ROOTS = {
    "hs0": HERE / "results" / "decision_ledger_h2",
    "hs1": HERE / "results" / "decision_ledger_h2_hs1",
    "hs2": HERE / "results" / "decision_ledger_h2_hs2",
}


def scenarios_present(root: Path):
    return {s for s in PERT
            if all((root / f"seed_{b}" / f"agribrain__{s}.jsonl").exists() for b in BASE)}


def metrics_for(root: Path, scen_set):
    n_instr = n_changed = 0
    mcp_nec = pir_nec = syn = 0
    mcp_only = pir_only = redundant = 0
    for seed in BASE:
        for scn in scen_set:
            p = root / f"seed_{seed}" / f"agribrain__{scn}.jsonl"
            if not p.exists():
                continue
            for r in _load_episode(p):
                base = r.get("base_logits"); cm = r.get("context_modifier")
                ss = r.get("slca_shaping"); amp = r.get("slca_amp"); T = r.get("policy_temperature")
                mm = r.get("modifier_mcp"); mp = r.get("modifier_pirag")
                if None in (base, cm, ss, amp, T, mm, mp) or float(T) <= 0:
                    continue
                T = float(T); z = np.zeros(3)
                d_none = _decision(base, z, ss, amp, T, False)
                d_full = _decision(base, cm, ss, amp, T, True)
                d_mcp = _decision(base, mm, ss, amp, T, True)
                d_pir = _decision(base, mp, ss, amp, T, True)
                n_instr += 1
                if d_full == d_none:
                    continue
                n_changed += 1
                mn = d_full != d_pir   # MCP necessary
                pn = d_full != d_mcp   # piRAG necessary
                mcp_nec += mn; pir_nec += pn
                if mn and pn:
                    syn += 1
                elif mn:
                    mcp_only += 1
                elif pn:
                    pir_only += 1
                else:
                    redundant += 1
    if n_instr == 0:
        return None
    return {
        "n_instr": n_instr, "n_changed": n_changed,
        "context_decisive": n_changed / n_instr,
        "mcp_necessary": mcp_nec / n_instr,
        "pirag_necessary": pir_nec / n_instr,
        "synergy": syn / n_instr,
        "complementarity": (mcp_only + pir_only + syn) / n_changed if n_changed else 0.0,
    }


# Use only scenarios that are fully present in EVERY available hash-seed group
# (apples-to-apples). Groups must have >=1 seed dir to count.
avail = {l: r for l, r in ROOTS.items() if r.exists() and any((r / f"seed_{b}").exists() for b in BASE)}
common = set(PERT)
for r in avail.values():
    common &= scenarios_present(r)
common = sorted(common, key=PERT.index)
if not common:
    raise SystemExit("no scenario fully present across all hash-seed groups yet")

rows = {}
for label, root in avail.items():
    m = metrics_for(root, common)
    if m:
        rows[label] = m

keys = ["context_decisive", "mcp_necessary", "pirag_necessary", "synergy", "complementarity"]
print(f"Cross-hash-seed stability (4 base seeds {BASE}, common scenarios = {common})")
print(f"{'metric':18s} " + "  ".join(f"{l:>8s}" for l in rows) + "   range  rel%")
for k in keys:
    vals = [rows[l][k] for l in rows]
    rng = max(vals) - min(vals)
    rel = 100 * rng / (np.mean(vals) + 1e-9)
    print(f"{k:18s} " + "  ".join(f"{rows[l][k]*100:7.2f}%" for l in rows)
          + f"   {rng*100:5.2f}pp  {rel:4.1f}%")
print(f"\nn_instr per group: " + ", ".join(f"{l}={rows[l]['n_instr']}" for l in rows))

payload = {"common_scenarios": common, "base_seeds": BASE, "by_hashseed": rows,
           "ranges": {k: (max(rows[l][k] for l in rows) - min(rows[l][k] for l in rows))
                      for k in keys}}
out = HERE / "results" / "channel_hashseed_stability.json"
out.write_text(json.dumps(payload, indent=2))
print(f"Saved {out}")
