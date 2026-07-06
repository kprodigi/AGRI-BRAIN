"""Run one seed of the instrumented 8-canonical-mode benchmark for the §5.8
H2 channel-attribution analysis. Writes per-seed decision ledgers (with the
observer-only channel-attribution fields) under
``results/decision_ledger_h2/seed_<seed>/``.

The 8 canonical modes are run in their published order so the agribrain
episode sees the same multi-mode global context as the headline benchmark.
This run does NOT overwrite the canonical headline artefacts; it only
produces the instrumented ledgers the new aggregator consumes.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "agribrain" / "backend"))

seed = int(sys.argv[1])
# Optional output-root override (used by the cross-hash-seed stability check).
_root = os.environ.get("H2_LEDGER_ROOT", "decision_ledger_h2")
ledger = HERE / "results" / _root / f"seed_{seed}"
ledger.mkdir(parents=True, exist_ok=True)
os.environ["DECISION_LEDGER_DIR"] = str(ledger)
os.environ.setdefault("DETERMINISTIC_MODE", "false")

import generate_results as gr  # noqa: E402

gr.MODES = ["static", "hybrid_rl", "no_pinn", "no_slca", "agribrain",
            "no_context", "mcp_only", "pirag_only"]

data = gr.run_all(seed=seed)

# Consistency snapshot: per-scenario ARI for every mode, so we can confirm the
# instrumented run reproduces the canonical ranking / margins within CI.
sanity = {}
for sc in gr.SCENARIOS:
    sc_res = data["results"].get(sc, {})
    sanity[sc] = {
        m: {
            "ari": float(sc_res[m]["ari"]),
            "waste": float(sc_res[m]["waste"]),
            "slca": float(sc_res[m]["slca"]),
            "context_influence_rate": float(sc_res[m].get("context_influence_rate", 0.0)),
        }
        for m in gr.MODES if m in sc_res
    }
(ledger / "_ari_sanity.json").write_text(json.dumps(sanity, indent=2))
print(f"SEED {seed} DONE -> {ledger}")
