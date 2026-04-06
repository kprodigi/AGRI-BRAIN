#!/usr/bin/env python3
"""Run simulation for a single seed and save metrics to benchmark_seeds/.

Usage:
    python run_single_seed.py 42
    python run_single_seed.py 1337
"""
import json
import sys
from pathlib import Path

from generate_results import run_all, SCENARIOS

if len(sys.argv) < 2:
    print("Usage: python run_single_seed.py <seed>")
    sys.exit(1)

seed = int(sys.argv[1])
print(f"Running full simulation with seed={seed}...")
data = run_all(seed=seed)

out_dir = Path(__file__).resolve().parent / "results" / "benchmark_seeds"
out_dir.mkdir(parents=True, exist_ok=True)

metrics = {}
for sc in SCENARIOS:
    metrics[sc] = {}
    for mode in ("agribrain", "mcp_only", "pirag_only", "no_context",
                 "static", "hybrid_rl", "no_pinn", "no_slca"):
        ep = data["results"][sc][mode]
        metrics[sc][mode] = {
            "ari": float(ep["ari"]),
            "waste": float(ep["waste"]),
            "rle": float(ep["rle"]),
            "slca": float(ep["slca"]),
            "carbon": float(ep["carbon"]),
            "equity": float(ep["equity"]),
        }

out_file = out_dir / f"seed_{seed}.json"
out_file.write_text(json.dumps(metrics, indent=2))
print(f"Saved: {out_file}")
