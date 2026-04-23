#!/usr/bin/env python3
"""Run simulation for a single seed and save metrics to benchmark_seeds/.

Usage:
    python run_single_seed.py 42
    python run_single_seed.py 1337
    python run_single_seed.py 42 --output-dir /scratch/run_abc123/seed_42
"""
import argparse
import json
from pathlib import Path

try:
    from ..generate_results import run_all, SCENARIOS
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from generate_results import run_all, SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", type=int, help="Seed for this run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write seed_<seed>.json into. Defaults to "
             "mvp/simulation/results/benchmark_seeds/ when omitted.",
    )
    args = parser.parse_args()

    seed = args.seed
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = Path(__file__).resolve().parent.parent / "results" / "benchmark_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running full simulation with seed={seed}...")
    data = run_all(seed=seed)

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


if __name__ == "__main__":
    main()
