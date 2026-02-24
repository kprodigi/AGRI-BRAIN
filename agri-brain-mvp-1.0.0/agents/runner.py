# agents/runner.py
"""
Multi-episode agent runner for AGRI-BRAIN.

Each episode:
  1. Reloads data (POST /case/load)
  2. Steps through the full time-series calling POST /decide at each step
  3. Computes per-step ARI and tracks cumulative metrics
  4. Prints an episode summary

Usage:
    python -m agents.runner                       # 50 episodes, default API
    python -m agents.runner --episodes 10         # fewer episodes
    API_BASE=http://...:8100 python -m agents.runner
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Any, Dict, List

import requests

API = os.environ.get("API_BASE", "http://127.0.0.1:8100").rstrip("/")
ROLES = ["farm", "processor", "distributor", "retail"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict | None = None, **kw) -> dict | None:
    url = f"{API}{path}"
    try:
        r = requests.post(url, json=payload or {}, timeout=30, **kw)
        if not r.ok:
            print(f"  [ERR] {r.status_code} {r.reason} for POST {path}")
            return None
        return r.json()
    except requests.RequestException as e:
        print(f"  [ERR] {e}")
        return None


def _get(path: str, **kw) -> dict | None:
    url = f"{API}{path}"
    try:
        r = requests.get(url, timeout=30, **kw)
        if not r.ok:
            return None
        return r.json()
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(episode_id: int, seed: int, n_steps: int | None = None) -> Dict[str, Any]:
    """Run one episode and return a summary dict."""
    random.seed(seed)

    # 1) reload data
    resp = _post("/case/load")
    if resp is None:
        print(f"  Episode {episode_id}: failed to load data, skipping")
        return {}

    # 2) figure out how many steps we have
    kpi_resp = _get("/kpis")
    total_records = (kpi_resp or {}).get("records", 100)
    steps = n_steps if n_steps is not None else total_records

    # 3) iterate through the time-series
    ari_vals: List[float] = []
    rle_at_risk = 0
    rle_routed = 0
    total_waste = 0.0
    total_carbon = 0.0
    total_reward = 0.0
    step_count = 0

    for t in range(steps):
        role = ROLES[t % len(ROLES)]
        payload = {
            "agent_id": f"agent:{role}",
            "role": role,
            "step": t,
            "deterministic": True,
        }
        data = _post("/decide", payload)
        if data is None:
            continue

        memo: Dict[str, Any] = data.get("memo", data)
        step_count += 1

        # Extract values
        action = memo.get("action", "")
        rho = memo.get("spoilage_risk", 1.0 - memo.get("shelf_left", 1.0))
        waste = rho
        sc = memo.get("slca_components") or {}
        slca_c = sc.get("composite", memo.get("slca", 0.0))
        carbon = memo.get("carbon_kg", 0.0)
        rd = memo.get("reward_decomposition") or {}
        reward = rd.get("total", 0.0)

        # ARI = (1 - waste) * slca_composite * (1 - rho)
        ari = (1.0 - waste) * slca_c * (1.0 - rho)
        ari_vals.append(ari)

        total_waste += waste
        total_carbon += carbon
        total_reward += reward

        # RLE: at-risk if rho > 0.3
        if rho > 0.3:
            rle_at_risk += 1
            if action in ("local_redistribute", "recovery"):
                rle_routed += 1

    # 4) Episode summary
    mean_ari = sum(ari_vals) / max(len(ari_vals), 1)
    rle = rle_routed / max(rle_at_risk, 1)
    avg_waste = total_waste / max(step_count, 1)

    summary = {
        "episode": episode_id,
        "seed": seed,
        "steps": step_count,
        "mean_ari": round(mean_ari, 4),
        "rle": round(rle, 4),
        "avg_waste": round(avg_waste, 4),
        "total_carbon_kg": round(total_carbon, 4),
        "total_reward": round(total_reward, 4),
    }

    print(
        f"  Ep {episode_id:3d}  |  steps={step_count:4d}  "
        f"ARI={mean_ari:.4f}  RLE={rle:.4f}  "
        f"waste={avg_waste:.4f}  carbon={total_carbon:.2f} kg  "
        f"reward={total_reward:.2f}"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AGRI-BRAIN multi-episode agent runner")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=None,
                        help="Steps per episode (default: all records)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--api", type=str, default=None, help="API base URL override")
    args = parser.parse_args()

    global API
    if args.api:
        API = args.api.rstrip("/")

    print(f"AGRI-BRAIN Agent Runner")
    print(f"  API:      {API}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed:     {args.seed}")
    print("-" * 72)

    summaries: List[Dict[str, Any]] = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        s = run_episode(ep, seed, n_steps=args.steps)
        if s:
            summaries.append(s)

    # --------------- aggregate summary -----------------------------------
    if not summaries:
        print("\nNo episodes completed.")
        return

    n = len(summaries)
    agg_ari = sum(s["mean_ari"] for s in summaries) / n
    agg_rle = sum(s["rle"] for s in summaries) / n
    agg_waste = sum(s["avg_waste"] for s in summaries) / n
    agg_carbon = sum(s["total_carbon_kg"] for s in summaries) / n
    agg_reward = sum(s["total_reward"] for s in summaries) / n

    # Fetch final KPIs from the server
    final_kpis = _get("/kpis") or {}

    print("=" * 72)
    print(f"AGGREGATE  ({n} episodes)")
    print(f"  mean ARI:           {agg_ari:.4f}")
    print(f"  mean RLE:           {agg_rle:.4f}")
    print(f"  mean waste:         {agg_waste:.4f}")
    print(f"  mean carbon/ep:     {agg_carbon:.2f} kg")
    print(f"  mean reward/ep:     {agg_reward:.2f}")
    print(f"  total_energy_J:     {final_kpis.get('total_energy_J', 'N/A')}")
    print(f"  total_water_L:      {final_kpis.get('total_water_L', 'N/A')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
