#!/usr/bin/env python3
"""Export paper-ready evidence tables from decision traces.

Reads trace JSON files produced by generate_results.py and generates
formatted outputs for the paper:

1. Role x information table (which MCP tools / piRAG docs each role uses)
2. Sample decision explanations with provenance chains
3. Context feature activation heatmap data (role x feature x scenario)
4. MCP interoperability protocol trace examples
5. Provenance chain examples with Merkle roots

Standalone usage:
    cd mvp/simulation
    python export_paper_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]

FEATURE_NAMES = [
    "compliance_severity", "forecast_urgency",
    "retrieval_confidence", "regulatory_pressure", "recovery_saturation",
]


def load_traces(scenario: str) -> list:
    path = RESULTS_DIR / f"traces_{scenario}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def export_role_table() -> None:
    """Print role x information table across all scenarios."""
    print("=" * 80)
    print("Table: Role-Specific MCP Tool Usage and piRAG Retrieval Patterns")
    print("=" * 80)

    role_data: dict = {}
    for scenario in SCENARIOS:
        traces = load_traces(scenario)
        for t in traces:
            role = t["step"]["role"]
            if role not in role_data:
                role_data[role] = {"tools": set(), "docs": [], "guidance": [], "features": [], "n": 0}
            rd = role_data[role]
            rd["tools"].update(t["mcp_tools"]["invoked"])
            doc = t["pirag_retrieval"].get("top_document", "")
            if doc:
                rd["docs"].append(doc)
            for gtype in ["regulatory_guidance", "sop_guidance", "waste_hierarchy", "governance"]:
                if t["pirag_retrieval"].get(gtype):
                    rd["guidance"].append(gtype.replace("_guidance", ""))
            feats = t["context_decision"].get("features", {})
            if feats:
                rd["features"].append([feats.get(fn, 0.0) for fn in FEATURE_NAMES])
            rd["n"] += 1

    print(f"\n{'Role':<14s} {'MCP Tools':<40s} {'Primary KB Doc':<35s} {'Guidance':<18s} {'Mean psi'}")
    print("-" * 140)
    for role in sorted(role_data):
        rd = role_data[role]
        from collections import Counter
        top_doc = Counter(rd["docs"]).most_common(1)
        top_doc = top_doc[0][0][:32] if top_doc else "none"
        top_guide = Counter(rd["guidance"]).most_common(1)
        top_guide = top_guide[0][0] if top_guide else "none"
        tools_str = ", ".join(sorted(rd["tools"]))[:38]
        mean_f = np.mean(rd["features"], axis=0) if rd["features"] else np.zeros(5)
        feat_str = "[" + ", ".join(f"{v:.2f}" for v in mean_f) + "]"
        print(f"{role:<14s} {tools_str:<40s} {top_doc:<35s} {top_guide:<18s} {feat_str}")


def export_sample_explanation() -> None:
    """Print a sample decision trace with full provenance chain."""
    print("\n" + "=" * 80)
    print("Sample Decision Trace with Provenance Chain")
    print("=" * 80)

    # Find a trace with provenance and compliance violation
    for scenario in ["heatwave", "cyber_outage", "baseline"]:
        traces = load_traces(scenario)
        for t in traces:
            if (t["provenance"]["provenance_ready"]
                    and t["mcp_tools"].get("compliance")
                    and not t["mcp_tools"]["compliance"].get("compliant", True)):
                _print_trace(t, scenario)
                return

    # Fallback: first trace with any provenance
    for scenario in SCENARIOS:
        traces = load_traces(scenario)
        for t in traces:
            if t["provenance"]["provenance_ready"]:
                _print_trace(t, scenario)
                return

    print("  No provenance-ready traces found.")


def _print_trace(t: dict, scenario: str) -> None:
    s = t["step"]
    o = t["observation"]
    m = t["mcp_tools"]
    p = t["pirag_retrieval"]
    c = t["context_decision"]
    prov = t["provenance"]

    print(f"\nDecision Trace - Hour {s['hour']}, {s['role']} Agent, {scenario} Scenario")
    print("-" * 70)
    print(f"State: rho={o['rho']:.3f}, T={o['temperature']:.1f}C, "
          f"RH={o['humidity']:.0f}%, inventory={o['inventory']:.0f}")
    print()

    print("MCP Tool Outputs:")
    if m.get("compliance"):
        comp = m["compliance"]
        status = "COMPLIANT" if comp.get("compliant") else "VIOLATION"
        print(f"  check_compliance -> {status}")
        for v in comp.get("violations", []):
            print(f"    {v.get('parameter', '?')}: {v.get('value', '?')} "
                  f"(limit {v.get('limit', '?')}, {v.get('severity', '?')})")
    if m.get("forecast"):
        fc = m["forecast"]
        print(f"  spoilage_forecast -> rho={fc.get('forecast_rho', '?')} "
              f"({fc.get('urgency', '?')})")
    print()

    print("piRAG Retrieved Guidance:")
    print(f"  Top document: {p.get('top_document', 'none')} (score: {p.get('top_score', 0):.2f})")
    for gtype in ["regulatory_guidance", "sop_guidance", "waste_hierarchy", "governance"]:
        text = p.get(gtype, "")
        if text:
            print(f"  [{gtype}]: {text[:120]}...")
    print()

    feats = c.get("features", {})
    if feats:
        print("Context Features: psi =", [f"{feats.get(fn, 0):.2f}" for fn in FEATURE_NAMES])
        for fn in FEATURE_NAMES:
            v = feats.get(fn, 0)
            if v > 0.01:
                print(f"  {fn}={v:.2f}")

    logits = c.get("logit_adjustment", {})
    if logits:
        print(f"\nLogit Adjustment: CC={logits.get('ColdChain', 0):+.2f}, "
              f"LR={logits.get('LocalRedistribute', 0):+.2f}, "
              f"Rec={logits.get('Recovery', 0):+.2f}")

    probs = c.get("probabilities", {})
    if probs:
        print(f"Action: {s['action']} (prob CC={probs.get('ColdChain', 0):.3f}, "
              f"LR={probs.get('LocalRedistribute', 0):.3f}, "
              f"Rec={probs.get('Recovery', 0):.3f})")

    if s.get("governance_override"):
        print("[GOVERNANCE OVERRIDE: MCP compliance + forecast mandate rerouting]")

    print(f"\nProvenance Chain:")
    print(f"  Evidence items: {prov.get('total_evidence_items', 0)}")
    for h in prov.get("evidence_hashes", [])[:3]:
        print(f"  SHA-256: {h[:16]}...")
    if prov.get("merkle_root"):
        print(f"  Merkle root: {prov['merkle_root'][:16]}...")
    print(f"  Provenance ready: {prov.get('provenance_ready', False)}")


def export_feature_heatmap_data() -> None:
    """Export role x feature x scenario data for heatmap figure."""
    print("\n" + "=" * 80)
    print("Context Feature Activation Heatmap (mean psi per role per scenario)")
    print("=" * 80)

    heatmap: dict = {}
    for scenario in SCENARIOS:
        traces = load_traces(scenario)
        heatmap[scenario] = {}
        for t in traces:
            role = t["step"]["role"]
            feats = t["context_decision"].get("features", {})
            if not feats:
                continue
            if role not in heatmap[scenario]:
                heatmap[scenario][role] = {"sum": np.zeros(5), "n": 0}
            heatmap[scenario][role]["sum"] += np.array([feats.get(fn, 0) for fn in FEATURE_NAMES])
            heatmap[scenario][role]["n"] += 1

    # Print as a table
    roles_all = sorted({r for s in heatmap.values() for r in s})
    header = f"{'Scenario':<18s} {'Role':<14s} " + " ".join(f"{fn[:10]:>10s}" for fn in FEATURE_NAMES)
    print(header)
    print("-" * len(header))
    for scenario in SCENARIOS:
        for role in roles_all:
            rd = heatmap.get(scenario, {}).get(role)
            if rd and rd["n"] > 0:
                means = rd["sum"] / rd["n"]
                vals = " ".join(f"{v:10.3f}" for v in means)
                print(f"{scenario:<18s} {role:<14s} {vals}")

    # Save as JSON for figure generation
    out_path = RESULTS_DIR / "feature_heatmap_data.json"
    json_data = {}
    for scenario in SCENARIOS:
        json_data[scenario] = {}
        for role in roles_all:
            rd = heatmap.get(scenario, {}).get(role)
            if rd and rd["n"] > 0:
                json_data[scenario][role] = (rd["sum"] / rd["n"]).tolist()
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved to {out_path}")


def export_interop_summary() -> None:
    """Print MCP interoperability protocol summary."""
    print("\n" + "=" * 80)
    print("MCP Interoperability Protocol Traces")
    print("=" * 80)

    for scenario in SCENARIOS:
        path = RESULTS_DIR / f"mcp_interop_{scenario}.json"
        if not path.exists():
            continue
        with open(path) as f:
            interop = json.load(f)
        if not interop:
            continue

        print(f"\n  [{scenario}] {len(interop)} sample interactions")
        for entry in interop[:2]:
            print(f"    Hour {entry['hour']}, {entry['role']} agent, "
                  f"{entry['total_protocol_messages']} JSON-RPC messages")
            for msg in entry["mcp_interactions"][:3]:
                method = msg["request"]["method"]
                resp = msg.get("response_summary", msg.get("response", {}).get("result", {}).get("capabilities", "..."))
                print(f"      {method} -> {str(resp)[:80]}")


def export_provenance_summary() -> None:
    """Print provenance chain summary across all scenarios."""
    print("\n" + "=" * 80)
    print("Provenance Chain Summary")
    print("=" * 80)

    total_chains = 0
    total_hashes = 0
    for scenario in SCENARIOS:
        traces = load_traces(scenario)
        chains = [t for t in traces if t["provenance"]["provenance_ready"]]
        n_hashes = sum(t["provenance"]["total_evidence_items"] for t in chains)
        total_chains += len(chains)
        total_hashes += n_hashes
        print(f"  {scenario:<20s}: {len(chains)} chains, {n_hashes} evidence hashes")

    print(f"\n  Total: {total_chains} verifiable provenance chains, {total_hashes} evidence items")


def export_robustness_and_benchmark() -> None:
    """Print robustness and benchmark summaries if available."""
    print("\n" + "=" * 80)
    print("Robustness / Benchmark Summary")
    print("=" * 80)

    # MCP protocol robustness summary
    for scenario in SCENARIOS:
        proto = RESULTS_DIR / f"mcp_protocol_{scenario}.json"
        if not proto.exists():
            continue
        with open(proto) as f:
            records = json.load(f)
        methods = {}
        errors = 0
        latencies = []
        for r in records:
            req = r.get("request", {})
            m = req.get("method", "unknown")
            methods[m] = methods.get(m, 0) + 1
            if r.get("response", {}).get("error"):
                errors += 1
            if "latency_ms" in r:
                latencies.append(float(r["latency_ms"]))
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        print(f"  {scenario:<18s} interactions={len(records):<5d} errors={errors:<3d} avg_latency_ms={avg_lat:.2f}")
        print(f"    methods: {methods}")

    bench_path = RESULTS_DIR / "benchmark_summary.json"
    if bench_path.exists():
        print("\n  Multi-seed benchmark (from benchmark_summary.json):")
        data = json.loads(bench_path.read_text(encoding="utf-8"))
        for scenario in SCENARIOS:
            if scenario not in data:
                continue
            agr = data[scenario].get("agribrain", {}).get("ari", {})
            if agr:
                print(
                    f"    {scenario:<18s} ARI mean={agr.get('mean', 0):.3f} "
                    f"CI=[{agr.get('ci_low', 0):.3f}, {agr.get('ci_high', 0):.3f}]"
                )


def export_stress_and_significance() -> None:
    """Print OOD stress degradation and statistical significance summaries."""
    print("\n" + "=" * 80)
    print("Stress-Test / Statistical Significance")
    print("=" * 80)

    stress_path = RESULTS_DIR / "stress_degradation.csv"
    if stress_path.exists():
        import pandas as pd

        df = pd.read_csv(stress_path)
        if not df.empty:
            print("  Mean degradation by stressor (AGRIBRAIN):")
            agg = (
                df[df["Method"] == "agribrain"]
                .groupby("Stressor")[["ari_delta", "waste_delta", "latency_ms_delta"]]
                .mean()
                .reset_index()
            )
            for _, row in agg.iterrows():
                print(
                    f"    {row['Stressor']:<22s} "
                    f"dARI={row['ari_delta']:+.4f} "
                    f"dWaste={row['waste_delta']:+.4f} "
                    f"dLatencyMs={row['latency_ms_delta']:+.2f}"
                )

    sig_path = RESULTS_DIR / "benchmark_significance.json"
    if sig_path.exists():
        data = json.loads(sig_path.read_text(encoding="utf-8"))
        print("\n  Benchmark significance (ARI):")
        for scenario in SCENARIOS:
            sc = data.get(scenario, {})
            for comp in ("agribrain_vs_mcp_only", "agribrain_vs_pirag_only", "agribrain_vs_no_context"):
                rec = sc.get(comp, {}).get("ari")
                if not rec:
                    continue
                print(
                    f"    {scenario:<18s} {comp:<26s} "
                    f"p={rec.get('p_value', 1.0):.4f} d={rec.get('cohens_d', 0.0):+.3f}"
                )


if __name__ == "__main__":
    print("AGRI-BRAIN Paper Evidence Export")
    print("=" * 80)

    export_role_table()
    export_sample_explanation()
    export_feature_heatmap_data()
    export_interop_summary()
    export_provenance_summary()
    export_robustness_and_benchmark()
    export_stress_and_significance()

    print("\nDone.")
