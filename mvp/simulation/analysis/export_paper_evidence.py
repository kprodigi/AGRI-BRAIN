#!/usr/bin/env python3
"""Export the canonical paper-ready benchmark evidence table.

The publication path reads only the completed multi-seed benchmark summary
and significance files. Older single-run trace reports remain available as
an explicitly requested diagnostic because the parallel seed runner can leave
top-level trace files from whichever worker finished last; they must never be
silently mixed into the canonical publication evidence.

1. Role x information table (which MCP tools / piR docs each role uses)
2. Sample decision explanations with local Merkle commitment records
3. Context feature activation heatmap data (role x feature x scenario)
4. In-process project JSON-RPC/MCP-style dispatcher trace examples
5. Local Merkle commitment-record examples

Standalone publication usage:
    cd <repository-root>
    python mvp/simulation/analysis/export_paper_evidence.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"]

FEATURE_NAMES = [
    "compliance_severity", "forecast_urgency",
    "retrieval_confidence", "regulatory_pressure", "recovery_saturation",
]


def _publication_export_identity_errors(
    bench_payload: dict[str, Any],
    sig_payload: dict[str, Any],
) -> list[str]:
    """Return source/run identity errors before writing a canonical artifact."""

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from hpc.validate_source_checkout import validation_errors

    errors: list[str] = []
    declared_commit = os.environ.get("AGRIBRAIN_GIT_COMMIT", "").strip()
    declared_run_tag = os.environ.get("RUN_TAG", "").strip()
    if not declared_run_tag:
        errors.append("RUN_TAG must identify the publication run")

    recovery_path = os.environ.get("AGRIBRAIN_RECOVERY_RECEIPT", "").strip()
    simulation_commit = declared_commit
    publication_commit = declared_commit
    expected_dual = False
    checkout_environment = dict(os.environ)
    if recovery_path:
        simulation_commit = os.environ.get(
            "AGRIBRAIN_SIMULATION_COMMIT", ""
        ).strip()
        publication_commit = os.environ.get(
            "AGRIBRAIN_PUBLICATION_CODE_COMMIT", ""
        ).strip()
        original_receipt = os.environ.get(
            "CORE_SUBMISSION_RECEIPT", ""
        ).strip()
        actual_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
        if not original_receipt:
            errors.append(
                "CORE_SUBMISSION_RECEIPT is required for publication recovery"
            )
        elif not actual_job_id:
            errors.append("SLURM_JOB_ID is required for publication recovery")
        else:
            try:
                from hpc.publication_recovery_receipt import (
                    validate_recovery_receipt_file,
                )

                # The recovery launcher exports these receipt paths relative
                # to the repository root, but this stage runs from
                # mvp/simulation, so a bare Path() would resolve them against
                # the wrong base directory.
                recovery_receipt_file = Path(recovery_path)
                if not recovery_receipt_file.is_absolute():
                    recovery_receipt_file = REPO_ROOT / recovery_receipt_file
                original_receipt_file = Path(original_receipt)
                if not original_receipt_file.is_absolute():
                    original_receipt_file = REPO_ROOT / original_receipt_file
                validate_recovery_receipt_file(
                    recovery_receipt_file,
                    original_receipt_path=original_receipt_file,
                    expected_kind="core",
                    expected_run_tag=declared_run_tag,
                    expected_simulation_commit=simulation_commit,
                    expected_publication_commit=publication_commit,
                    expected_recovery_job_id=actual_job_id,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"publication recovery receipt is invalid: {exc}")
        if not simulation_commit or not publication_commit:
            errors.append("publication recovery commit identities are missing")
        if declared_commit != simulation_commit:
            errors.append(
                "AGRIBRAIN_GIT_COMMIT must retain the simulation commit in recovery"
            )
        if simulation_commit == publication_commit:
            errors.append(
                "publication recovery requires distinct simulation and publication commits"
            )
        checkout_environment["AGRIBRAIN_GIT_COMMIT"] = publication_commit
        expected_dual = simulation_commit != publication_commit

    errors = validation_errors(
        environ=checkout_environment,
        repo_root=REPO_ROOT,
        allow_run_artifacts=True,
    ) + errors

    for label, payload in (
        ("benchmark_summary.json", bench_payload),
        ("benchmark_significance.json", sig_payload),
    ):
        meta = payload.get("_meta") if isinstance(payload, dict) else None
        if not isinstance(meta, dict):
            errors.append(f"{label} lacks _meta source identity")
            continue
        expected_identity = {
            "git_commit": simulation_commit,
            "source_commit": simulation_commit,
            "simulation_source_commit": simulation_commit,
            "analysis_code_commit": publication_commit,
            "dual_provenance": expected_dual,
        }
        for key, expected in expected_identity.items():
            if meta.get(key) != expected:
                errors.append(
                    f"{label} {key} does not equal the authorized "
                    "simulation/publication identity"
                )
        if meta.get("run_tag") != declared_run_tag:
            errors.append(f"{label} run_tag does not equal RUN_TAG")
    return errors


def load_traces(scenario: str) -> list:
    path = RESULTS_DIR / f"traces_{scenario}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def export_role_table() -> None:
    """Print role x information table across all scenarios."""
    print("=" * 80)
    print("Table: Role-Specific MCP Tool Usage and piR Retrieval Patterns")
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
    """Print a sample trace with its local Merkle commitment record."""
    print("\n" + "=" * 80)
    print("Sample Decision Trace with Local Merkle Commitment Record")
    print("=" * 80)

    # Prefer a trace outside the declared synthetic operating envelope.
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

    print("  No traces with local Merkle commitments found.")


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
        status = "within declared synthetic envelope" if comp.get("compliant") else "outside declared synthetic envelope"
        print(f"  operating-envelope check -> {status}")
        for v in comp.get("violations", []):
            print(f"    {v.get('parameter', '?')}: {v.get('value', '?')} "
                  f"(limit {v.get('limit', '?')}, {v.get('severity', '?')})")
    if m.get("forecast"):
        fc = m["forecast"]
        print(f"  spoilage_forecast -> rho={fc.get('forecast_rho', '?')} "
              f"({fc.get('urgency', '?')})")
    print()

    print("piR Retrieved Guidance:")
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
        print(
            "[AUTHOR-DECLARED PROBABILITY-GAP OVERRIDE: pi(cold_chain) < 0.005 and "
            "pi(local_redistribute) - pi(cold_chain) > 0.80]"
        )

    print("\nLocal Merkle Commitment Record:")
    print(f"  Committed evidence items: {prov.get('total_evidence_items', 0)}")
    for h in prov.get("evidence_hashes", [])[:3]:
        print(f"  SHA-256: {h[:16]}...")
    if prov.get("merkle_root"):
        print(f"  Merkle root: {prov['merkle_root'][:16]}...")
    print(f"  Local commitment present: {prov.get('provenance_ready', False)}")


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
    """Print the in-process project JSON-RPC/MCP-style trace summary."""
    print("\n" + "=" * 80)
    print("In-Process Project JSON-RPC/MCP-Style Dispatcher Traces")
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
    """Print local Merkle commitment-record summary across scenarios."""
    print("\n" + "=" * 80)
    print("Local Merkle Commitment-Record Summary")
    print("=" * 80)

    total_chains = 0
    total_hashes = 0
    for scenario in SCENARIOS:
        traces = load_traces(scenario)
        chains = [t for t in traces if t["provenance"]["provenance_ready"]]
        n_hashes = sum(t["provenance"]["total_evidence_items"] for t in chains)
        total_chains += len(chains)
        total_hashes += n_hashes
        print(f"  {scenario:<20s}: {len(chains)} records, {n_hashes} committed evidence hashes")

    print(
        f"\n  Total: {total_chains} local Merkle commitment records, "
        f"{total_hashes} committed evidence items"
    )


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
        payload = json.loads(bench_path.read_text(encoding="utf-8"))
        data = (
            payload.get("summary", payload)
            if isinstance(payload, dict) else {}
        )
        for scenario in SCENARIOS:
            if scenario not in data:
                continue
            agr = data[scenario].get("agribrain", {}).get("ari", {})
            if agr:
                lo = agr.get("ci_low", agr.get("mean", 0.0))
                hi = agr.get("ci_high", agr.get("mean", 0.0))
                lo = agr.get("mean", 0.0) if lo is None else lo
                hi = agr.get("mean", 0.0) if hi is None else hi
                print(
                    f"    {scenario:<18s} ARI mean={agr.get('mean', 0):.3f} "
                    f"CI=[{float(lo):.3f}, {float(hi):.3f}]"
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

    stress_pf_path = RESULTS_DIR / "stress_passfail.csv"
    if stress_pf_path.exists():
        import pandas as pd

        pf = pd.read_csv(stress_pf_path)
        if not pf.empty:
            formal = pf[
                (pf["Method"] == "agribrain")
                & (pf.get("comparison_type", "") != "cross_mode_under_stress")
            ]
            total = len(formal)
            passed = int((formal["Pass_Equivalence"] == True).sum())  # noqa: E712
            print(f"\n  Formal H3 equivalence cells: {passed}/{total} equivalent")

    sig_path = RESULTS_DIR / "benchmark_significance.json"
    if sig_path.exists():
        payload = json.loads(sig_path.read_text(encoding="utf-8"))
        data = (
            payload.get("significance", payload)
            if isinstance(payload, dict) else {}
        )
        # Tolerant of n=1 degenerate-sample fallback records: when an
        # aggregator skips inferential tests, p_value / CIs come back
        # as null. Format those as "n/a" rather than crashing the
        # f-string. The matched-design effect is d_z; pooled d is retained as
        # a separate descriptive standardization, so print the two explicitly
        # rather than relying on the legacy ``cohens_d`` alias.
        def _f(v: Any, fmt: str = "+.4f", default: str = "  n/a ") -> str:
            if v is None:
                return default
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return default

        print("\n  Confirmatory directional benchmark tests (ARI):")
        confirmatory = (
            ("agribrain_vs_no_context", "H1", "p_value_adj_holm"),
            ("mcp_only_vs_no_context", "H2", "p_value_adj_holm_h2_directional"),
            ("pirag_only_vs_no_context", "H2", "p_value_adj_holm_h2_directional"),
            ("agribrain_vs_mcp_only", "H2", "p_value_adj_holm_h2_directional"),
            ("agribrain_vs_pirag_only", "H2", "p_value_adj_holm_h2_directional"),
        )
        for scenario in SCENARIOS:
            sc = data.get(scenario, {})
            for comp, family, adjusted_field in confirmatory:
                rec = sc.get(comp, {}).get("ari")
                if not rec:
                    continue
                degen = " [degen]" if rec.get("_degenerate") else ""
                print(
                    f"    {scenario:<18s} {family:<2s} {comp:<26s} "
                    f"p_dir={_f(rec.get('p_value_directional_greater'), '.4f')} "
                    f"p_holm={_f(rec.get(adjusted_field), '.4f')} "
                    f"dz={_f(rec.get('cohens_dz'), '+.3f')} "
                    f"d_pooled={_f(rec.get('cohens_d_pooled'), '+.3f')} "
                    f"dMean={_f(rec.get('mean_diff'), '+.4f')} "
                    f"CI=[{_f(rec.get('mean_diff_ci_low'), '+.4f')},"
                    f"{_f(rec.get('mean_diff_ci_high'), '+.4f')}]{degen}"
                )


def export_latex_benchmark_table() -> None:
    """Export a LaTeX-ready benchmark table with mean +/- CI and p-values."""
    print("\n" + "=" * 80)
    print("LaTeX-Ready Stochastic Benchmark Table")
    print("=" * 80)

    bench_path = RESULTS_DIR / "benchmark_summary.json"
    sig_path = RESULTS_DIR / "benchmark_significance.json"
    if not bench_path.exists():
        print(
            "  benchmark_summary.json not found — run the canonical "
            "hpc/hpc_run.sh workflow; its dependent hpc/hpc_publish.sh "
            "stage invokes aggregate_seeds.py."
        )
        return

    bench_payload = json.loads(bench_path.read_text(encoding="utf-8"))
    # Unwrap the aggregator's {"_meta": ..., "summary": {...}} envelope.
    # The previous code read the top-level dict and silently produced an
    # empty LaTeX table because every bench.get(scenario) returned {}.
    bench = (
        bench_payload["summary"]
        if isinstance(bench_payload, dict) and isinstance(bench_payload.get("summary"), dict)
        else bench_payload
    )
    sig_payload = json.loads(sig_path.read_text(encoding="utf-8")) if sig_path.exists() else {}
    identity_errors = _publication_export_identity_errors(
        bench_payload, sig_payload,
    )
    if identity_errors:
        raise RuntimeError(
            "Publication export blocked by source/run identity errors:\n  - "
            + "\n  - ".join(identity_errors)
        )
    sig = (
        sig_payload.get("significance", sig_payload)
        if isinstance(sig_payload, dict)
        else {}
    )

    # Exact locked eleven-arm benchmark panel: eight primary modes followed by
    # three secondary one-factor ablations. The action-specific b_tau
    # coordinates belong to the separate structural-sensitivity design, not
    # to additional benchmark rows.
    methods = [
        "static", "hybrid_rl", "no_pinn", "no_slca", "no_context",
        "mcp_only", "pirag_only", "agribrain",
        "agribrain_standard_rag", "agribrain_no_peer",
        "agribrain_sign_unconstrained",
    ]
    metrics = ["ari", "waste", "slca", "rle", "carbon", "equity"]

    # Print human-readable table
    print(f"\n  {'Scenario':<18s} {'Method':<14s} {'Metric':>6s} {'Mean':>8s} {'95% CI':>18s} {'Std':>8s}")
    print("  " + "-" * 76)
    for scenario in SCENARIOS:
        for method in methods:
            m_data = bench.get(scenario, {}).get(method, {})
            if not m_data:
                continue
            for metric in metrics:
                d = m_data.get(metric, {})
                if not d:
                    continue
                # CI / mean / std may be null under the n=1
                # degenerate-sample fallback. Format as "n/a" when
                # absent rather than crashing.
                def _num(key: str, fmt: str = ".4f") -> str:
                    val = d.get(key)
                    if isinstance(val, (int, float)):
                        return format(float(val), fmt)
                    return "n/a"
                ci_str = f"[{_num('ci_low'):>8s}, {_num('ci_high'):>8s}]"
                mean_str = _num("mean")
                std_str = _num("std", ".6f")
                print(f"  {scenario:<18s} {method:<14s} {metric:>6s} "
                      f"{mean_str:>8s} {ci_str:>18s} {std_str:>8s}")

    # Print only the prespecified confirmatory directional tests. Generic
    # two-sided p-values remain in the machine-readable artifact for secondary
    # analyses, but labeling those as the paper-ready H1/H2 evidence would be
    # statistically incorrect.
    if sig:
        print(
            f"\n  {'Scenario':<18s} {'Family':<6s} {'Comparison':<30s} "
            f"{'p-dir':>8s} {'p-Holm':>8s} {'Cohen dz':>8s} {'Mean diff':>10s}"
        )
        print("  " + "-" * 103)
        confirmatory = (
            ("agribrain_vs_no_context", "H1", "p_value_adj_holm"),
            ("mcp_only_vs_no_context", "H2", "p_value_adj_holm_h2_directional"),
            ("pirag_only_vs_no_context", "H2", "p_value_adj_holm_h2_directional"),
            ("agribrain_vs_mcp_only", "H2", "p_value_adj_holm_h2_directional"),
            ("agribrain_vs_pirag_only", "H2", "p_value_adj_holm_h2_directional"),
        )
        for scenario in SCENARIOS:
            sc = sig.get(scenario, {})
            for comp_key, family, adjusted_field in confirmatory:
                comp_data = sc.get(comp_key, {})
                ari = comp_data.get("ari", {})
                if not ari:
                    continue
                p_val = ari.get("p_value_directional_greater")
                p_str = f"{p_val:8.4f}" if isinstance(p_val, (int, float)) else "  n/a  "
                adjusted = ari.get(adjusted_field)
                adjusted_str = (
                    f"{adjusted:8.4f}"
                    if isinstance(adjusted, (int, float)) else "  n/a  "
                )
                d_val = ari.get("cohens_dz")
                d_str = f"{d_val:+8.3f}" if isinstance(d_val, (int, float)) else "  n/a   "
                md_val = ari.get("mean_diff", 0.0)
                md_str = f"{md_val:+10.4f}" if isinstance(md_val, (int, float)) else "  n/a    "
                print(
                    f"  {scenario:<18s} {family:<6s} {comp_key:<30s} "
                    f"{p_str} {adjusted_str} {d_str} {md_str}"
                )

    # Save as JSON for downstream LaTeX generation. The 2026-05 audit
    # caught that this file shipped without a top-level ``_meta`` block,
    # so a reviewer reading the paper-evidence export couldn't see the
    # commit / seed-count it was aggregated from without cross-
    # referencing benchmark_summary.json + artifact_manifest.json.
    # Propagate the upstream ``_meta`` (git_commit, n_seeds,
    # seeds_loaded, bootstrap_alpha, n_boot, bca_fallback_stats) and
    # add ``generated_at`` + a ``source_artifacts`` list so this file
    # self-attributes its provenance.
    from datetime import datetime, timezone
    out_path = RESULTS_DIR / "paper_benchmark_table.json"
    upstream_meta = (
        bench_payload.get("_meta", {})
        if isinstance(bench_payload, dict) else {}
    )
    sig_meta = (
        sig_payload.get("_meta", {})
        if isinstance(sig_payload, dict) else {}
    )
    export = {
        "_meta": {
            "git_commit": upstream_meta.get("git_commit"),
            "source_commit": upstream_meta.get("source_commit"),
            "simulation_source_commit": upstream_meta.get(
                "simulation_source_commit"
            ),
            "analysis_code_commit": upstream_meta.get(
                "analysis_code_commit"
            ),
            "dual_provenance": upstream_meta.get("dual_provenance"),
            "run_tag": upstream_meta.get("run_tag"),
            "n_seeds": upstream_meta.get("n_seeds"),
            "seeds_loaded": upstream_meta.get("seeds_loaded"),
            "bootstrap_alpha": upstream_meta.get("bootstrap_alpha"),
            "n_boot": upstream_meta.get("n_boot"),
            "n_perm": upstream_meta.get("n_perm"),
            "std_ddof": upstream_meta.get("std_ddof"),
            "bca_fallback_stats": upstream_meta.get("bca_fallback_stats"),
            "generated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "source_artifacts": [
                "mvp/simulation/results/benchmark_summary.json",
                "mvp/simulation/results/benchmark_significance.json",
                "mvp/simulation/results/h2_directional_evidence.csv",
            ],
            # Significance correction families are documented in
            # benchmark_significance.json's _meta. Carrying through the
            # primary keys here so a downstream LaTeX generator does
            # not have to open both files to learn which correction
            # family each p_value_adj belongs to.
            "significance_correction_meta": {
                "primary_h1_family": sig_meta.get("primary_h1_family"),
                "primary_h1_correction": sig_meta.get("primary_h1_correction"),
                "h2_directional_family": sig_meta.get(
                    "h2_directional_family"
                ),
                "h2_directional_correction": sig_meta.get(
                    "h2_directional_correction"
                ),
                "h2_directional_canonical_field": sig_meta.get(
                    "h2_directional_canonical_field"
                ),
                "h2_global_support_rule": sig_meta.get(
                    "h2_global_support_rule"
                ),
                "h2_synergy_status": sig_meta.get("h2_synergy_status"),
                "confirmatory_test": sig_meta.get("confirmatory_test"),
                "n_perm_scope": sig_meta.get("n_perm_scope"),
                "channel_decomposition_family": sig_meta.get(
                    "channel_decomposition_family"
                ),
                "channel_decomposition_correction": sig_meta.get(
                    "channel_decomposition_correction"
                ),
                "channel_decomposition_status": sig_meta.get(
                    "channel_decomposition_status"
                ),
                "secondary_correction": sig_meta.get("secondary_correction"),
                "secondary_family_scope": sig_meta.get("secondary_family_scope"),
                "primary_h1_holm_adjusted": (
                    sig_payload.get("primary_h1_holm_adjusted")
                    if isinstance(sig_payload, dict) else None
                ),
                "primary_h1_supported_by_cell": (
                    sig_payload.get("primary_h1_supported_by_cell")
                    if isinstance(sig_payload, dict) else None
                ),
                "primary_h1_supported_all_cells": (
                    sig_payload.get("primary_h1_supported_all_cells")
                    if isinstance(sig_payload, dict) else None
                ),
                "pinn_ablation_family": sig_meta.get("pinn_ablation_family"),
                "pinn_ablation_correction": sig_meta.get(
                    "pinn_ablation_correction"
                ),
                "pinn_ablation_scope": sig_meta.get("pinn_ablation_scope"),
                "pinn_ablation_holm_adjusted": (
                    sig_payload.get("pinn_ablation_holm_adjusted")
                    if isinstance(sig_payload, dict) else None
                ),
                "pinn_ablation_supported_by_cell": (
                    sig_payload.get("pinn_ablation_supported_by_cell")
                    if isinstance(sig_payload, dict) else None
                ),
                "pinn_ablation_supported_all_cells": (
                    sig_payload.get("pinn_ablation_supported_all_cells")
                    if isinstance(sig_payload, dict) else None
                ),
                "h2_directional_holm_adjusted": (
                    sig_payload.get("h2_directional_holm_adjusted")
                    if isinstance(sig_payload, dict) else None
                ),
                "h2_directional_supported_by_cell": (
                    sig_payload.get("h2_directional_supported_by_cell")
                    if isinstance(sig_payload, dict) else None
                ),
                "h2_directional_supported_all_cells": (
                    sig_payload.get("h2_directional_supported_all_cells")
                    if isinstance(sig_payload, dict) else None
                ),
                # Historical two-contrast subset retained for audit only;
                # never use it as the confirmatory H2 correction.
                "channel_decomposition_holm_adjusted": (
                    sig_payload.get("channel_decomposition_holm_adjusted")
                    if isinstance(sig_payload, dict) else None
                ),
            },
        },
        "benchmark": bench,
        "significance": sig,
        "h2_directional_evidence": (
            sig_payload.get("h2_directional_evidence")
            if isinstance(sig_payload, dict) else None
        ),
    }
    out_path.write_text(
        json.dumps(export, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\n  Saved combined export: {out_path}")
    _commit = export["_meta"]["git_commit"]
    _nseeds = export["_meta"]["n_seeds"]
    print(f"  Stamped _meta: git_commit={_commit!s:.16}... "
          f"n_seeds={_nseeds} generated_at={export['_meta']['generated_at']}")


if __name__ == "__main__":
    print("AGRI-BRAIN Paper Evidence Export")
    print("=" * 80)

    import os
    if os.environ.get("EXPORT_LEGACY_SINGLE_RUN_TRACES", "0") == "1":
        print(
            "WARNING: exporting non-canonical single-run trace diagnostics; "
            "do not cite them as 20-seed publication evidence."
        )
        export_role_table()
        export_sample_explanation()
        export_feature_heatmap_data()
        export_interop_summary()
        export_provenance_summary()
        export_robustness_and_benchmark()
        export_stress_and_significance()
    export_latex_benchmark_table()

    print("\nDone.")
