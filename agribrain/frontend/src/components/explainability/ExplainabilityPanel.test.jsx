/**
 * ExplainabilityPanel tests.
 *
 * The Decisions page renders an ExplainabilityPanel when an
 * explainability blob is present on a decision memo. The panel is the
 * primary user-visible rendering of the paper's causal-reasoning
 * claims (BECAUSE/WITHOUT highlighting, [KB:] citations, 5-axis
 * context feature radar, Merkle-rooted provenance). These tests pin
 * the contracts that drove those claims so a refactor cannot silently
 * remove or rename any of them.
 */
import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ExplainabilityPanel from "./ExplainabilityPanel";

const memoFixture = {
  agent: "agent:farm",
  role: "farm",
  action: "local_redistribute",
  carbon_kg: 12.34,
  slca: 0.681,
  waste: 0.04,
};

const explainabilityFixture = {
  context_features: {
    compliance_severity: 0.7,
    forecast_urgency: 0.4,
    retrieval_confidence: 0.85,
    regulatory_pressure: 0.5,
    recovery_saturation: 0.2,
  },
  logit_adjustment: {
    cold_chain: -0.3,
    local_redistribute: 0.5,
    recovery: 0.2,
  },
  mcp_tools_invoked: ["check_compliance", "spoilage_forecast", "pirag_query"],
  compliance: { compliant: false, violations: [{ severity: "warning" }] },
  forecast: { urgency: "critical", forecast_rho: 0.42 },
  pirag_top_doc: "kb_fda_temp_excursion",
  pirag_top_score: 0.812,
  keywords: {
    regulatory: ["FDA cold-chain", "temperature excursion"],
    sop: ["redirect to local market"],
  },
  provenance: {
    evidence_hashes: ["a1b2c3d4e5f6", "0123456789ab"],
    guards_passed: true,
    guard_breakdown: { unit: true, range: true, retrieval: true },
    merkle_root: "abc123def4567890",
  },
  causal_text:
    "BECAUSE compliance_severity=0.7 [KB:fda_temp_excursion] AND forecast_urgency=0.4 " +
    "WITHOUT cold_chain capacity available local_redistribute is selected.",
  attribution_chain: { primary_cause: "compliance_severity" },
  ablation_delta: { cold_chain: -0.3 },
  causal_chain: { primary_cause: "compliance_severity" },
  counterfactual: { action: "cold_chain", probs: { cold_chain: 0.6 } },
  summary: "Compliance violation triggered local redistribution.",
};

describe("ExplainabilityPanel", () => {
  it("renders the causal narrative with BECAUSE / WITHOUT emphasis", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // BECAUSE and WITHOUT appear as visually-emphasised tokens.
    expect(screen.getByText("BECAUSE")).toBeInTheDocument();
    expect(screen.getByText("WITHOUT")).toBeInTheDocument();
  });

  it("renders [KB:] citations as badges", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The narrative contains [KB:fda_temp_excursion]; the panel breaks
    // those out into recognisable citation badges.
    expect(screen.getByText(/KB:fda_temp_excursion/i)).toBeInTheDocument();
  });

  it("surfaces each invoked MCP tool in the provenance chain", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The provenance chain renders tool steps as "MCP: <name>" plus
    // (where applicable) a piRAG step "piRAG: <doc>". Assert both
    // shapes so a refactor that drops the prefix or the steps
    // surfaces here.
    expect(screen.getAllByText(/MCP:\s*check_compliance/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MCP:\s*spoilage_forecast/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MCP:\s*pirag_query/i).length).toBeGreaterThan(0);
    // piRAG top doc step
    expect(screen.getAllByText(/piRAG:\s*kb_fda_temp_excursion/i).length).toBeGreaterThan(0);
  });

  it("renders the Merkle root for provenance verification", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The panel shows a truncated form of the merkle root; we only
    // assert the prefix is reachable to lock the contract that the
    // root is exposed at all.
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
  });

  it("falls back to an explicit 'unavailable' state when the narrative is missing", () => {
    const stripped = { ...explainabilityFixture, causal_text: "", summary: "" };
    render(<ExplainabilityPanel explainability={stripped} memo={memoFixture} />);
    // Fail-loud: an empty narrative must be visible, not silently
    // absent from the panel. (Paper § 4.10 honesty claim.)
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
  });

  it("renders all five canonical context-feature axes", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The 5-axis radar uses these labels (matches the paper figure
    // and the FEATURE_LABELS array in the component).
    for (const label of ["Compliance", "Forecast", "Retrieval", "Regulatory", "Recovery"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });
});
