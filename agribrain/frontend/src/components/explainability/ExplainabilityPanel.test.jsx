/**
 * ExplainabilityPanel tests.
 *
 * The Decisions page renders an ExplainabilityPanel when an
 * explainability blob is present on a decision memo. The panel is the
 * primary user-visible rendering of the paper's policy-trace
 * contracts ([KB:] citation rendering, 5-axis context feature radar,
 * and a local Merkle commitment). These tests pin
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
    operating_envelope_severity: 0.7,
    forecast_urgency: 0.4,
    normalized_fused_rank_strength: 0.85,
    source_labelled_guidance_flag: 0.5,
    recovery_saturation: 0.2,
  },
  logit_adjustment: {
    cold_chain: -0.3,
    local_redistribute: 0.5,
    recovery: 0.2,
  },
  mcp_tools_invoked: ["check_compliance", "spoilage_forecast", "pirag_query"],
  operating_envelope: { compliant: false, violations: [{ severity: "warning" }] },
  forecast: { urgency: "critical", forecast_rho: 0.42 },
  institutional_retrieval_top_doc: "constructed_temperature_excursion_note",
  institutional_retrieval_top_score: 0.0312,
  keywords: {
    institutional_guidance: ["declared cold-chain envelope", "temperature excursion"],
    sop: ["redirect to local market"],
  },
  provenance: {
    evidence_hashes: ["a1b2c3d4e5f6", "0123456789ab"],
    guards_passed: true,
    guard_breakdown: { unit: true, range: true, retrieval: true },
    merkle_root: "abc123def4567890",
  },
  policy_trace_text:
    "The largest recorded calculation component was the operating-envelope signal " +
    "[KB:synthetic_temp_excursion]. The context-ablation delta was also recorded.",
  attribution_chain: { primary_cause: "operating-envelope exceedance" },
  ablation_delta: { cold_chain: -0.3 },
  summary: "An operating-envelope excursion triggered local redistribution.",
};

describe("ExplainabilityPanel", () => {
  it("renders cautious calculation-trace prose", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    expect(screen.getByText(/largest recorded calculation component/i)).toBeInTheDocument();
  });

  it("renders [KB:] citations as badges", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The narrative contains a [KB:] tag; the panel breaks
    // those out into recognisable citation badges.
    expect(screen.getByText(/KB:synthetic_temp_excursion/i)).toBeInTheDocument();
  });

  it("surfaces each invoked project MCP-style tool in the calculation trace", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The local calculation trace renders tool steps as "MCP: <name>" plus
    // (where applicable) an institutional retrieval step. Assert both
    // shapes so a refactor that drops the prefix or the steps
    // surfaces here.
    expect(screen.getAllByText(/MCP:\s*operating-envelope check/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MCP:\s*spoilage_forecast/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MCP:\s*pirag_query/i).length).toBeGreaterThan(0);
    // Institutional retrieval top document step
    expect(screen.getAllByText(/Institutional retrieval:\s*constructed_temperature_excursion_note/i).length).toBeGreaterThan(0);
  });

  it("renders the local Merkle commitment root", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The panel shows a truncated form of the merkle root; we only
    // assert the prefix is reachable to lock the contract that the
    // root is exposed at all.
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
  });

  it("falls back to an explicit 'unavailable' state when the narrative is missing", () => {
    const stripped = { ...explainabilityFixture, policy_trace_text: "", summary: "" };
    render(<ExplainabilityPanel explainability={stripped} memo={memoFixture} />);
    // Fail-loud: an empty narrative must be visible, not silently
    // absent from the panel. (Paper § 4.10 honesty claim.)
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
  });

  it("renders all five canonical context-feature axes", () => {
    render(<ExplainabilityPanel explainability={explainabilityFixture} memo={memoFixture} />);
    // The 5-axis radar uses these labels (matches the paper figure
    // and the FEATURE_LABELS array in the component).
    for (const label of ["Envelope", "Forecast", "Retrieval", "Guidance", "Recovery"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });

  it("keeps legacy stored feature keys as compatibility aliases", () => {
    const legacy = {
      ...explainabilityFixture,
      context_features: {
        compliance_severity: 0.7,
        forecast_urgency: 0.4,
        retrieval_confidence: 0.85,
        regulatory_pressure: 0.5,
        recovery_saturation: 0.2,
      },
    };
    render(<ExplainabilityPanel explainability={legacy} memo={memoFixture} />);
    for (const label of ["Envelope", "Forecast", "Retrieval", "Guidance", "Recovery"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });
});
