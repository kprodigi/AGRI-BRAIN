import { describe, expect, it } from "vitest";
import {
  PRIMARY_PUBLICATION_MODES, SECONDARY_PUBLICATION_MODES, REGIME_BIAS_VECTOR,
  canonicalH1Evidence, canonicalH2Evidence, directionalAdvantagePercent,
  descriptiveCrossScenarioSummary, withinPanelRelativeScore,
} from "./publicationEvidence.js";

describe("publication evidence contract", () => {
  it("locks the eight primary and three secondary modes", () => {
    expect(PRIMARY_PUBLICATION_MODES).toHaveLength(8);
    expect(SECONDARY_PUBLICATION_MODES).toHaveLength(3);
    expect(new Set([...PRIMARY_PUBLICATION_MODES, ...SECONDARY_PUBLICATION_MODES]).size).toBe(11);
  });

  it("exposes the action-specific regime-bias vector", () => {
    expect(REGIME_BIAS_VECTOR).toEqual([0.25, 0.05, -0.25]);
  });

  it("treats lower waste and emissions as favorable in comparisons", () => {
    expect(directionalAdvantagePercent(0.08, 0.10, "Waste")).toBeCloseTo(20);
    expect(directionalAdvantagePercent(800, 1000, "Carbon")).toBeCloseTo(20);
    expect(directionalAdvantagePercent(0.8, 0.5, "ARI")).toBeCloseTo(60);
    expect(directionalAdvantagePercent(1, 0, "ARI")).toBeNull();
  });

  it("builds transparent direction-aware within-panel radar scores", () => {
    const values = [10, 20, 30];
    expect(withinPanelRelativeScore(10, values, "Carbon")).toBe(1);
    expect(withinPanelRelativeScore(30, values, "Carbon")).toBe(0);
    expect(withinPanelRelativeScore(30, values, "ARI")).toBe(1);
    expect(withinPanelRelativeScore(20, [20, 20], "Waste")).toBe(0.5);
  });

  it("derives the descriptive dashboard summary from exact scenario means", () => {
    const scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"];
    const summary = Object.fromEntries(scenarios.map((scenario, index) => [scenario, {
      agribrain: {
        ari: { mean: 0.7 + index * 0.01 },
        waste: { mean: 0.08 + index * 0.001 },
        carbon: { mean: 80 + index },
        rle: { mean: 0.4 + index * 0.01 },
      },
      static: {
        ari: { mean: 0.5 + index * 0.01 },
        waste: { mean: 0.10 + index * 0.001 },
        carbon: { mean: 100 + index },
        rle: { mean: 0 },
      },
    }]));
    const result = descriptiveCrossScenarioSummary(summary);
    expect(result.scenarioCount).toBe(5);
    expect(result.exactUnweightedMeans.ari.agribrain).toBeCloseTo(0.72);
    expect(result.ariRelativeDifferencePct).toBeCloseTo((0.2 / 0.52) * 100);
    expect(result.wasteRelativeDifferencePct).toBeCloseTo((1 - 0.082 / 0.102) * 100);
    expect(result.carbonRelativeDifferencePct).toBeCloseTo((1 - 82 / 102) * 100);
    expect(result.rleDisplayScore).toBeCloseTo(42);
    delete summary.baseline.agribrain.rle;
    expect(descriptiveCrossScenarioSummary(summary)).toBeNull();
  });

  it("never promotes a raw p-value to confirmatory H1 support", () => {
    expect(canonicalH1Evidence({ p_value: 0.0001, mean_diff: 1 })).toEqual({
      available: false, adjustedP: null, supported: false,
    });
    expect(canonicalH1Evidence({
      p_value_adj_holm: 0.03,
      mean_diff: 0.01,
      h1_positive_effect_supported: true,
      correction_method: "holm_bonferroni_across_scenarios",
    }).supported).toBe(true);
    expect(canonicalH1Evidence({
      p_value_adj_holm: 0.03,
      mean_diff: -0.01,
      h1_positive_effect_supported: true,
      correction_method: "holm_bonferroni_across_scenarios",
    }).available).toBe(false);
  });

  it("fails H2 closed unless the complete canonical 20-cell family is present", () => {
    expect(canonicalH2Evidence(null).supported).toBe(false);
    expect(canonicalH2Evidence({ h2_directional_supported_all_cells: true }).available).toBe(false);
    const scenarios = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline"];
    const comparisons = [
      "mcp_only_vs_no_context", "pirag_only_vs_no_context",
      "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
    ];
    const keys = scenarios.flatMap((scenario) => comparisons.map((comparison) => `${scenario}:${comparison}`));
    const support = Object.fromEntries(keys.map((key) => [key, true]));
    const adjusted = Object.fromEntries(keys.map((key) => [key, 0.01]));
    const significance = Object.fromEntries(scenarios.map((scenario) => [scenario,
      Object.fromEntries(comparisons.map((comparison) => [comparison, { ari: {
        mean_diff: 0.01,
        p_value_directional_greater: 0.001,
        p_value_adj_holm_h2_directional: 0.01,
        h2_cell_supported: true,
      } }]))]));
    const valid = {
      h2_directional_supported_all_cells: true,
      h2_directional_supported_by_cell: support,
      h2_directional_holm_adjusted: adjusted,
      significance,
    };
    expect(canonicalH2Evidence(valid).supported).toBe(true);
    valid.h2_directional_supported_by_cell[keys[0]] = false;
    expect(canonicalH2Evidence(valid).available).toBe(false);
  });
});
