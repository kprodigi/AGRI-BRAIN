export const PRIMARY_PUBLICATION_MODES = Object.freeze([
  "static", "hybrid_rl", "no_pinn", "no_slca", "no_context", "mcp_only",
  "pirag_only", "agribrain",
]);

export const SECONDARY_PUBLICATION_MODES = Object.freeze([
  "agribrain_standard_rag", "agribrain_no_peer",
  "agribrain_sign_unconstrained",
]);

export const REGIME_BIAS_VECTOR = Object.freeze([0.25, 0.05, -0.25]);
export const LOWER_IS_BETTER_METRICS = Object.freeze(["Waste", "Carbon"]);

const lowerIsBetter = (metric) => LOWER_IS_BETTER_METRICS.includes(metric);

export function descriptiveCrossScenarioSummary(benchmarkSummary) {
  if (!benchmarkSummary || typeof benchmarkSummary !== "object") return null;
  const exactMeans = (mode, metric) => {
    const values = CONFIRMATORY_SCENARIOS.map((scenario) => (
      Number(benchmarkSummary?.[scenario]?.[mode]?.[metric]?.mean)
    ));
    return values.every(Number.isFinite) ? values : null;
  };
  const mean = (values) => values.reduce((sum, value) => sum + value, 0)
    / values.length;
  const pairs = {};
  for (const metric of ["ari", "waste", "carbon", "rle"]) {
    const agribrain = exactMeans("agribrain", metric);
    const staticPolicy = exactMeans("static", metric);
    if (!agribrain || !staticPolicy) return null;
    pairs[metric] = {
      agribrain: mean(agribrain),
      static: mean(staticPolicy),
    };
  }
  return {
    scenarioCount: CONFIRMATORY_SCENARIOS.length,
    ariRelativeDifferencePct: directionalAdvantagePercent(
      pairs.ari.agribrain, pairs.ari.static, "ARI",
    ),
    wasteRelativeDifferencePct: directionalAdvantagePercent(
      pairs.waste.agribrain, pairs.waste.static, "Waste",
    ),
    carbonRelativeDifferencePct: directionalAdvantagePercent(
      pairs.carbon.agribrain, pairs.carbon.static, "Carbon",
    ),
    rleDisplayScore: pairs.rle.agribrain * 100,
    exactUnweightedMeans: pairs,
  };
}

export function directionalAdvantagePercent(first, reference, metric) {
  const left = Number(first);
  const right = Number(reference);
  if (!Number.isFinite(left) || !Number.isFinite(right) || right === 0) return null;
  const rawPercent = ((left - right) / Math.abs(right)) * 100;
  return lowerIsBetter(metric) ? -rawPercent : rawPercent;
}

export function withinPanelRelativeScore(value, panelValues, metric) {
  const observed = Number(value);
  const finitePanel = (panelValues || []).map(Number).filter(Number.isFinite);
  if (!Number.isFinite(observed) || finitePanel.length === 0) return null;
  const minimum = Math.min(...finitePanel);
  const maximum = Math.max(...finitePanel);
  if (maximum === minimum) return 0.5;
  const raw = (observed - minimum) / (maximum - minimum);
  const directed = lowerIsBetter(metric) ? 1 - raw : raw;
  return Math.max(0, Math.min(1, directed));
}

const CONFIRMATORY_SCENARIOS = Object.freeze([
  "heatwave", "overproduction", "cyber_outage", "adaptive_pricing", "baseline",
]);
const H2_COMPARISONS = Object.freeze([
  "mcp_only_vs_no_context", "pirag_only_vs_no_context",
  "agribrain_vs_mcp_only", "agribrain_vs_pirag_only",
]);

const probability = (value) => {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 && number <= 1 ? number : null;
};

export function canonicalH1Evidence(ari) {
  const adjustedP = probability(ari?.p_value_adj_holm);
  const meanDiff = Number(ari?.mean_diff);
  const derivedSupport = adjustedP !== null && Number.isFinite(meanDiff)
    && meanDiff > 0 && adjustedP < 0.05;
  const complete = adjustedP !== null
    && Number.isFinite(meanDiff)
    && typeof ari?.h1_positive_effect_supported === "boolean"
    && ari?.correction_method === "holm_bonferroni_across_scenarios"
    && ari.h1_positive_effect_supported === derivedSupport;
  return {
    available: complete,
    adjustedP: complete ? adjustedP : null,
    supported: Boolean(complete && derivedSupport),
  };
}

export function canonicalH2Evidence(payload) {
  const globalSupport = payload?.h2_directional_supported_all_cells;
  const supportMap = payload?.h2_directional_supported_by_cell;
  const adjustedMap = payload?.h2_directional_holm_adjusted;
  const significance = payload?.significance;
  const expectedKeys = CONFIRMATORY_SCENARIOS.flatMap((scenario) => (
    H2_COMPARISONS.map((comparison) => `${scenario}:${comparison}`)
  ));
  const hasExactKeys = (value) => value && typeof value === "object"
    && Object.keys(value).length === expectedKeys.length
    && expectedKeys.every((key) => Object.prototype.hasOwnProperty.call(value, key));
  let complete = typeof globalSupport === "boolean"
    && hasExactKeys(supportMap)
    && hasExactKeys(adjustedMap)
    && significance && typeof significance === "object";
  const derivedMap = {};
  if (complete) {
    for (const scenario of CONFIRMATORY_SCENARIOS) {
      for (const comparison of H2_COMPARISONS) {
        const key = `${scenario}:${comparison}`;
        const record = significance?.[scenario]?.[comparison]?.ari;
        const adjustedP = probability(adjustedMap[key]);
        const cellAdjustedP = probability(record?.p_value_adj_holm_h2_directional);
        const rawP = probability(record?.p_value_directional_greater);
        const meanDiff = Number(record?.mean_diff);
        const derived = adjustedP !== null && rawP !== null
          && Number.isFinite(meanDiff) && meanDiff > 0
          && rawP < 0.05 && adjustedP < 0.05;
        derivedMap[key] = derived;
        if (adjustedP === null || cellAdjustedP === null || rawP === null
          || !Number.isFinite(meanDiff)
          || cellAdjustedP !== adjustedP
          || typeof record?.h2_cell_supported !== "boolean"
          || record.h2_cell_supported !== derived
          || typeof supportMap[key] !== "boolean"
          || supportMap[key] !== derived) {
          complete = false;
        }
      }
    }
  }
  const derivedGlobal = expectedKeys.length > 0
    && expectedKeys.every((key) => derivedMap[key] === true);
  if (complete && globalSupport !== derivedGlobal) complete = false;
  return {
    available: Boolean(complete),
    supported: Boolean(complete && derivedGlobal),
    supportMap: complete ? supportMap : {},
    adjustedMap: complete ? adjustedMap : {},
  };
}
