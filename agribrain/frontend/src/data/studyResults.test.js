import { describe, expect, it } from "vitest";
import results from "./studyResults.json";

describe("study results shown on the dashboard", () => {
  it("covers three modes, five scenarios and 23 weight settings", () => {
    expect(results.meta.modes).toEqual(["static", "no_context", "agribrain"]);
    expect(results.meta.scenarios).toHaveLength(5);
    expect(results.sensitivity).toHaveLength(23);
  });

  it("reports the overall adaptive resilience index of each mode", () => {
    const ari = results.overall.ari;
    expect(ari.static.mean).toBeCloseTo(0.456, 3);
    expect(ari.no_context.mean).toBeCloseTo(0.587, 3);
    expect(ari.agribrain.mean).toBeCloseTo(0.6108, 4);
  });

  it("keeps the gain over No-Context positive in every scenario and setting", () => {
    for (const scenario of results.meta.scenarios) {
      expect(results.scenarios[scenario].paired_gain.low).toBeGreaterThan(0.01);
    }
    for (const setting of results.sensitivity) {
      expect(setting.low).toBeGreaterThan(0.01);
    }
  });

  it("lists route shares that sum to 100 percent", () => {
    for (const mode of ["no_context", "agribrain"]) {
      const total = results.routes_percent[mode].reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(100, 0);
    }
  });
});
