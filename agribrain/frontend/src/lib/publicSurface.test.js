import { describe, expect, it } from "vitest";
import { decisionReportPath } from "./publicSurface.js";

describe("publication-facing report scope", () => {
  it("does not silently map the all-roles selection to farm", () => {
    expect(decisionReportPath("all")).toBe("/report/pdf");
    expect(decisionReportPath("farm")).toBe("/report/pdf?role=farm");
    expect(decisionReportPath("recovery")).toBe("/report/pdf?role=recovery");
  });
});
