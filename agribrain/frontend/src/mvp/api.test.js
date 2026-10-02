import { describe, it, expect, beforeEach, vi } from "vitest";
import { getApiBase, setApiBase, memoPdfUrl, Scenarios, Decide } from "./api.js";

describe("api base URL helpers", () => {
  beforeEach(() => {
    localStorage.clear();
    delete window.API_BASE;
    setApiBase("http://127.0.0.1:8100");
  });

  it("getApiBase returns normalized URL without trailing slash", () => {
    setApiBase("http://127.0.0.1:8100/");
    expect(getApiBase()).toBe("http://127.0.0.1:8100");
  });

  it("memoPdfUrl appends /report/pdf", () => {
    expect(memoPdfUrl()).toBe("http://127.0.0.1:8100/report/pdf");
  });
});

describe("Scenarios API", () => {
  beforeEach(() => {
    localStorage.clear();
    setApiBase("http://127.0.0.1:8100");
    vi.restoreAllMocks();
  });

  it("list returns remote payload when /scenarios/list is OK", async () => {
    const payload = { scenarios: [{ id: "heatwave", label: "H" }], active: "heatwave" };
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(payload),
      })
    );
    const out = await Scenarios.list();
    expect(out).toEqual(payload);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8100/scenarios/list",
      expect.objectContaining({ headers: expect.any(Object) })
    );
  });
});

describe("Decide API", () => {
  beforeEach(() => {
    localStorage.clear();
    setApiBase("http://127.0.0.1:8100");
    vi.restoreAllMocks();
  });

  it("once parses memo from wrapped response", async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            memo: {
              agent: "demo:farm",
              action: "cold_chain",
              slca: 0.5,
              carbon_kg: 12,
            },
          }),
      })
    );
    const out = await Decide.once({ role: "farm" });
    expect(out.agent).toBe("demo:farm");
    expect(out.action).toBe("cold_chain");
    expect(out.slca_score).toBe(0.5);
  });
});
