import { describe, expect, it } from "vitest";
import { fmt, getApiKey, short } from "./utils";

function makeStore() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, String(v)),
    removeItem: (k) => data.delete(k),
    clear: () => data.clear(),
  };
}

describe("utils formatting helpers", () => {
  it("formats numbers with requested precision", () => {
    expect(fmt(1.23456, 2)).toBe("1.23");
    expect(fmt(2, 3)).toBe("2.000");
  });

  it("shortens long hashes and keeps short strings", () => {
    expect(short("0x1234567890abcdef")).toContain("…");
    expect(short("short")).toBe("short");
  });

  it("migrates API key from localStorage to sessionStorage", () => {
    const local = makeStore();
    const session = makeStore();
    globalThis.localStorage = local;
    globalThis.sessionStorage = session;
    local.setItem("API_KEY", "legacy-key");

    const key = getApiKey();
    expect(key).toBe("legacy-key");
    expect(local.getItem("API_KEY")).toBe(null);
    expect(session.getItem("API_KEY")).toBe("legacy-key");
  });
});
