import { describe, expect, it } from "vitest";
import {
  DEFAULT_RUNTIME_PHASE,
  runtimePhaseOrDefault,
} from "./runtimePhase.js";

describe("development runtime phase fallback", () => {
  it("fails safe to monitoring when the response omits or corrupts phase", () => {
    expect(DEFAULT_RUNTIME_PHASE).toBe("monitoring");
    expect(runtimePhaseOrDefault(undefined)).toBe("monitoring");
    expect(runtimePhaseOrDefault(null)).toBe("monitoring");
    expect(runtimePhaseOrDefault("unexpected")).toBe("monitoring");
  });

  it("preserves each explicit compatibility phase", () => {
    expect(runtimePhaseOrDefault("monitoring")).toBe("monitoring");
    expect(runtimePhaseOrDefault("advisory")).toBe("advisory");
    expect(runtimePhaseOrDefault("autonomous")).toBe("autonomous");
  });
});
