import { describe, expect, it } from "vitest";
import {
  isAnchoredTransactionHash,
  provenanceGuardState,
} from "./provenance.js";

describe("fail-closed provenance presentation", () => {
  it("uses three explicit guard states", () => {
    expect(provenanceGuardState(true)).toBe("passed");
    expect(provenanceGuardState(false)).toBe("failed");
    expect(provenanceGuardState(undefined)).toBe("unknown");
    expect(provenanceGuardState(null)).toBe("unknown");
  });

  it("accepts only a full 32-byte transaction hash", () => {
    expect(isAnchoredTransactionHash(`0x${"a".repeat(64)}`)).toBe(true);
    expect(isAnchoredTransactionHash("0x0")).toBe(false);
    expect(isAnchoredTransactionHash("0xabc123")).toBe(false);
    expect(isAnchoredTransactionHash(null)).toBe(false);
  });
});
