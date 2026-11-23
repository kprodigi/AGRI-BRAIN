/** Fail-closed presentation helpers for provenance evidence. */
export function provenanceGuardState(value) {
  if (value === true) return "passed";
  if (value === false) return "failed";
  return "unknown";
}

/** Ethereum transaction hashes are exactly 32 bytes rendered as 0x + 64 hex. */
export function isAnchoredTransactionHash(value) {
  return typeof value === "string" && /^0x[0-9a-fA-F]{64}$/.test(value);
}
