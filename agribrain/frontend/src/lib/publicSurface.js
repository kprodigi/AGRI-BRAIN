/** Publication-facing URL and label helpers. */

export function decisionReportPath(role) {
  const normalized = String(role || "all").trim().toLowerCase();
  return normalized === "all"
    ? "/report/pdf"
    : `/report/pdf?role=${encodeURIComponent(normalized)}`;
}
