export const DEFAULT_RUNTIME_PHASE = "monitoring";

export const RUNTIME_PHASES = Object.freeze([
  "monitoring",
  "advisory",
  "autonomous",
]);

export function runtimePhaseOrDefault(value) {
  return RUNTIME_PHASES.includes(value) ? value : DEFAULT_RUNTIME_PHASE;
}
