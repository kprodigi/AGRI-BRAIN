#!/usr/bin/env python3
"""Fail fast if a publication job inherited a treatment-changing setting."""
from __future__ import annotations

import os
import sys

EXPECTED = {
    "APP_ENV": "dev",
    "FORECAST_METHOD": "holt_linear",
    "SUPPLY_FORECAST_METHOD": "persistence",
    "ONLINE_LEARNING": "false",
    "LLM_PROVIDER": "template",
    "SIM_API_BASE": "",
    "DETERMINISTIC_MODE": "false",
    "STOCH_TEMP_STD_C": "2.5",
    "STOCH_RH_STD": "7.0",
    "STOCH_DEMAND_FRAC_STD": "0.25",
    "STOCH_INVENTORY_FRAC_STD": "0.22",
    "STOCH_TRANSPORT_KM_STD": "0.22",
    "STOCH_K_REF_STD": "0.20",
    "STOCH_EA_R_STD": "0.14",
    "STOCH_ONSET_JITTER_H": "6.0",
    "STOCH_THETA_NOISE_STD": "0.15",
    "STOCH_POLICY_TEMP_STD": "0.0",
    "STOCH_DELAY_PROB": "0.10",
    "FAILURE_INJECTION": "false",
    "MCP_RELIABILITY": "false",
    "MCP_QOS_ROUTING": "false",
    "PIRAG_COUNTERFACTUAL": "false",
    "PHYSICS_CONSISTENCY_GATE": "false",
    "HETEROGENEOUS_PROFILES": "false",
    "RESEARCH_METRICS": "false",
    "DYNAMIC_KB_FEEDBACK": "false",
    "MCP_RATE_LIMITS": "disabled",
    "PROTOCOL_MAX_RECORDS": "4096",
    "CHAIN_SUBMIT": "0",
    "BENCHMARK_USE_TABLES": "false",
    "BENCHMARK_WRITE_COMPAT": "false",
    "EXPORT_LEGACY_SINGLE_RUN_TRACES": "0",
    "STRICT_VALIDATION": "1",
    "FULL_EVIDENCE_CAPTURE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

MUST_BE_UNSET = {
    "DATA_CSV",
    "DECISION_LEDGER_DIR",
    "LLM_API_URL", "LLM_MODEL", "LLM_API_KEY",
    "CHAIN_CFG_JSON", "CHAIN_RPC", "CHAIN_PRIVKEY", "POLICY_URI",
    "APP_API_KEY", "WS_API_KEY", "GOVERNANCE_API_KEY", "CHAIN_API_KEY",
    "PHASE_API_KEY", "MCP_API_KEY",
    "AGRIBRAIN_ALLOW_DIRTY", "PYTHONPATH", "PYTHONHOME",
}
SUPPORTED_PYTHON_MINORS = {(3, 11)}


def errors_for_environment(environ: dict[str, str]) -> list[str]:
    errors = []
    for name, expected in EXPECTED.items():
        actual = environ.get(name)
        if actual != expected:
            errors.append(f"{name}: expected {expected!r}, got {actual!r}")
    for name in sorted(MUST_BE_UNSET):
        if name in environ:
            errors.append(f"{name}: must be unset, got {environ[name]!r}")
    return errors


def interpreter_error(version_info=None) -> str | None:
    version = tuple((version_info or sys.version_info)[:2])
    if version not in SUPPORTED_PYTHON_MINORS:
        allowed = ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_PYTHON_MINORS))
        return f"Python {version[0]}.{version[1]} is not lock-verified; use {allowed}"
    return None


def main() -> int:
    errors = errors_for_environment(dict(os.environ))
    version_error = interpreter_error()
    if version_error:
        errors.append(version_error)
    if errors:
        print("BLOCK: non-canonical publication environment:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print("Canonical publication environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
