"""Fail closed when the active retrieval corpus drifts from the benchmark."""
from __future__ import annotations

import re
from pathlib import Path

from src.models.action_selection import (
    RHO_RECOVERY_KNEE,
    THERMAL_DELTA_MAX,
    THERMAL_T0,
)
from src.models.carbon import REFRIG_COP_PENALTY
from src.models.footprint import (
    DEFAULT_ASSUMED_ACTIVE_POWER_W,
    DEFAULT_ENERGY_PER_PROXY_STEP_J,
    DEFAULT_WATER_PER_PROXY_STEP_L,
    DEFAULT_WATER_RATE_L_PER_SERVER_SECOND,
)
from src.models.policy import Policy
from src.models.slca import _ACTION_BASES

from pirag.mcp.tools.footprint_query import CO2_PER_KWH_PROXY


KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"
EXPECTED_IDS = {
    "animal_feed_diversion_standards",
    "blockchain_audit_requirements",
    "carbon_accounting_transport",
    "composting_bioenergy_requirements",
    "cooperative_governance_policy",
    "cyber_outage_contingency",
    "demand_volatility_response",
    "emergency_rerouting_sop",
    "green_ai_reporting",
    "heatwave_contingency_plan",
    "iot_sensor_spec",
    "redistribution_food_bank_protocol",
    "regulatory_fda_leafy_greens",
    "slca_community_resilience_metrics",
    "slca_guidelines",
    "slca_labor_fairness_standards",
    "slca_price_transparency_framework",
    "sop_cold_chain",
    "temperature_excursion_protocol",
    "waste_hierarchy_protocol",
}
SOURCE_SCOPE = (
    "Source scope: Constructed institutional-retrieval input for the synthetic "
    "AGRI-BRAIN benchmark."
)


def _text(name: str) -> str:
    return (KB_DIR / name).read_text(encoding="utf-8")


def test_corpus_inventory_and_source_scope_are_fixed() -> None:
    files = sorted(KB_DIR.glob("*.txt"))
    assert {path.stem for path in files} == EXPECTED_IDS
    assert len(files) == 20
    for path in files:
        first_lines = "\n".join(path.read_text(encoding="utf-8").splitlines()[:5])
        assert SOURCE_SCOPE in first_lines, path.name


def test_social_priors_equal_the_executable_table() -> None:
    mapping = {
        "L": "slca_labor_fairness_standards.txt",
        "R": "slca_community_resilience_metrics.txt",
        "P": "slca_price_transparency_framework.txt",
    }
    for component, filename in mapping.items():
        observed = {
            action: float(value)
            for action, value in re.findall(
                rf"- (cold_chain|local_redistribute|recovery): {component} = ([0-9.]+)",
                _text(filename),
            )
        }
        expected = {
            action: float(values[component])
            for action, values in _ACTION_BASES.items()
        }
        assert observed == expected


def test_carbon_and_risk_notes_equal_executable_defaults() -> None:
    policy = Policy()
    carbon = _text("carbon_accounting_transport.txt")
    assert f"(T_C - {THERMAL_T0:.1f}) / {THERMAL_DELTA_MAX:.1f}" in carbon
    assert f"0.12 * 1.0 * (1 + {REFRIG_COP_PENALTY:.2f}*theta)" in carbon
    assert f"{policy.km_coldchain:g} km for cold_chain" in carbon
    assert f"{policy.km_local:g} km for" in carbon
    assert f"{policy.km_recovery:g} km for recovery" in carbon
    assert f"E/{policy.carbon_cap:g}" in carbon

    feed = _text("animal_feed_diversion_standards.txt")
    assert f"rho greater than {RHO_RECOVERY_KNEE:.2f}" in feed


def test_green_ai_note_equal_executable_defaults_and_boundary() -> None:
    green_ai = _text("green_ai_reporting.txt")
    assert f"elapsed_seconds * {DEFAULT_ASSUMED_ACTIVE_POWER_W:g} W" in green_ai
    assert (
        f"elapsed_seconds * {DEFAULT_WATER_RATE_L_PER_SERVER_SECOND:g} L "
        "per server-second"
    ) in green_ai
    assert f"proxies of {DEFAULT_ENERGY_PER_PROXY_STEP_J:g} joule" in green_ai
    assert f"and {DEFAULT_WATER_PER_PROXY_STEP_L:g} litre" in green_ai
    assert f"declared {CO2_PER_KWH_PROXY:g} kg CO2-equivalent per kWh" in green_ai
    assert "not be described as neural forward-pass counts" in green_ai
    assert "does not measure cloud-versus-edge performance" in green_ai


def test_removed_fabrications_do_not_reenter_corpus() -> None:
    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(KB_DIR.glob("*.txt"))
    )
    forbidden = (
        "Regulatory Basis:",
        "REQUIRED ACTIONS",
        "LCL-500",
        "AFDC-300",
        "OOL-600",
        "RTF-200",
        "CBDM-400",
        "quorum of 3 agents",
        "derived from EPA emission factors",
        "price transparency audit for any routing decision",
        "must be queued locally and batch-submitted",
        "edge example and 200 millijoules",
        "400 millijoules per step",
    )
    for phrase in forbidden:
        assert phrase not in corpus
