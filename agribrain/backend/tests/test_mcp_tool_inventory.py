"""MCP tool inventory drift guard.

The README advertises "14 statically registered tools and 5 additional
runtime role-capability tools". The inventory document at
``agribrain/backend/pirag/mcp/TOOL_INVENTORY.md`` lists each by name.
This test asserts the documented set matches what
``get_default_registry()`` plus ``register_role_capabilities()``
actually expose.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_INVENTORY = _REPO_ROOT / "agribrain" / "backend" / "pirag" / "mcp" / "TOOL_INVENTORY.md"


def _documented_static_tools() -> set[str]:
    text = _INVENTORY.read_text(encoding="utf-8")
    section = re.search(
        r"## Statically registered tools[\s\S]+?(?=\n## )", text
    )
    assert section, "TOOL_INVENTORY.md missing 'Statically registered tools' section"
    rows = re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`", section.group(0), re.MULTILINE)
    return set(rows)


def _documented_runtime_tools() -> set[str]:
    text = _INVENTORY.read_text(encoding="utf-8")
    section = re.search(
        r"## Runtime role-capability tools[\s\S]+?(?=\n## |\Z)", text
    )
    assert section, "TOOL_INVENTORY.md missing 'Runtime role-capability tools' section"
    rows = re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`", section.group(0), re.MULTILINE)
    return set(rows)


def test_documented_static_count_matches_readme():
    documented = _documented_static_tools()
    assert len(documented) == 14, (
        f"Documented static tool count drifted from 14: {len(documented)} -> {sorted(documented)}"
    )


def test_documented_runtime_count_matches_readme():
    documented = _documented_runtime_tools()
    assert len(documented) == 5, (
        f"Documented runtime tool count drifted from 5: {len(documented)} -> {sorted(documented)}"
    )


def test_documented_static_tools_match_registry():
    """Every documented static tool must be reachable in the registry.

    A discrepancy means either the doc is stale or the registry has
    drifted. The registry is the source of truth at runtime; doc the
    truth.
    """
    from pirag.mcp.registry import get_default_registry, mcp_registration_status

    documented = _documented_static_tools()
    get_default_registry()
    status = mcp_registration_status()
    registered = set(status["registered"])
    failed = set(status["failed"].keys())

    # `simulate` is conditionally registered (only when SIM_API_BASE is
    # set). The inventory acknowledges it as a known by-design absence
    # in CI/simulator contexts; allow it to appear in `failed`.
    expected_present = documented - {"simulate"}
    missing = expected_present - registered
    assert not missing, (
        "Documented static tools missing from registry: "
        f"{sorted(missing)}. Registered: {sorted(registered)}; failed: {sorted(failed)}"
    )

    # `simulate` must be in either registered or failed, never simply
    # absent (would mean register attempt was dropped).
    assert "simulate" in registered or "simulate" in failed, (
        "simulate not present in either registry.registered or registry.failed; "
        "the conditional-registration branch was bypassed."
    )


def test_documented_runtime_tools_appear_in_capabilities_module():
    """Every documented runtime tool must be defined in agent_capabilities.py."""
    documented = _documented_runtime_tools()
    cap_path = (
        _REPO_ROOT / "agribrain" / "backend" / "pirag" / "mcp" / "agent_capabilities.py"
    )
    text = cap_path.read_text(encoding="utf-8")
    missing = [t for t in documented if f'name="{t}"' not in text]
    assert not missing, (
        f"Runtime tools documented but not defined in agent_capabilities.py: {missing}"
    )
